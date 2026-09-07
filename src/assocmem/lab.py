"""Student-facing API for the scaling-law tutorial.

Everything a student needs is four objects::

    lab = Lab("alice")                                  # budget + round counter
    s   = Sweep(c=[4e9], n=[64, 128, 256], lr=[0.03, 0.06])
    s.estimate(lab)                                     # free: what would this cost?
    r   = lab.run_round("lr landscape", s)              # spends 1 round, plots itself
    #   fit the laws yourself, from `assocmem.fit`: isoflop_optimum, powerlaw, joint_fit
    lab.hero(c=lab.compute_left(), n=..., lr=..., predicted=...)     # one shot

The interesting machinery -- lazy Zipf data, hashed embeddings, vmapped training,
flop accounting -- stays out of sight in `data`/`train`/`ledger`.
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fit as _fit
from . import ledger
from .data import D_OUT
from .grid import strat_evalset
from .problem import get_stream
from .train import BATCH, evaluate, eval_flops, plan_cost, train_flops, train_sweep

FLOPS_PER_TOKEN_PARAM = 6.0
DEFAULT_BUDGET = 1e13
DEFAULT_ROUNDS = 3
# Eval sets are *stratified*, not drawn from p(x): the head is taken exactly with its
# true weights p(i) and the tail is log-binned, sampled and reweighted by each bin's
# exact mass (see assocmem.grid).  At an identical billed cost of 4096 contexts that is
# ~5x less noisy than sampling eval tokens -- 0.006 nats of spread instead of 0.029 --
# which matters because the IsoFLOP minimum students fit is itself only ~0.03 deep.
SCREEN_STRAT = dict(head=1024, per_bin=64, per_decade=4, seed=11)
HERO_STRAT = dict(head=4096, per_bin=512, per_decade=8, seed=11)
CHECK_STRAT = dict(head=4096, per_bin=512, per_decade=8, seed=23)  # resampled tail

EVAL_TOKENS = 4096  # padded size of SCREEN_STRAT; every run is scored on it
HERO_EVAL_TOKENS = 34816  # padded size of HERO_STRAT
HERO_CHECK_TOKENS = 34816
HERO_CURVE_POINTS = 8
GFLOPS_GUESS = 350.0  # accounted flops/s, refined from the lab's own history

# Where lab.report() sends a student, and which answer of that form each number
# belongs to -- "Tutorial scaling laws MLSS", whose three questions are Predicted
# loss / Obtained loss / Fraction spent on hero run.  It deliberately does NOT ask
# for a name: the deck plots the room as an anonymous cloud of dots, and the way
# to keep it that way is to never collect the names in the first place.  The
# scoreboard slide reads this same form's responses sheet
# (figures/hero-board.md), so the two have to name the same form.  Nothing breaks
# if this is blanked out: report() then just prints the numbers to type in.
#
# The ids are the form's own field names.  They are NOT in the page's markup as
# `entry.<digits>` -- a live Google Form carries them inside its
# `FB_PUBLIC_LOAD_DATA_` blob -- so the two ways to get them are the prefill
# dialog (three dots > "Get pre-filled link", fill in dummies, read the link) or,
# from the blob, the fifth element of each question entry.
FORM_URL = ("https://docs.google.com/forms/d/e/"
            "1FAIpQLSdbUs-b8SNnQa0Ex5ckXchFdIhL99hpyXQcsyNXFaeEH9NM5A/viewform")
FORM_FIELDS = {"predicted": "entry.938226873",
               "actual": "entry.1247755212",
               "share": "entry.1372306388"}
COMPILE_S = 0.45  # per distinct (n, steps) group


class BudgetError(RuntimeError):
    """Raised instead of spending flops or rounds you do not have."""


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """One run.  ``params`` is the size axis the student sets: the PARAMETER count N.

    The model is a ``D_OUT x width`` matrix, so ``width = params / D_OUT`` is the embedding
    dimension the trainer actually needs, and a parameter count is only realisable when it is a
    whole number of embedding dimensions (``Sweep`` rounds it).
    """

    params: int
    steps: int
    lr: float
    init: float = 0.0
    seed: int = 0
    c: float = 0.0  # the compute rung this config belongs to (label, for plots)

    @property
    def width(self) -> int:
        """Embedding dimension: the parameter count divided by the vocabulary size."""
        return self.params // D_OUT

    @property
    def key(self) -> str:
        return f"n{self.width}-s{self.steps}-lr{self.lr:.6g}-i{self.init:.4g}-z{self.seed}"

    @property
    def group(self) -> tuple:
        """Configs sharing this can be trained in one vmapped call."""
        return (self.width, self.steps, self.seed)

    @property
    def tokens(self) -> int:
        return self.steps * BATCH

    @property
    def flops(self) -> float:
        return train_flops(self.width, self.steps)


def _listify(x):
    if x is None:
        return None
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    return [float(v) for v in a]


def _lr_of(fn, c: float, n: int, d: int) -> float:
    """Call an lr function with as many of (c, n, d) as it accepts.

    A rule may only need one of them -- an lr law fitted against compute takes ``c`` alone --
    so the arity is read off the signature rather than forced on the caller; anything variadic
    gets all three.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return float(fn(c, n, d))
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return float(fn(c, n, d))
    k = sum(p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params)
    return float(fn(*(c, n, d)[:max(1, min(k, 3))]))


class Sweep:
    """A cartesian product of things to try.

    Give exactly **two** of the three axes and the third follows from C = 6ND:

    * ``c``  flops per run
    * ``n``  parameters (N); the model is 512 x (n/512), so n is rounded to a whole
      embedding dimension
    * ``d``  tokens per run; rounded to a whole number of steps of ``BATCH`` = 64 tokens

    ::

        Sweep(c=[4e9, 1.2e10], n=[16_384, 65_536], lr=[0.03, 0.06])   # d derived: IsoFLOP
        Sweep(c=[4e9], d=[100_000, 400_000], lr=[0.05])               # n derived
        Sweep(n=[65_536], d=[100_000, 400_000], lr=[0.05])            # c derived

    ``lr`` is a list of values, or a **function** evaluated per config: it is called with as
    many of ``(c, n, d)`` as it takes, so ``lambda c, n, d: 0.001 * n`` is a size-dependent
    rule and ``lambda c: 0.08 * (c / 1e10) ** -0.05`` reuses an lr law you fitted yourself.

    Sweeps concatenate with ``+``, so an irregular design is still one round::

        Sweep(c=[3e11], n=[143_360], lr=[0.028]) + Sweep(c=[3e11], n=[286_720], lr=[0.016])
    """

    def __init__(self, lr=None, n=None, c=None, d=None, init=0.0, seed=0, _configs=None):
        if _configs is not None:
            self.configs = tuple(_configs)
            return
        given = {k: v for k, v in (("c", c), ("n", n), ("d", d)) if v is not None}
        if len(given) != 2:
            raise ValueError(
                "give exactly two of c= (flops per run), n= (parameters) and d= (tokens per "
                f"run); the third follows from C = 6ND. Got: {sorted(given) or 'none'}")
        axes = {k: _listify(v) for k, v in given.items()}
        lr_fn = lr if callable(lr) else None
        lrs = [None] if lr_fn is not None else _listify(lr)
        if lrs is None:
            raise ValueError("give lr= as a list of values or a function of (c, n, d)")
        if lr_fn is None and min(lrs) <= 0:
            raise ValueError("lr must be > 0")
        inits, seeds = _listify(init), [int(s) for s in _listify(seed)]

        (k1, v1), (k2, v2) = sorted(axes.items())
        cfgs, snapped = [], []
        for a, b, lr_, i_, z_ in itertools.product(v1, v2, lrs, inits, seeds):
            vals = {k1: a, k2: b}
            cc, nn, dd = vals.get("c"), vals.get("n"), vals.get("d")
            if cc is None:
                cc = _fit.compute_of(nn, dd)
            elif nn is None:
                nn = _fit.params_of(cc, dd)
            else:
                dd = _fit.d_of(cc, nn)
            width, steps = int(round(nn / D_OUT)), int(round(dd / BATCH))
            if width < 1:
                raise ValueError(
                    f"n={nn:.4g} parameters is less than one embedding dimension "
                    f"({D_OUT} parameters)"
                    + (f" -- at C={cc:.3g} that is what d={dd:.0f} tokens buys" if "d" in vals
                       else ""))
            if steps < 1:
                raise ValueError(
                    f"d={dd:.4g} tokens is less than one batch of {BATCH}"
                    + (f" -- C={cc:.3g} is too small for n={nn:.4g} parameters; use "
                       f"n <= {int(cc / (6.0 * BATCH))} at this C, or raise C"
                       if "c" in vals and "n" in vals else ""))
            params, tokens = width * D_OUT, steps * BATCH
            if abs(params - nn) > 0.5:
                snapped.append(("n", nn, params))
            if abs(tokens - dd) > 0.5:
                snapped.append(("d", dd, tokens))
            # With c given, the requested rung stays the label, so an IsoFLOP profile groups
            # cleanly even though rounding moves each run's true cost by a hair.
            c_label = float(cc) if "c" in vals else _fit.compute_of(params, tokens)
            if lr_ is None:
                lr_ = _lr_of(lr_fn, c_label, params, tokens)
                if lr_ <= 0:
                    raise ValueError(f"the lr function returned {lr_:.4g} at "
                                     f"(c={c_label:.3g}, n={params}, d={tokens})")
            cfgs.append(Config(params, steps, lr_, i_, z_, c=c_label))
        for axis in ("n", "d"):
            rounded = [(v, w) for k, v, w in snapped if k == axis]
            if rounded:
                worst = max(rounded, key=lambda p: abs(p[1] / p[0] - 1))
                unit = f"whole embedding dimensions of {D_OUT} parameters" if axis == "n" \
                    else f"whole batches of {BATCH} tokens"
        self.configs = tuple(cfgs)

    # -- composition ---------------------------------------------------------
    def __add__(self, other: "Sweep") -> "Sweep":
        return Sweep(None, None, _configs=self.configs + other.configs)

    def __len__(self) -> int:
        return len(self.configs)

    def __repr__(self) -> str:
        cs = sorted({cfg.c for cfg in self.configs})
        return (f"Sweep({len(self.configs)} runs, {len(self.groups())} groups, "
                f"C rungs {[f'{v:.3g}' for v in cs]})")

    def groups(self) -> dict:
        g: dict = {}
        for cfg in self.configs:
            g.setdefault(cfg.group, []).append(cfg)
        return g

    # -- costing -------------------------------------------------------------
    def cost(self, eval_tokens: int = EVAL_TOKENS) -> float:
        return sum(plan_cost(n, s, len(v), eval_tokens, 1)
                   for (n, s, _), v in self.groups().items())

    def estimate(self, lab: "Lab" = None, eval_tokens: int | None = None,
                 quiet: bool = False):
        """What this sweep would cost.  Free -- runs nothing."""
        if eval_tokens is None:
            eval_tokens = lab.eval_tokens if lab is not None else EVAL_TOKENS
        est = Estimate(self, lab, eval_tokens)
        if not quiet:
            print(est)
        return est


@dataclass
class Estimate:
    sweep: Sweep
    lab: "Lab" = None
    eval_tokens: int = EVAL_TOKENS

    @property
    def flops(self) -> float:
        return self.sweep.cost(self.eval_tokens)

    @property
    def seconds(self) -> float:
        gf = self.lab.gflops if self.lab else GFLOPS_GUESS
        return self.flops / (gf * 1e9) + COMPILE_S * len(self.sweep.groups())

    @property
    def fits(self) -> bool:
        return self.lab is None or self.flops <= self.lab.remaining

    def breakdown(self) -> list[tuple[float, float, int]]:
        """[(C rung, flops, n_runs)] sorted by cost, so it is obvious what to cut."""
        by: dict = {}
        for (n, s, _), v in self.sweep.groups().items():
            by.setdefault(v[0].c, [0.0, 0])
            by[v[0].c][0] += plan_cost(n, s, len(v), self.eval_tokens, 1)
            by[v[0].c][1] += len(v)
        return sorted(((c, f, k) for c, (f, k) in by.items()), key=lambda t: -t[1])

    def __str__(self) -> str:
        L = [f"{len(self.sweep)} runs in {len(self.sweep.groups())} groups  ->  "
             f"{self.flops:.3g} flops, ~{self.seconds:.0f} s"]
        if self.lab is not None:
            pct = 100 * self.flops / self.lab.budget
            L.append(f"  {pct:.1f}% of the {self.lab.budget:.2g} budget; "
                     f"{self.lab.remaining:.3g} left ({self.lab.rounds_left} rounds)")
            L.append("  FITS" if self.fits else
                     f"  DOES NOT FIT -- {self.flops / self.lab.remaining:.2f}x too big")
        if len(self.breakdown()) > 1:
            L.append("  by C rung: " + ",  ".join(
                f"{c:.3g}: {f:.3g} ({k} runs)" for c, f, k in self.breakdown()))
        return "\n".join(L)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Results:
    rows: list[dict]
    name: str = ""
    flops: float = 0.0
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, **eq) -> "Results":
        keep = [r for r in self.rows if all(np.isclose(r[k], v) for k, v in eq.items())]
        return Results(keep, self.name)

    def best(self, **eq) -> dict:
        return min(self.select(**eq).rows, key=lambda r: r["loss"])

    @property
    def rungs(self) -> list[float]:
        return sorted({r["c"] for r in self.rows})

    def isoflop(self) -> list[dict]:
        """Per-rung IsoFLOP optimum (n*, L*) from a parabola in log n."""
        out = []
        for c in self.rungs:
            sub = self.select(c=c)
            ns = sorted({r["n"] for r in sub.rows})
            if len(ns) < 3:
                continue
            loss = [min(r["loss"] for r in sub.rows if r["n"] == n) for n in ns]
            f = _fit.isoflop_optimum(ns, loss)
            out.append(dict(c=c, n_star=f.n_star, loss_star=f.loss_star, ns=ns,
                            loss=loss, clipped=f.clipped))
        return out

    def envelope(self, keep=("c", "n")) -> "Results":
        """The best run in every ``keep`` cell -- i.e. the minimum over everything else.

        Fitting anything to raw rows mixes in runs that only did badly because their
        learning rate was wrong; a law wants the envelope over lr, one point per (C, n)
        cell.  ``plot_runs(..., reduce='min')`` draws exactly this.
        """
        cells: dict[tuple, dict] = {}
        for r in self.rows:
            k = tuple(r[c] for c in keep)
            if k not in cells or r["loss"] < cells[k]["loss"]:
                cells[k] = r
        return Results([cells[k] for k in sorted(cells)], name=f"{self.name} (envelope)")

    def lr_optima(self) -> list[dict]:
        """Per-(C, n) cell: where the lr optimum is, and whether the grid bracketed it.

        A cell's loss-vs-lr curve is a parabola in log lr, so three or more lrs locate the
        optimum.  ``where`` is 'interior' when the parabola's vertex falls inside the
        swept range -- the only case in which the cell's best loss is an estimate of the
        cell's *achievable* loss rather than an upper bound on it.  Cells whose best lr
        sits at an edge ('low'/'high') are reported so they can be widened next round
        instead of being read as measurements.
        """
        out = []
        for c in self.rungs:
            for n in sorted({r["n"] for r in self.select(c=c).rows}):
                sub = sorted(self.select(c=c, n=n).rows, key=lambda r: r["lr"])
                lrs = [r["lr"] for r in sub]
                losses = [r["loss"] for r in sub]
                i = int(np.argmin(losses))
                rec = dict(c=c, n=n, lrs=lrs, losses=losses, lr_best=lrs[i],
                           loss_best=losses[i], lr_star=lrs[i], where="single")
                if len(lrs) >= 3:
                    co = np.polyfit(np.log(lrs), losses, 2)
                    if co[0] > 0:
                        v = float(np.exp(-co[1] / (2 * co[0])))
                        rec["lr_star"] = v
                        rec["where"] = ("low" if v < min(lrs) else
                                        "high" if v > max(lrs) else "interior")
                        rec["loss_star"] = float(np.polyval(co, np.log(rec["lr_star"])))
                    else:
                        rec["where"] = "flat"
                elif len(lrs) == 2:
                    rec["where"] = "low" if i == 0 else "high"
                out.append(rec)
        return out

    def table(self, sort: str = "loss") -> str:
        rows = sorted(self.rows, key=lambda r: r[sort])
        L = [f"{'C':>10} {'N':>10} {'D':>9} {'lr':>8} {'loss':>8}"]
        for r in rows:
            L.append(f"{r['c']:10.3g} {r['n']:10,} {r['tokens']:9,.0f} {r['lr']:8.4g} "
                     f"{r['loss']:8.4f}")
        return "\n".join(L)

    @property
    def df(self):
        import pandas as pd  # optional; only if the student wants a dataframe

        return pd.DataFrame(self.rows)

    def plot(self, path=None, show=None):
        from .plots import plot_round

        return plot_round(self, path=path, show=show)


# --------------------------------------------------------------------------- #
# Lab
# --------------------------------------------------------------------------- #
class Lab:
    """Budget + round accounting for one student.

    State lives in ``runs/<name>/`` so a kernel restart does not reset (or refund)
    anything.  Repeating an identical run is served from cache: free, and it does
    not burn a round.
    """

    def __init__(self, name: str = "me", budget: float = DEFAULT_BUDGET,
                 rounds: int = DEFAULT_ROUNDS, root: str | Path | None = None,
                 eval_tokens: int = EVAL_TOKENS, quiet: bool = False,
                 deterministic: bool = False,
                 hero_curve_points: int = HERO_CURVE_POINTS,
                 hero_eval_tokens: int = HERO_EVAL_TOKENS,
                 hero_check_tokens: int = HERO_CHECK_TOKENS):
        self.name = name
        # anchored at the project root, so a notebook and a script share one lab
        self.dir = Path(root or os.environ.get("ASSOCMEM_RUNS")
                        or Path(__file__).resolve().parents[2] / "runs") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.eval_tokens = eval_tokens
        self.hero_curve_points = hero_curve_points
        self.hero_eval_tokens = hero_eval_tokens
        self.hero_check_tokens = hero_check_tokens
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        # The variant is a property of the lab, not of a call: a lab that has already
        # spent flops on one set of labels cannot be compared against the other, so the
        # stored value wins and a mismatch is refused rather than silently honoured.
        self.deterministic = bool(st.get("deterministic", deterministic))
        if st and self.deterministic != bool(deterministic):
            raise BudgetError(
                f"lab '{name}' already holds runs on the "
                f"{'one-correct-answer' if self.deterministic else 'entropy'} variant; "
                f"asking for the other one would make its rows incomparable. Use a "
                f"different lab name, or delete {self.dir}.")
        self.budget = float(st.get("budget", budget))
        self.max_rounds = int(st.get("max_rounds", rounds))
        self.rounds_used = int(st.get("rounds_used", 0))
        self.rows: list[dict] = st.get("rows", [])
        self.round_log: list[dict] = st.get("round_log", [])
        self.hero_record = st.get("hero")
        self.gflops = float(st.get("gflops", GFLOPS_GUESS))
        if st and (abs(self.budget - budget) > 1 or self.max_rounds != rounds):
            print(f"note: reusing the existing lab '{name}' (budget {self.budget:.3g}, "
                  f"{self.max_rounds} rounds); delete {self.dir} to start over.")
        ledger.configure(self.dir / "ledger.jsonl", self.budget)
        self._evals = None
        if not quiet:
            print(self.status())

    # -- state ---------------------------------------------------------------
    def _save(self):
        self.state_path.write_text(json.dumps(dict(
            budget=self.budget, max_rounds=self.max_rounds, rounds_used=self.rounds_used,
            deterministic=self.deterministic,
            rows=self.rows, round_log=self.round_log, hero=self.hero_record,
            gflops=self.gflops), indent=1, default=float))

    @property
    def spent(self) -> float:
        return ledger.total()["total"]

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    @property
    def rounds_left(self) -> int:
        return self.max_rounds - self.rounds_used

    @property
    def results(self) -> Results:
        return Results(list(self.rows), name="all rounds")

    @property
    def evals(self):
        if self._evals is None:
            self._evals = strat_evalset(**SCREEN_STRAT, deterministic=self.deterministic)
        return self._evals

    def reset(self, confirm: bool = False) -> None:
        """Wipe this lab: refunds the whole budget and all rounds.  Use between attempts."""
        if not confirm:
            raise BudgetError("reset() throws away every run in this lab. "
                              "Call lab.reset(confirm=True) if you mean it.")
        for f in ("state.json", "ledger.jsonl", "hero_W.npy"):
            (self.dir / f).unlink(missing_ok=True)
        self.rounds_used, self.rows, self.round_log, self.hero_record = 0, [], [], None
        self._save()
        print(f"lab '{self.name}' reset.  " + self.status())

    def status(self) -> str:
        return (f"lab '{self.name}'"
                f"{' [one correct answer]' if self.deterministic else ''}:  "
                f"{self.spent:.4g} / {self.budget:.3g} flops spent "
                f"({100 * self.spent / self.budget:.1f}%),  {self.remaining:.4g} left"
                f"  |  rounds {self.rounds_used}/{self.max_rounds}"
                f"  |  {len(self.rows)} runs recorded"
                f"{'  |  HERO DONE' if self.hero_record else ''}")

    # -- running -------------------------------------------------------------
    def _refuse(self, est: Estimate, what: str):
        f = est.flops / max(self.remaining, 1.0)
        msg = [f"{what} would cost {est.flops:.4g} flops but only {self.remaining:.4g} "
               f"remain ({f:.2f}x too big).", "Options:"]
        bd = est.breakdown()
        if len(bd) > 1:
            msg.append(f"  - drop the C={bd[0][0]:.3g} rung, which alone costs "
                       f"{bd[0][1]:.3g} ({100 * bd[0][1] / est.flops:.0f}% of the sweep)")
        n_lr = len({c.lr for c in est.sweep.configs})
        if n_lr > 1 and int(n_lr / f) < n_lr:
            msg.append(f"  - cut the lr grid from {n_lr} to "
                       f"{max(1, int(n_lr / f))} values")
        if f > 1.05:
            msg.append(f"  - or scale every C down by {1 / f:.2f}x")
        else:
            order = sorted(est.sweep.configs, key=lambda c: -c.flops)
            drop, acc = 0, 0.0
            while acc < est.flops - self.remaining and drop < len(order):
                acc += order[drop].flops
                drop += 1
            msg.append(f"  - or drop the {drop} most expensive of these "
                       f"{len(est.sweep)} runs (it is only just too big)")
        msg.append("  cost breakdown by rung: " + ",  ".join(
            f"C={c:.3g}: {fl:.3g}" for c, fl, _ in bd))
        raise BudgetError("\n".join(msg))

    def run_round(self, name: str, sweep: Sweep, plot: bool = True) -> Results:
        """Train every config in `sweep`.  Spends one round.

        Configs sharing (n, steps) are trained in a single vmapped call, all on the
        same data stream, so comparisons between them are far less noisy than the
        absolute losses.

        Nothing here is a dry run, because nothing needs to be: a malformed sweep raises
        when it is built, ``sweep.estimate(lab)`` prices it for free, and a sweep that does
        not fit is refused below before anything is trained.
        """
        # cache first: re-running a cell must never cost flops or a round
        done = {r["key"] for r in self.rows}
        todo = [c for c in sweep.configs if c.key not in done]
        cached = len(sweep.configs) - len(todo)
        if not todo:
            print(f"all {cached} configs already run -- served from cache, "
                  f"no flops and no round spent.")
            out = Results([r for r in self.rows
                           if r["key"] in {c.key for c in sweep.configs}], name=name)
            if plot:
                out.plot(show=None)
            return out
        if self.rounds_left <= 0:
            raise BudgetError(
                f"no screening rounds left ({self.rounds_used}/{self.max_rounds} used), "
                f"and {len(todo)} of these {len(sweep)} configs are new. "
                f"Fit your laws on what you have and spend the remaining "
                f"{self.remaining:.3g} flops on lab.hero().")

        sub = Sweep(None, None, _configs=todo)
        est = Estimate(sub, self, self.eval_tokens)
        if not est.fits:
            self._refuse(est, "that sweep")
        print(f"round '{name}': {len(todo)} runs"
              + (f" ({cached} cached)" if cached else "")
              + f" in {len(sub.groups())} groups, {est.flops:.3g} flops, "
                f"~{est.seconds:.0f} s")

        stream = get_stream(max(c.tokens for c in todo),
                            deterministic=self.deterministic)
        t0 = time.time()
        new: list[dict] = []
        groups = sorted(sub.groups().items(), key=lambda kv: kv[1][0].flops * len(kv[1]))
        for i, ((n, steps, seed), cfgs) in enumerate(groups, 1):
            r = train_sweep(n=n, steps=steps, lrs=[c.lr for c in cfgs],
                            init_scales=[c.init for c in cfgs], stream=stream,
                            eval_set=self.evals, eval_tokens=self.eval_tokens,
                            instance_seed=seed, tag=f"round{self.rounds_used + 1}-{name}")
            for c, loss in zip(cfgs, r.loss):
                new.append(dict(key=c.key, round=self.rounds_used + 1, round_name=name,
                                c=c.c, n=c.params, width=c.width, steps=c.steps,
                                tokens=c.tokens, lr=c.lr,
                                init=c.init, seed=c.seed, loss=float(loss),
                                flops=c.flops))
            print(f"  [{i}/{len(groups)}] N={n * D_OUT:9,} D={steps * BATCH:9,}  "
                  f"best lr={r.best()['lr']:.4g} loss={r.best()['loss']:.4f}")
        dt = time.time() - t0

        self.rows += new
        self.rounds_used += 1
        self.gflops = 0.5 * self.gflops + 0.5 * est.flops / max(dt, 1e-3) / 1e9
        self.round_log.append(dict(name=name, round=self.rounds_used, flops=est.flops,
                                   seconds=dt, n_runs=len(new)))
        self._save()

        out = Results(new, name=name, flops=est.flops, seconds=dt)
        b = out.best()
        print(f"\nbest this round: N={b['n']:,} D={b['tokens']:,} lr={b['lr']:.4g}"
              f"  ->  loss {b['loss']:.4f}   ({dt:.0f} s)")
        print(self.status())
        if plot:
            out.plot(path=self.dir / f"round{self.rounds_used}_{name.replace(' ', '_')}.png",
                     show=None)
        return out

    # -- hero ----------------------------------------------------------------
    def _hero_eval_flops(self, width: int) -> float:
        """What the hero's own evaluations cost: the learning curve plus two final scorings."""
        return (eval_flops(width, self.eval_tokens) * self.hero_curve_points
                + eval_flops(width, self.hero_eval_tokens)
                + eval_flops(width, self.hero_check_tokens))

    def compute_left(self, n: int | None = None) -> float:
        """Flops available to *train* the hero run: what is left, less its evaluations.

        The hero is scored on a much larger eval set than a screening run, twice, plus a
        learning curve, and none of that is free -- so the number to hand to ``hero(c=...)``
        is not ``lab.remaining``.  The reserve scales with the model, so pass the ``n`` you
        are about to use for the exact figure; without one it is priced at the largest model
        run so far.  Either way ``hero`` trims the run to what fits, so an estimate here
        costs a few steps at worst and never an error.
        """
        widths = [int(r.get("width", r["n"] / D_OUT)) for r in self.rows]
        width = (max(1, int(round(float(n) / D_OUT))) if n is not None
                 else (max(widths) if widths else 1))
        return max(0.0, self.remaining - self._hero_eval_flops(width))

    def hero(self, c=None, n=None, lr=None, *, d=None, predicted=None) -> dict:
        """Spend what is left on one run, at the recipe you fitted.  One shot only.

        Give ``lr`` and any **two** of ``c`` (flops to train on), ``n`` (parameters) and
        ``d`` (tokens); the third follows from C = 6ND, the same way a ``Sweep`` is given.
        ``lab.compute_left()`` is the ``c`` that fits.

        ``predicted`` is what YOUR law says this run will score.  It is not needed to
        train, but it is the number the scoreboard plots against the result, and writing it
        down before the run is what makes it a prediction -- so pass it.
        """
        if lr is None:
            raise BudgetError("hero needs a learning rate: hero(c=..., n=..., lr=...)")
        given = {k: v for k, v in dict(c=c, n=n, d=d).items() if v is not None}
        if len(given) < 2:
            raise BudgetError(
                f"hero needs two of c, n, d (it got {sorted(given) or 'none'}): the third "
                f"follows from C = 6ND.  Try hero(c=lab.compute_left(), n=..., lr=...).")
        if "n" in given:
            width = max(1, int(round(float(n) / D_OUT)))
        else:
            width = max(1, int(round(_fit.params_of(float(c), float(d)) / D_OUT)))
        n_par = width * D_OUT
        if "d" in given:
            steps = max(1, int(round(float(d) / BATCH)))
        else:
            steps = _fit.steps_for(float(c), width)
        if len(given) == 3:
            want = _fit.compute_of(n_par, steps * BATCH)
            if abs(want / float(c) - 1) > 0.02:
                raise BudgetError(
                    f"c, n and d disagree: 6ND = {want:.4g} for the n and d you gave, but "
                    f"c={float(c):.4g}.  Give any two and let the third follow.")

        if self.hero_record is not None:
            # re-running the cell is fine; asking for a *different* hero run is not
            rec = self.hero_record
            same = (rec["n"] == n_par and rec["steps"] == steps
                    and abs(float(lr) / rec["lr_max"] - 1) < 1e-6)
            if same:
                print(f"hero run already done -- replaying it (no flops spent).\n"
                      f"  n={rec['n']:,} params, d={rec['tokens']:,}, "
                      f"lr={rec['lr_max']:.5f}\n"
                      f"  ACTUAL {rec['loss']:.4f}"
                      + (f"   (predicted {rec['predicted']:.4f}, error "
                         f"{rec['loss'] - rec['predicted']:+.4f})"
                         if np.isfinite(rec.get("predicted", np.nan)) else ""))
                self.report(predicted=predicted)
                return rec
            raise BudgetError(
                f"the hero run has already been done: n={rec['n']:,} params, "
                f"d={rec['tokens']:,}, "
                f"lr={rec['lr_max']:.5f}, loss {rec['loss']:.4f}.\nYou only get one shot -- "
                f"this call asks for n={n_par:,}, d={steps * BATCH:,}, lr={float(lr):.5f} "
                f"instead.  Start a fresh Lab(name=...) if you want another attempt.")

        ev = self._hero_eval_flops(width)
        # A recipe that overshoots the remainder is trimmed rather than refused: the run is
        # the point, and a handful of steps is a cheaper correction than a raised exception
        # at the last cell of the notebook.
        afford = int((self.remaining - ev) / train_flops(width, 1))
        if afford < 1:
            raise BudgetError(
                f"nothing left for a hero run at n={n_par:,}: its evaluations alone cost "
                f"{ev:.3g} flops and only {self.remaining:.3g} remain.  A smaller n leaves "
                f"room -- lab.compute_left(n) prices it.")
        if steps > afford:
            print(f"trimmed the hero run from {steps} to {afford} steps: "
                  f"{train_flops(width, steps) + ev:.4g} flops would not fit in the "
                  f"{self.remaining:.4g} remaining (evals take {ev:.3g}).")
            steps = afford
        c_train = train_flops(width, steps)
        lr = float(lr)
        pred = float("nan") if predicted is None else float(predicted)
        print(f"HERO RECIPE at C={c_train:.4g} (+{ev:.3g} for evals)\n"
              f"  n = {n_par:,} params  ({D_OUT} x {width}) | d = {steps * BATCH:,} tokens "
              f"({steps} steps of {BATCH}) | lr = {lr:.5f} -> {lr / 10:.6f} cosine\n"
              + (f"  PREDICTED LOSS = {pred:.4f} nats" if np.isfinite(pred) else
                 "  no predicted loss given -- pass predicted=... to score the "
                 "extrapolation"), flush=True)

        r = train_sweep(n=width, steps=steps, lrs=[lr],
                        stream=get_stream(steps * BATCH,
                                          deterministic=self.deterministic),
                        eval_set=self.evals, eval_tokens=self.eval_tokens,
                        eval_points=self.hero_curve_points, instance_seed=0, tag="hero",
                        return_params=True)
        ea = strat_evalset(**HERO_STRAT, deterministic=self.deterministic)
        eb = strat_evalset(**CHECK_STRAT, deterministic=self.deterministic)
        exact_a, samp_a, ma = evaluate(r.params, ea, n=width, instance_seed=0, y_seed=11)
        exact_b, samp_b, mb = evaluate(r.params, eb, n=width, instance_seed=0, y_seed=12)
        ledger.log("hero-final-eval",
                   eval=eval_flops(width, ma) + eval_flops(width, mb), n=width)

        rec = dict(n=n_par, width=width, steps=steps, tokens=steps * BATCH, lr_max=lr,
                   lr_min=lr / 10, c_train=c_train, c_eval=ev, predicted=pred,
                   loss=float(exact_a[0]),
                   loss_sampled=float(samp_a[0]), loss_heldout_set=float(exact_b[0]),
                   curve_steps=[int(x) for x in r.curve_steps],
                   curve_loss=[float(x) for x in r.curve.ravel()])
        self.hero_record = rec
        self._save()
        np.save(self.dir / "hero_W.npy", np.asarray(r.params[0]))
        print(f"\n=== HERO RESULT ===\n"
              f"  ACTUAL loss   = {rec['loss']:.4f} nats  ({ma} held-out tokens)\n"
              + (f"  PREDICTED     = {pred:.4f}          error "
                 f"{rec['loss'] - pred:+.4f}\n" if np.isfinite(pred) else "")
              + f"  cross-checks  : {rec['loss_sampled']:.4f} (sampled-y CE), "
                f"{rec['loss_heldout_set']:.4f} (independent eval set)\n"
              + self.status())
        # Printed here rather than left to the student to ask for: the scoreboard
        # slide is only as good as the number of runs that make it onto it.
        self.report()
        return rec

    # -- reporting -------------------------------------------------------------
    def report(self, name: str | None = None, url: str | None = None,
               fields: dict | None = None, predicted: float | None = None) -> dict:
        """The numbers the room's scoreboard wants, and a link that carries them.

        The slide "How did the room do?" plots one dot per submission -- predicted
        loss across, obtained loss up, coloured by the share of the budget that
        went into the hero run -- and reads them out of the responses sheet of a
        Google Form.  Typing three floats off a screen is where that goes wrong,
        so this prints them together and, when the form is configured, a prefilled
        link that fills the answers in for you.  Submitting is still a click: the
        link opens the form, it does not post anything on your behalf.

        No name is asked for or sent -- `name` is accepted so a call that passes one
        still works, and then ignored.  The plot is a cloud of dots, so a second
        submission is a second dot rather than a correction -- submit once.

        `predicted` fills in (or overrides) what your law said this run would score,
        for the case where `hero` was called without it.

        `share` is the WHOLE cost of the hero run, its final evaluations included
        (the ~1e10 flops of `c_eval`), as a FRACTION of the lab's budget -- what is
        left over from screening, which is the quantity the deck's colour axis
        means.  A fraction rather than a percentage because that is what the form
        asks for ("e.g. 0.42; needs to be between 0 and 1"), and a prefilled link
        that fails the form's own validation is worse than no link; the printout
        gives the percentage too, since that is what a person reads.
        """
        rec = self.hero_record
        if rec is None:
            raise BudgetError("no hero run yet -- lab.hero(c=..., n=..., lr=...) first, "
                              "and report what it gives you.")
        if predicted is not None:
            rec["predicted"] = float(predicted)
            self._save()
        pred = float(rec.get("predicted", float("nan")))
        c_hero = rec["c_train"] + rec["c_eval"]
        out = dict(predicted=round(pred, 4) if np.isfinite(pred) else None,
                   actual=round(rec["loss"], 4),
                   share=round(c_hero / self.budget, 3))
        url = FORM_URL if url is None else url
        fields = FORM_FIELDS if fields is None else fields
        # A prefilled link with a blank in it is worse than no link, so it is only built
        # once all three answers exist.
        if url and all(fields.get(k) for k in out) and out["predicted"] is not None:
            query = urllib.parse.urlencode(
                dict([("usp", "pp_url")] + [(fields[k], out[k]) for k in out]))
            out["url"] = f"{url.split('?')[0]}?{query}"
        # Labelled the way the form's three questions are, and in their order, so
        # the block is read off rather than translated.
        print(f"\n=== REPORT YOUR RUN ===   (the form's three fields, in order)\n"
              f"  predicted loss           : "
              + (f"{out['predicted']:.4f}" if out["predicted"] is not None else
                 "-- none recorded: lab.report(predicted=<what your law said>)") + "\n"
              + f"  obtained loss            : {out['actual']:.4f}\n"
                f"  fraction on the hero run : {out['share']:.3f}"
                f"   ({100 * out['share']:.1f}% = {c_hero:.3g} of "
                f"{self.budget:.3g} flops)")
        if "url" in out:
            print(f"  -> or open this link, which arrives with all three filled in:\n"
                  f"     {out['url']}")
        elif url:
            print(f"  -> form: {url}")
        return out
