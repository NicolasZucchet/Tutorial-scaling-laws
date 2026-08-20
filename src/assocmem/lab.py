"""Student-facing API for the scaling-law tutorial.

Everything a student needs is four objects::

    lab = Lab("alice")                                  # budget + round counter
    s   = Sweep(c=[4e9], n=[64, 128, 256], lr=[0.03, 0.06])
    s.estimate(lab)                                     # free: what would this cost?
    r   = lab.run_round("lr landscape", s)              # spends 1 round, plots itself
    laws = lab.fit()                                    # power laws + .recipe(C)
    lab.hero(laws)                                      # one shot, sized to the remainder

The interesting machinery -- lazy Zipf data, hashed embeddings, vmapped training,
flop accounting -- stays out of sight in `data`/`train`/`ledger`.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import fit as _fit
from . import ledger
from .problem import get_evalset, get_stream
from .train import BATCH, evaluate, eval_flops, plan_cost, train_flops, train_sweep

FLOPS_PER_TOKEN_PARAM = 6.0
DEFAULT_BUDGET = 1e13
DEFAULT_ROUNDS = 3
EVAL_TOKENS = 4096  # screening eval set; every run is scored on the same tokens
HERO_EVAL_TOKENS = 65536
HERO_CHECK_TOKENS = 32768
HERO_CURVE_POINTS = 8
GFLOPS_GUESS = 350.0  # accounted flops/s, refined from the lab's own history
COMPILE_S = 0.45  # per distinct (n, steps) group


class BudgetError(RuntimeError):
    """Raised instead of spending flops or rounds you do not have."""


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    n: int
    steps: int
    lr: float
    init: float = 0.0
    seed: int = 0
    c: float = 0.0  # the compute rung this config belongs to (label, for plots)

    @property
    def key(self) -> str:
        return f"n{self.n}-s{self.steps}-lr{self.lr:.6g}-i{self.init:.4g}-z{self.seed}"

    @property
    def group(self) -> tuple:
        """Configs sharing this can be trained in one vmapped call."""
        return (self.n, self.steps, self.seed)

    @property
    def tokens(self) -> int:
        return self.steps * BATCH

    @property
    def flops(self) -> float:
        return train_flops(self.n, self.steps)


def _listify(x):
    if x is None:
        return None
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    return [float(v) for v in a]


class Sweep:
    """A cartesian product of things to try.

    Give the time axis as either ``c`` (flops per run, steps are derived) or
    ``steps``.  ``c`` is what you want for IsoFLOP scaling laws::

        Sweep(c=[4e9, 1.2e10], n=[64, 128, 256, 512], lr=[0.03, 0.06, 0.12])   # 24 runs
        Sweep(n=[512], steps=[1000, 4000], lr=[0.05])                          # 2 runs

    Sweeps concatenate with ``+``, so an irregular design is still one round::

        Sweep(c=[3e11], n=[280, 560, 1120], lr=[0.028]) \\
            + Sweep(c=[3e11], n=[560], lr=[0.016, 0.048])
    """

    def __init__(self, n, lr, c=None, steps=None, init=0.0, seed=0, _configs=None):
        if _configs is not None:
            self.configs = tuple(_configs)
            return
        if (c is None) == (steps is None):
            raise ValueError("give exactly one of c= (flops per run) or steps=")
        ns = [int(round(v)) for v in _listify(n)]
        lrs, inits, seeds = _listify(lr), _listify(init), [int(s) for s in _listify(seed)]
        if min(ns) < 1:
            raise ValueError("n must be >= 1")
        if min(lrs) <= 0:
            raise ValueError("lr must be > 0")

        cfgs = []
        times = _listify(c) if c is not None else _listify(steps)
        for t, nn, lr_, i_, z_ in itertools.product(times, ns, lrs, inits, seeds):
            if c is not None:
                st = _fit.steps_for(t, nn)
                if t / (_fit.FLOPS_PER_ND * nn * BATCH) < 1.0:
                    raise ValueError(
                        f"C={t:.3g} is too small for n={nn}: that is "
                        f"{t / (_fit.FLOPS_PER_ND * nn * BATCH):.2f} steps. "
                        f"Use n <= {int(t / (_fit.FLOPS_PER_ND * BATCH))} at this C, "
                        f"or raise C.")
                cfgs.append(Config(nn, st, lr_, i_, z_, c=float(t)))
            else:
                st = int(round(t))
                cfgs.append(Config(nn, st, lr_, i_, z_, c=train_flops(nn, st)))
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

    def table(self, sort: str = "loss") -> str:
        rows = sorted(self.rows, key=lambda r: r[sort])
        L = [f"{'C':>10} {'n':>6} {'steps':>7} {'lr':>8} {'loss':>8}"]
        for r in rows:
            L.append(f"{r['c']:10.3g} {r['n']:6d} {r['steps']:7d} {r['lr']:8.4g} "
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
# Laws
# --------------------------------------------------------------------------- #
@dataclass
class Laws:
    l_inf: float
    rungs: list[dict]
    n_law: tuple  # (a, b, r2)    n*  = a C^b
    lr_law: tuple  # (a, p)        lr* = a C^p
    loss_law: tuple  # (a, alpha, r2)  L*  = l_inf + a C^-alpha
    loss_law_free: tuple  # (l_inf, a, alpha)
    lr_anchors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def n_star(self, c: float) -> float:
        a, b, _ = self.n_law
        return a * c**b

    def lr(self, c: float) -> float:
        a, p = self.lr_law
        return a * c**p

    def predict(self, c: float) -> float:
        a, al, _ = self.loss_law
        return self.l_inf + a * c**-al

    def predict_free(self, c: float) -> float:
        li, a, al = self.loss_law_free
        return li + a * c**-al

    def recipe(self, c: float) -> dict:
        """The compute-optimal (n, steps, lr) for a compute budget `c`."""
        n = int(round(self.n_star(c) / 8) * 8)
        steps = _fit.steps_for(c, n)
        return dict(n=n, steps=steps, lr=self.lr(train_flops(n, steps)),
                    tokens=steps * BATCH, params=512 * n,
                    predicted_loss=self.predict(train_flops(n, steps)))

    def summary(self) -> str:
        an, bn, r2n = self.n_law
        al, pl = self.lr_law
        aL, alL, r2L = self.loss_law
        li, aF, alF = self.loss_law_free
        L = [f"fitted on {len(self.rungs)} IsoFLOP rungs "
             f"({min(r['c'] for r in self.rungs):.3g} -> "
             f"{max(r['c'] for r in self.rungs):.3g} flops)",
             f"  n*(C)  = {an:.4g} * C^{bn:.4f}          r2={r2n:.4f}",
             f"  lr*(C) = {al:.4g} * C^{pl:.4f}          from {len(self.lr_anchors)} "
             f"bracketed lr sweep(s)",
             f"  L*(C)  = {self.l_inf:.4f} + {aL:.4g} * C^-{alL:.4f}   r2={r2L:.4f}   "
             f"(L_inf pinned to the eval set's mean entropy)",
             f"  L*(C)  = {li:.4f} + {aF:.4g} * C^-{alF:.4f}   (3-param, L_inf free)",
             f"  => N ~ C^{bn:.3f}, D ~ C^{1 - bn:.3f}"]
        L += [f"  note: {t}" for t in self.notes]
        return "\n".join(L)

    def plot(self, path=None, show=None):
        from .plots import plot_laws

        return plot_laws(self, path=path, show=show)


def fit_laws(rows, l_inf: float, lr_curvature: float | None = None) -> Laws:
    """IsoFLOP optima -> the three power laws.  See `Lab.fit`."""
    res = Results(list(rows))
    rungs = res.isoflop()
    if len(rungs) < 2:
        have = ", ".join(f"{r['c']:.3g}" for r in rungs) or "none"
        raise ValueError("need >=3 values of n at >=2 values of C to fit the laws; "
                         f"rungs with enough widths so far: {have}")

    # --- lr law: use rungs whose lr grid bracketed the optimum (interior minimum) --
    anchors, curv = [], []
    for c in res.rungs:
        sub = res.select(c=c)
        # take the n closest to that rung's optimum, then look along lr
        star = next((r["n_star"] for r in rungs if np.isclose(r["c"], c)), None)
        ns = sorted({r["n"] for r in sub.rows})
        n_pick = min(ns, key=lambda n: abs(np.log(n / star))) if star else ns[len(ns) // 2]
        cell = sorted(sub.select(n=n_pick).rows, key=lambda r: r["lr"])
        if len(cell) < 3:
            continue
        i = int(np.argmin([r["loss"] for r in cell]))
        if i in (0, len(cell) - 1):
            continue  # clipped: the true optimum is outside the grid
        x = np.log([r["lr"] for r in cell])
        y = np.array([r["loss"] for r in cell])
        co = np.polyfit(x, y, 2)
        if co[0] > 0:
            anchors.append((c, float(np.exp(-co[1] / (2 * co[0])))))
            curv.append(float(co[0]))
    notes = []
    if len(anchors) >= 2:
        a_lr, p_lr, _ = _fit.powerlaw([c for c, _ in anchors], [v for _, v in anchors])
    elif len(anchors) == 1:
        a_lr, p_lr = anchors[0][1], 0.0
        notes.append("only one bracketed lr sweep -> lr* assumed constant in C. "
                     "Sweep >=3 lrs at two different C to get the trend.")
    else:
        a_lr, p_lr = res.best()["lr"], 0.0
        notes.append("no lr sweep bracketed its optimum -> using the single best lr seen. "
                     "Your lr grid is probably too narrow.")
    k = lr_curvature if lr_curvature is not None else (float(np.mean(curv)) if curv else 0.0)

    # --- correct rungs whose best lr sat away from lr*(C) --------------------
    cs, nstar, lstar = [], [], []
    for r in rungs:
        c = r["c"]
        if r.get("clipped") in ("low", "high"):
            below = r["clipped"] == "low"
            notes.append(
                f"WARNING rung C={c:.3g}: the fitted optimum lies "
                f"{'below your smallest' if below else 'above your largest'} "
                f"n ({min(r['ns']) if below else max(r['ns'])}), so n* is a bound, not an "
                f"optimum. Widen the n grid at this rung or the law will lie.")
        elif r.get("clipped") == "flat":
            notes.append(f"WARNING rung C={c:.3g}: the loss-vs-n profile is not convex over "
                         f"the widths you tried, so n* is just the best point, not a fitted "
                         f"optimum. Add widths on both sides of it.")
        n_pick = min(r["ns"], key=lambda n: abs(np.log(n / r["n_star"])))
        lr_used = Results(res.select(c=c).select(n=n_pick).rows).best()["lr"]
        pen = k * np.log(lr_used / (a_lr * c**p_lr)) ** 2 if k > 0 else 0.0
        if pen > 0.005:
            notes.append(f"rung C={c:.3g} was trained at lr={lr_used:.4g} vs lr*="
                         f"{a_lr * c ** p_lr:.4g}; L* corrected by -{pen:.4f}")
        cs.append(c); nstar.append(r["n_star"]); lstar.append(r["loss_star"] - pen)

    cs, nstar, lstar = np.array(cs), np.array(nstar), np.array(lstar)
    an, bn, r2n = _fit.powerlaw(cs, nstar)
    aL, bL, r2L = _fit.powerlaw(cs, lstar - l_inf)
    if len(cs) >= 3:
        free = _fit.saturating_powerlaw(cs, lstar, l_inf0=l_inf)
    else:
        free = (l_inf, aL, -bL)
    return Laws(l_inf=l_inf,
                rungs=[dict(c=float(c), n_star=float(n), loss_star=float(l))
                       for c, n, l in zip(cs, nstar, lstar)],
                n_law=(an, bn, r2n), lr_law=(a_lr, p_lr),
                loss_law=(aL, -bL, r2L), loss_law_free=free,
                lr_anchors=anchors, notes=notes)


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
                 hero_curve_points: int = HERO_CURVE_POINTS,
                 hero_eval_tokens: int = HERO_EVAL_TOKENS,
                 hero_check_tokens: int = HERO_CHECK_TOKENS):
        self.name = name
        # anchored at the project root, so a notebook and a script share one lab
        self.dir = Path(root or Path(__file__).resolve().parents[2] / "runs") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.eval_tokens = eval_tokens
        self.hero_curve_points = hero_curve_points
        self.hero_eval_tokens = hero_eval_tokens
        self.hero_check_tokens = hero_check_tokens
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
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
            self._evals = get_evalset(self.eval_tokens, seed=0)
        return self._evals

    @property
    def l_inf(self) -> float:
        """Irreducible loss: the eval set's mean conditional entropy."""
        return float(self.evals.entropy.mean())

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
        return (f"lab '{self.name}':  {self.spent:.4g} / {self.budget:.3g} flops spent "
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

    def run_round(self, name: str, sweep: Sweep, plot: bool = True,
                  smoke: bool = False) -> Results:
        """Train every config in `sweep`.  Spends 1 round (unless ``smoke=True``).

        Configs sharing (n, steps) are trained in a single vmapped call, all on the
        same data stream, so comparisons between them are far less noisy than the
        absolute losses.
        """
        if smoke:
            sweep = Sweep(None, None, _configs=[
                replace(cfg, steps=max(10, cfg.steps // 100)) for cfg in sweep.configs])

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
        if not smoke and self.rounds_left <= 0:
            raise BudgetError(
                f"no screening rounds left ({self.rounds_used}/{self.max_rounds} used), "
                f"and {len(todo)} of these {len(sweep)} configs are new. "
                f"Fit your laws with lab.fit() and spend the remaining "
                f"{self.remaining:.3g} flops on lab.hero().")

        sub = Sweep(None, None, _configs=todo)
        est = Estimate(sub, self, self.eval_tokens)
        if not est.fits:
            self._refuse(est, "that sweep")
        print(f"round '{name}': {len(todo)} runs"
              + (f" ({cached} cached)" if cached else "")
              + f" in {len(sub.groups())} groups, {est.flops:.3g} flops, "
                f"~{est.seconds:.0f} s" + ("   [SMOKE]" if smoke else ""))

        stream = get_stream(max(c.tokens for c in todo))
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
                                c=c.c, n=c.n, steps=c.steps, tokens=c.tokens, lr=c.lr,
                                init=c.init, seed=c.seed, loss=float(loss),
                                flops=c.flops, smoke=bool(smoke)))
            print(f"  [{i}/{len(groups)}] n={n:5d} steps={steps:6d}  "
                  f"best lr={r.best()['lr']:.4g} loss={r.best()['loss']:.4f}")
        dt = time.time() - t0

        if not smoke:
            self.rows += new
            self.rounds_used += 1
        self.gflops = 0.5 * self.gflops + 0.5 * est.flops / max(dt, 1e-3) / 1e9
        self.round_log.append(dict(name=name, round=self.rounds_used, flops=est.flops,
                                   seconds=dt, n_runs=len(new), smoke=bool(smoke)))
        self._save()

        out = Results(new, name=name, flops=est.flops, seconds=dt)
        b = out.best()
        print(f"\nbest this round: n={b['n']} steps={b['steps']} lr={b['lr']:.4g}"
              f"  ->  loss {b['loss']:.4f}   ({dt:.0f} s)")
        print(self.status())
        if plot:
            out.plot(path=self.dir / f"round{self.rounds_used}_{name.replace(' ', '_')}.png",
                     show=None)
        return out

    # -- fitting -------------------------------------------------------------
    def fit(self, plot: bool = True, quiet: bool = False) -> Laws:
        """Fit n*(C), lr*(C) and L*(C) to every round run so far."""
        laws = fit_laws(self.rows, self.l_inf)
        if not quiet:
            print(laws.summary())
        if plot:
            laws.plot(path=self.dir / "laws.png", show=None)
        return laws

    # -- hero ----------------------------------------------------------------
    def hero(self, laws: Laws, plot: bool = True, margin: float = 2e10) -> dict:
        """Spend everything that is left on one run.  Can only be done once."""
        c = self.remaining - margin
        if self.hero_record is not None:
            # re-running the cell is fine; asking for a *different* hero run is not
            rec = self.hero_record
            want = laws.recipe(rec["c_train"])
            if want["n"] == rec["n"] and abs(want["lr"] / rec["lr_max"] - 1) < 1e-6:
                print(f"hero run already done -- replaying it (no flops spent).\n"
                      f"  n={rec['n']}, steps={rec['steps']}, lr={rec['lr_max']:.5f}\n"
                      f"  PREDICTED {rec['predicted']:.4f}  ->  ACTUAL {rec['loss']:.4f}"
                      f"   (error {rec['loss'] - rec['predicted']:+.4f})")
                if plot:
                    from .plots import plot_hero

                    plot_hero(rec, laws, show=None)
                return rec
            raise BudgetError(
                f"the hero run has already been done: n={rec['n']}, steps={rec['steps']}, "
                f"lr={rec['lr_max']:.5f}, loss {rec['loss']:.4f}.\nYou only get one shot -- "
                f"these laws would have asked for n={want['n']}, lr={want['lr']:.5f} instead. "
                f"Start a fresh Lab(name=...) if you want another attempt.")
        for _ in range(50):  # eval cost depends on n, which depends on c
            n = int(round(laws.n_star(c) / 8) * 8)
            ev = (eval_flops(n, self.eval_tokens) * self.hero_curve_points
                  + eval_flops(n, self.hero_eval_tokens)
                  + eval_flops(n, self.hero_check_tokens))
            c_new = self.remaining - margin - ev
            if abs(c_new - c) < 1e6:
                break
            c = c_new
        steps = _fit.steps_for(c, n)
        c_train = train_flops(n, steps)
        lr = laws.lr(c_train)
        pred, pred_free = laws.predict(c_train), laws.predict_free(c_train)
        print(f"HERO RECIPE at C={c_train:.4g} (+{ev:.3g} for evals)\n"
              f"  n = {n}  ({512 * n:,} params) | steps = {steps}  "
              f"({steps * BATCH:,} tokens) | lr = {lr:.5f} -> {lr / 10:.6f} cosine\n"
              f"  PREDICTED LOSS = {pred:.4f} nats   ({pred_free:.4f} from the "
              f"3-param fit)", flush=True)
        if c_train + ev > self.remaining:
            raise BudgetError("hero run does not fit -- this should not happen")

        r = train_sweep(n=n, steps=steps, lrs=[lr], stream=get_stream(steps * BATCH),
                        eval_set=self.evals, eval_tokens=self.eval_tokens,
                        eval_points=self.hero_curve_points, instance_seed=0, tag="hero",
                        return_params=True)
        ea = get_evalset(self.hero_eval_tokens, seed=0)
        eb = get_evalset(self.hero_check_tokens, seed=1)
        exact_a, samp_a, ma = evaluate(r.params, ea, n=n, instance_seed=0, y_seed=11)
        exact_b, samp_b, mb = evaluate(r.params, eb, n=n, instance_seed=0, y_seed=12)
        ledger.log("hero-final-eval", eval=eval_flops(n, ma) + eval_flops(n, mb), n=n)

        rec = dict(n=n, steps=steps, tokens=steps * BATCH, params=512 * n, lr_max=lr,
                   lr_min=lr / 10, c_train=c_train, c_eval=ev, predicted=pred,
                   predicted_free=pred_free, loss=float(exact_a[0]),
                   loss_sampled=float(samp_a[0]), loss_heldout_set=float(exact_b[0]),
                   irreducible=float(ea.entropy[:ma].mean()),
                   curve_steps=[int(x) for x in r.curve_steps],
                   curve_loss=[float(x) for x in r.curve.ravel()])
        self.hero_record = rec
        self._save()
        np.save(self.dir / "hero_W.npy", np.asarray(r.params[0]))
        print(f"\n=== HERO RESULT ===\n"
              f"  ACTUAL loss   = {rec['loss']:.4f} nats  ({ma} held-out tokens)\n"
              f"  PREDICTED     = {pred:.4f}          error {rec['loss'] - pred:+.4f}\n"
              f"  cross-checks  : {rec['loss_sampled']:.4f} (sampled-y CE), "
              f"{rec['loss_heldout_set']:.4f} (independent eval set)\n"
              f"  irreducible   = {rec['irreducible']:.4f}   -> excess "
              f"{rec['loss'] - rec['irreducible']:.4f} nats\n" + self.status())
        if plot:
            from .plots import plot_hero

            plot_hero(rec, laws, path=self.dir / "hero.png", show=None)
        return rec
