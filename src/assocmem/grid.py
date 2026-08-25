"""Machinery for the (N, D) scan: a stratified evaluator and one grid cell.

Why a separate evaluator.  With alpha = 1.2 the input distribution is so heavy-tailed
that the top 100 contexts carry 65 % of the mass and the top 10^4 carry 86 %, while
the remaining 14 % is spread over 10^11 contexts.  Sampling eval tokens from p(x), as
:mod:`assocmem.problem` does, therefore puts almost every token on the head and leaves
the tail -- the part that is *not* learned, i.e. the part the scaling law is about --
estimated from a handful of draws.  The standard error of the resulting loss is ~0.008
nats, which is the same size as the effects we are trying to fit an exponent to.

So instead: take the head *exactly* (all contexts i <= `head`, with their true weights
p(i)), and stratify the tail into log-spaced bins, sampling a fixed number of contexts
per bin and reweighting by the bin's exact mass.  Because the per-context loss varies
across bins (learned -> not learned) far more than within a bin, this removes nearly
all of the variance for ~34k tokens, and it costs nothing: evaluation is a few tenths
of a percent of the training flops.

Two further gains, both used by the analysis:

* the loss is reported as an **excess** over the irreducible entropy, token by token,
  ``sum_j w_j (CE_j - H_j)``.  H_j is known exactly here, so this is a paired
  difference rather than a difference of two noisy means.
* the per-bin excess is a direct picture of the step function the back-of-the-envelope
  calculation assumes: loss ~0 for contexts below capacity(N) and below D^(1/alpha),
  ~l = log d - E[H] above both.  How soft those two cliffs really are is the whole
  explanation for any gap between the predicted and measured exponents.
"""

from __future__ import annotations

import functools
import time as _time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import zeta as _hurwitz_zeta

from . import data as D
from .data import D_OUT, embed
from .problem import CACHE

# ------------------------------------------------------------------ theory side


def tail_mass(k: float, gamma: float = D.GAMMA) -> float:
    """sum_{i > k} p(i).  Exact (Hurwitz zeta), not the continuum approximation."""
    v = float(D.vocab_size(gamma))
    z = D.zipf_norm(gamma)
    return float(_hurwitz_zeta(gamma, k + 1.0) - _hurwitz_zeta(gamma, v + 1.0)) / z


def bin_mass(a: float, b: float, gamma: float = D.GAMMA,
             max_context: int | None = None) -> float:
    """sum_{a <= i < b} p(i)."""
    z = (D.zipf_norm(gamma) if max_context is None else
         float(_hurwitz_zeta(gamma, 1.0) - _hurwitz_zeta(gamma, max_context + 1.0)))
    return float(_hurwitz_zeta(gamma, a) - _hurwitz_zeta(gamma, b)) / z


# ------------------------------------------------------------------ eval set


@dataclass
class StratEval:
    """Importance-weighted eval set: exact head, log-stratified tail."""

    hi: np.ndarray  # (M,) uint32
    lo: np.ndarray  # (M,) uint32
    probs: np.ndarray  # (M, 512) float32, exact p(y|x)
    entropy: np.ndarray  # (M,) nats
    weight: np.ndarray  # (M,) float64, sums to 1 over the whole vocabulary
    index: np.ndarray  # (M,) float64, the context id (for plotting)
    bin_lo: np.ndarray  # (n_bins,) left edge of each stratum
    bin_id: np.ndarray  # (M,) int32

    def __len__(self) -> int:
        return len(self.hi)

    @property
    def l_inf(self) -> float:
        """The irreducible loss E_{x~p}[H(y|x)], under this set's own weights."""
        return float(self.weight @ self.entropy)


def _strat_file(head: int, per_bin: int, per_decade: int, seed: int,
                gamma: float, max_context: int | None = None) -> Path:
    g = "" if gamma == D.GAMMA else f"_g{gamma:.3f}"
    v = "" if max_context is None else f"_v{max_context}"
    return CACHE / f"strat_eval2_h{head}_b{per_bin}_d{per_decade}_s{seed}{g}{v}.npz"


def build_strat_eval(head: int = 4096, per_bin: int = 512, per_decade: int = 8,
                     seed: int = 11, gamma: float = D.GAMMA,
                     max_context: int | None = None) -> StratEval:
    """Log-spaced strata over the whole vocabulary, `per_decade` of them.

    A stratum is taken **exactly** (every context in it, with its true weight p(i)) when
    it lies below `head` or is narrower than `per_bin`; wider strata are sampled
    uniformly and reweighted by the stratum's exact mass.  The head is therefore not
    one lump: the strata run from the single most frequent context upwards, which is
    what lets the per-stratum loss curve resolve the capacity cliff.
    """
    f = _strat_file(head, per_bin, per_decade, seed, gamma, max_context)
    if f.exists():
        z = np.load(f)
        return StratEval(**{k: z[k] for k in z.files})

    v = float(D.vocab_size(gamma) if max_context is None else max_context)
    n_edge = int(np.ceil(np.log10(v + 1.0) * per_decade)) + 1
    edges = np.unique(np.floor(10 ** (np.arange(n_edge) / per_decade)).astype(np.int64))
    edges = np.append(edges[edges <= v], int(v) + 1)
    head_edge = int(edges[edges >= head][0]) if (edges >= head).any() else int(edges[-1])

    rng = np.random.default_rng(seed)
    idx, wts, bid, lows = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        width = int(b - a)
        if b <= head_edge or width <= per_bin:  # exact stratum
            tok = np.arange(a, b, dtype=np.int64)
            w = np.array([bin_mass(i, i + 1, gamma, max_context) for i in tok])
        else:  # sampled stratum, reweighted by its exact mass
            tok = np.unique(rng.integers(a, b, size=per_bin))
            w = np.full(len(tok), bin_mass(a, b, gamma, max_context) / len(tok))
        idx.append(tok)
        wts.append(w)
        bid.append(np.full(len(tok), len(lows), dtype=np.int32))
        lows.append(int(a))

    index = np.concatenate(idx)
    weight = np.concatenate(wts)
    bin_id = np.concatenate(bid)
    weight /= weight.sum()  # the zeta/cutoff arithmetic leaves ~1e-5; make it exact
    probs = D.conditional(index)
    entropy = -(probs * np.log(np.maximum(probs, 1e-45))).sum(1)
    hi, lo = D.split_u32(index)
    se = StratEval(hi=hi, lo=lo, probs=probs, entropy=entropy.astype(np.float64),
                   weight=weight, index=index.astype(np.float64),
                   bin_lo=np.array(lows, dtype=np.float64), bin_id=bin_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: getattr(se, k) for k in se.__dataclass_fields__})
    return se


# ------------------------------------------------------------------ weighted eval


@functools.partial(jax.jit, static_argnames=("n", "chunk"))
def _ce_per_token(w, hi, lo, probs, key, n: int, chunk: int = 2048):
    """(K, M) exact cross-entropy -sum_y p(y|x) log p_hat(y|x), token by token."""
    nchunk = hi.shape[0] // chunk

    def body(i, out):
        sl = jax.lax.dynamic_slice
        h = sl(hi, (i * chunk,), (chunk,))
        l = sl(lo, (i * chunk,), (chunk,))
        pr = sl(probs, (i * chunk, 0), (chunk, D_OUT))
        e = embed(key, h, l, n)
        lp = jax.nn.log_softmax(jnp.einsum("kdn,bn->kbd", w, e), axis=-1)
        ce = -(pr[None] * lp).sum(-1)  # (K, chunk)
        return jax.lax.dynamic_update_slice(out, ce, (0, i * chunk))

    out = jnp.zeros((w.shape[0], nchunk * chunk), jnp.float32)
    return jax.lax.fori_loop(0, nchunk, body, out)


def excess_loss(params, se: StratEval, *, n: int, instance_seed: int = 0,
                chunk: int = 2048):
    """Weighted excess loss per config, and its breakdown over strata.

    Returns ``(excess, per_bin, bin_weight)`` with shapes ``(K,)``, ``(K, n_bins)``,
    ``(n_bins,)``.  ``excess[k] = sum_j w_j (CE_jk - H_j)`` estimates
    ``E_{x~p}[CE] - L_inf``; ``per_bin[k, m]`` is the mean excess *within* stratum m,
    i.e. the per-context loss the envelope calculation idealises as a step.
    """
    from .train import embedding_key

    m = len(se)
    pad = (-m) % chunk  # zero-weight padding so the kernel can use a fixed chunk
    def _pad(a, val=0):
        return np.concatenate([a, np.full((pad,) + a.shape[1:], val, a.dtype)]) if pad else a

    ce = np.asarray(_ce_per_token(
        params, jnp.asarray(_pad(se.hi)), jnp.asarray(_pad(se.lo)),
        jnp.asarray(_pad(se.probs)), embedding_key(instance_seed), n, chunk))[:, :m]
    ex = ce.astype(np.float64) - se.entropy[None, :]
    w = se.weight
    n_bins = int(se.bin_id.max()) + 1
    bw = np.bincount(se.bin_id, weights=w, minlength=n_bins)
    per_bin = np.stack([np.bincount(se.bin_id, weights=w * e, minlength=n_bins) / bw
                        for e in ex])
    return (ex @ w), per_bin, bw


# ------------------------------------------------------------------ one grid cell


def lr_guess(h: int, steps: int) -> float:
    """Prior on lr*, only used to centre the initial grid (which is then refined).

    The reference solution's fit was lr* = 14.85 C^-0.238 with C the run's flops; that
    conflates the width and the length of the run, but it is the right ballpark and a
    wrong guess costs one extra rung, not correctness.
    """
    c = 6.0 * D_OUT * h * 64 * steps
    return float(14.85 * c**-0.238)


def _parabola_min(log_lr, excess, k: int = 3):
    """Vertex of a parabola through the k best points.  Returns (lr*, excess*, edge)."""
    log_lr = np.asarray(log_lr, float)
    excess = np.asarray(excess, float)
    order = np.argsort(excess)
    if len(log_lr) < 3:
        i = int(order[0])
        return float(np.exp(log_lr[i])), float(excess[i]), "few"
    keep = np.sort(order[:min(k, len(log_lr))])
    coef = np.polyfit(log_lr[keep], excess[keep], 2)
    imin = int(order[0])
    if coef[0] <= 0:  # not convex over these points: fall back to the argmin
        return float(np.exp(log_lr[imin])), float(excess[imin]), "flat"
    xs = -coef[1] / (2 * coef[0])
    edge = ("low" if imin == int(np.argmin(log_lr)) else
            "high" if imin == int(np.argmax(log_lr)) else "")
    if not (log_lr.min() <= xs <= log_lr.max()):
        edge = edge or ("low" if xs < log_lr.min() else "high")
        xs = float(np.clip(xs, log_lr.min(), log_lr.max()))
    return float(np.exp(xs)), float(np.polyval(coef, xs)), edge


def run_cell(h: int, steps: int, se: StratEval, stream, eval_set, *, seed: int = 0,
             lrs=None, refine: int = 3, eval_tokens: int = 4096,
             seg_steps: int = 102_400, tag: str = "grid") -> dict:
    """Train an (h, steps) cell over a learning-rate grid and return the best.

    The grid is *refined until the optimum is interior*: an argmin sitting on an edge
    means the reported loss is an upper bound, and since the bias would vary
    systematically with h and steps it would corrupt the fitted exponents rather than
    just shift them.  This is the failure mode `Laws` warns about in the student lab,
    and here it is closed automatically.
    """
    from .train import train_sweep

    if lrs is None:
        g = lr_guess(h, steps)
        lrs = [g * 2.0**e for e in (-2, -1, 0, 1, 2)]
    lrs = sorted(float(x) for x in lrs)
    eval_points = max(1, int(np.ceil(steps / seg_steps)))

    t0 = _time.time()
    seen: dict[float, tuple[float, float, np.ndarray]] = {}
    flops = 0.0
    for _ in range(refine + 1):
        todo = [x for x in lrs if x not in seen]
        if todo:
            r = train_sweep(h, steps, todo, stream, eval_set, instance_seed=seed,
                            eval_points=eval_points, eval_tokens=eval_tokens,
                            tag=f"{tag}-h{h}-s{steps}", return_params=True)
            ex, per_bin, _ = excess_loss(r.params, se, n=h, instance_seed=seed)
            flops += r.train_flops + r.eval_flops
            for i, x in enumerate(todo):
                seen[x] = (float(ex[i]), float(r.loss[i]), per_bin[i])
        ks = np.array(sorted(seen))
        vals = np.array([seen[k][0] for k in ks])
        _, _, edge = _parabola_min(np.log(ks), vals)
        if edge in ("", "flat", "few"):
            break
        lrs = list(ks) + [ks.min() / 4 if edge == "low" else ks.max() * 4]

    ks = np.array(sorted(seen))
    vals = np.array([seen[k][0] for k in ks])
    lr_star, ex_star, edge = _parabola_min(np.log(ks), vals)
    i = int(np.argmin(vals))
    return dict(h=h, steps=steps, seed=seed, n_params=D_OUT * h, tokens=64 * steps,
                lrs=[float(x) for x in ks], excess=[float(v) for v in vals],
                loss_std_eval=[float(seen[k][1]) for k in ks],
                lr_best=float(ks[i]), excess_best=float(vals[i]),
                lr_star=lr_star, excess_star=ex_star, lr_edge=edge,
                per_bin=[float(v) for v in seen[ks[i]][2]],
                flops=float(flops), seconds=round(_time.time() - t0, 1))
