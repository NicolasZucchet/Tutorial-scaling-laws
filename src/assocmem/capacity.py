"""Storage capacity of the toy associative memory: Hebbian vs trained.

The slides' model is ``p(.|i) = softmax(W e_i)`` with ``W in R^{d x h}``: context ``i``
carries a random embedding ``e_i`` on the unit sphere of ``R^h`` and has a single
correct next token ``y*_i`` out of ``d``.  A context is **stored** when
``argmax_y (W e_i)_y == y*_i``, and the *capacity* is the largest number of contexts
that can all be stored at once.

The context frequencies of the full problem play no role here -- a 100 %-accuracy
criterion does not care how often a context appears -- so contexts are unweighted and
the Zipf law of :mod:`assocmem.data` is deliberately absent.

Two ways of building ``W``:

``hebbian``
    ``W = sum_i z_i e_i^T`` with ``z_i`` the one-hot of ``y*_i``: one gradient step on
    ``(1/2) sum_i ||W e_i - z_i||^2`` starting from ``W = 0``.
``trained``
    Adam on the cross-entropy of the ``n`` pairs, full batch, **run to saturation**:
    training stops when the accuracy hits 100 % or when it stops improving, never on a
    step budget, so the number reported is not an artefact of short training.  Several
    learning rates are trained in parallel (vmapped, so the small matmuls batch) and
    the best one wins -- the number is "reachable by Adam", not "reachable by this lr".

Instances are **nested** in ``n``: the ``n``-context problem is a prefix of the
``n'``-context one for ``n < n'``.  That is what makes "the largest ``n``" a
well-defined bisection target.

Two caveats worth knowing before reading the numbers:

* "all stored" is only *approximately* monotone in ``n``.  Adding a context usually
  hurts, but it can also repair one, by adding weight to a class that was losing --
  so a handful of successes can sit above the first failure, and a bisection reports
  one crossing rather than the first.  A scan at ``h >= 64`` finds 0-7 such successes
  out of ~300 probed ``n``, i.e. the crossing is sharp; at ``h = 32`` it is not, and
  the seed-to-seed spread there is genuinely large.
* the trained capacity is what Adam **reaches**, and near the boundary the accuracy
  creeps for thousands of steps.  ``TrainConfig.max_steps`` is a backstop, and probes
  that hit it are flagged ``capped``.  Measured at ``h = 32``: raising the backstop
  from 8k to 32k steps moves the capacity by 0.4 %, and 4k to 8k by 0 %, so 8k is
  converged for the purpose of a log-log plot.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

D_TOK = 256  # number of possible next tokens (the slides' d)

_LABEL_OFFSET = 1 << 32  # keeps the label stream independent of the embedding stream


# --------------------------------------------------------------------------- #
# problem instance
# --------------------------------------------------------------------------- #
def instance(n: int, h: int, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``n`` unit-sphere embeddings in ``R^h`` and their ``n`` target tokens.

    Nested in ``n``: numpy's Generator fills an array in C order straight from the
    bit stream, so the first ``k`` rows of an ``(n, h)`` draw are exactly the ``(k, h)``
    draw.  Two contexts are therefore "the same context" across every ``n`` probed
    during one bisection, which is what the capacity is defined against.
    """
    e = np.random.default_rng([seed, h]).standard_normal((n, h), dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    y = np.random.default_rng([seed + _LABEL_OFFSET, h]).integers(
        0, D_TOK, size=n, dtype=np.int32)
    return jnp.asarray(e), jnp.asarray(y)


# --------------------------------------------------------------------------- #
# Hebbian
# --------------------------------------------------------------------------- #
@jax.jit
def hebbian_accuracy(e: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Fraction of contexts stored by ``W = sum_i z_i e_i^T``."""
    w = jnp.zeros((D_TOK, e.shape[1]), jnp.float32).at[y].add(e)  # (d, h)
    return (jnp.argmax(e @ w.T, axis=1) == y).mean()


# --------------------------------------------------------------------------- #
# trained
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """How far to push Adam before calling a candidate ``n`` unreachable."""

    lrs: tuple[float, ...] = (0.3, 1.0, 3.0)
    segment: int = 500  # steps between accuracy checks
    patience: int = 8  # segments without improvement before declaring saturation
    tol: float = 1e-6  # improvement below this does not count
    max_steps: int = 200_000  # hard backstop; saturation should fire long before


@functools.partial(jax.jit, static_argnames=("segment",))
def _train_segment(w, m, v, t, e, y, lrs, segment: int,
                   b1=0.9, b2=0.999, eps=1e-8):
    """``segment`` full-batch Adam steps at every learning rate at once.

    Gradient of the mean cross-entropy, written by hand (one linear layer):
    ``dW = (softmax(W e) - onehot(y))^T e / n``.
    """
    n = e.shape[0]
    ar = jnp.arange(n)

    def step(carry, _):
        w, m, v, t = carry
        p = jax.nn.softmax(jnp.einsum("kdh,nh->knd", w, e), axis=-1)
        g = p.at[:, ar, y].add(-1.0) / n  # (k, n, d)
        gw = jnp.einsum("knd,nh->kdh", g, e)
        m = b1 * m + (1 - b1) * gw
        v = b2 * v + (1 - b2) * gw * gw
        t = t + 1
        upd = (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
        return (w - lrs[:, None, None] * upd, m, v, t), None

    (w, m, v, t), _ = jax.lax.scan(step, (w, m, v, t), None, length=segment)
    acc = (jnp.argmax(jnp.einsum("kdh,nh->knd", w, e), axis=-1) == y[None]).mean(axis=1)
    return w, m, v, t, acc


@dataclass
class TrainOutcome:
    stored: bool  # did some learning rate store every context?
    best_acc: float
    steps: int
    capped: bool = False  # stopped by max_steps rather than by saturation
    per_lr: list[float] = field(default_factory=list)
    history: list[list[float]] = field(default_factory=list)


def train_to_saturation(e, y, cfg: TrainConfig = TrainConfig()) -> TrainOutcome:
    """Full-batch Adam at every ``cfg.lrs``, stopped by success or by saturation."""
    h = e.shape[1]
    lrs = jnp.asarray(cfg.lrs, dtype=jnp.float32)
    w = jnp.zeros((len(cfg.lrs), D_TOK, h), jnp.float32)
    m, v = jnp.zeros_like(w), jnp.zeros_like(w)
    t = jnp.zeros((), jnp.float32)

    best = np.zeros(len(cfg.lrs))
    stale, steps, history = 0, 0, []
    while steps < cfg.max_steps:
        w, m, v, t, acc = _train_segment(w, m, v, t, e, y, lrs, cfg.segment)
        acc = np.asarray(acc, dtype=np.float64)
        steps += cfg.segment
        history.append([round(float(a), 6) for a in acc])
        if float(acc.max()) >= 1.0:
            return TrainOutcome(True, 1.0, steps, False, list(acc), history)
        # saturation is judged on the whole lr front: as long as *any* learning rate
        # is still improving, training has not run out of room.
        if (acc > best + cfg.tol).any():
            best, stale = np.maximum(best, acc), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                return TrainOutcome(False, float(best.max()), steps, False,
                                    list(best), history)
    # still improving when the backstop hit: the answer is budget-limited, and the
    # caller is told so rather than being handed a silent under-estimate.
    return TrainOutcome(False, float(best.max()), steps, True, list(best), history)


# --------------------------------------------------------------------------- #
# capacity by bisection
# --------------------------------------------------------------------------- #
def stores_all(model: str, n: int, h: int, seed: int,
               cfg: TrainConfig = TrainConfig()) -> tuple[bool, float, int, bool]:
    """Can ``model`` store all ``n`` contexts?  -> (stored, accuracy, steps, capped)."""
    e, y = instance(n, h, seed)
    if model == "hebbian":
        acc = float(hebbian_accuracy(e, y))
        return acc >= 1.0, acc, 0, False
    if model == "trained":
        out = train_to_saturation(e, y, cfg)
        return out.stored, out.best_acc, out.steps, out.capped
    raise ValueError(f"unknown model {model!r}")


@dataclass
class CapacityResult:
    model: str
    h: int
    seed: int
    capacity: int  # largest n known to store every context
    upper: int  # smallest n known to fail; capacity is in [capacity, upper)
    precision: int
    n_probes: int
    n_capped: int = 0  # probes whose verdict was decided by the step backstop
    trace: list[dict] = field(default_factory=list)


def capacity(model: str, h: int, seed: int, *, precision: int = 8,
             guess: int | None = None, cfg: TrainConfig = TrainConfig(),
             log=None) -> CapacityResult:
    """Largest ``n`` whose every context is stored, by bisection to ``precision``.

    ``guess`` seeds the bracket (a rough prior on the answer costs nothing to be wrong
    about: the bracket is grown by doubling / halving until it really brackets).  The
    bisection then narrows until fewer than ``precision`` contexts separate the largest
    success from the smallest failure, and the success is returned.
    """
    trace: list[dict] = []
    cache: dict[int, bool] = {}

    def probe(n: int) -> bool:
        n = max(1, int(n))
        if n not in cache:
            stored, acc, steps, capped = stores_all(model, n, h, seed, cfg)
            cache[n] = stored
            trace.append(dict(n=n, stored=stored, accuracy=acc, steps=steps,
                              capped=capped))
            if log:
                tail = f"  {steps} steps{' (CAPPED)' if capped else ''}" if steps else ""
                log(f"    n={n:>7d}  acc={acc:.6f}  "
                    f"{'stored' if stored else 'lost':>6s}{tail}")
        return cache[n]

    lo = max(1, int(guess if guess else 4 * h))
    hi = 2 * lo
    while lo > 1 and not probe(lo):  # bracket from above: halve until something works
        hi, lo = lo, max(1, lo // 2)
    while probe(hi):  # bracket from below: double until something fails
        lo, hi = hi, 2 * hi
    while hi - lo > precision:
        mid = (lo + hi) // 2
        if probe(mid):
            lo = mid
        else:
            hi = mid
    return CapacityResult(model, h, seed, lo, hi, precision, len(cache),
                          sum(1 for t in trace if t["capped"]), trace)
