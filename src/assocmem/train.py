"""Vectorised trainer for  p_hat(.|x) = softmax(W e_x),  W in R^{512 x n}.

A single call trains a whole *sweep* of hyper-parameter configurations at once
(``vmap`` over the config axis) so that (a) the tiny matmuls get batched into one
big one and (b) every config sees exactly the same data stream -- common random
numbers, which makes the comparisons between configs far less noisy than the
absolute losses.

Gradients are written by hand (the model is one linear layer) which avoids all
autodiff overhead:  dW = (softmax(W e) - onehot(y)) e^T / B.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import ledger
from .data import D_OUT, embed

BATCH = 64  # fixed by the task


# --------------------------------------------------------------------------- #
# flop accounting
# --------------------------------------------------------------------------- #
def train_flops(n: int, steps: int, batch: int = BATCH, n_cfg: int = 1) -> float:
    """6*N*D convention (N = 512n params, D = batch*steps tokens)."""
    return 6.0 * (D_OUT * n) * batch * steps * n_cfg


def eval_flops(n: int, n_tokens: int, n_cfg: int = 1) -> float:
    return 2.0 * (D_OUT * n) * n_tokens * n_cfg


# --------------------------------------------------------------------------- #
# data bundle
# --------------------------------------------------------------------------- #
@dataclass
class Stream:
    """A pre-generated training stream: token ids (as u32 halves) and labels."""

    hi: np.ndarray  # (T,) uint32
    lo: np.ndarray  # (T,) uint32
    y: np.ndarray  # (T,) int32

    def __len__(self) -> int:
        return len(self.y)

    def batches(self, steps: int, batch: int = BATCH, offset: int = 0):
        need = steps * batch
        assert offset + need <= len(self), f"stream too short: {len(self)} < {offset + need}"
        s = slice(offset, offset + need)
        return (
            jnp.asarray(self.hi[s].reshape(steps, batch)),
            jnp.asarray(self.lo[s].reshape(steps, batch)),
            jnp.asarray(self.y[s].reshape(steps, batch).astype(np.int32)),
        )


@dataclass
class EvalSet:
    """Evaluation tokens with their *exact* conditionals, for a zero-variance-in-y loss.

    ``weight`` carries the importance weights of a stratified set (exact head, sampled
    tail reweighted by each stratum's exact mass); left as ``None`` it means a plain
    Monte-Carlo set drawn from p(x), which is weighted uniformly.
    """

    hi: np.ndarray  # (M,) uint32
    lo: np.ndarray  # (M,) uint32
    probs: np.ndarray  # (M, 512) float32
    entropy: np.ndarray = field(default=None)  # (M,) nats, irreducible loss
    weight: np.ndarray = field(default=None)  # (M,) float64, sums to 1

    def __len__(self) -> int:
        return len(self.hi)

    @property
    def w(self) -> np.ndarray:
        """Normalised weights, uniform if this is a Monte-Carlo set."""
        if self.weight is None:
            return np.full(len(self), 1.0 / len(self))
        return self.weight / self.weight.sum()

    @property
    def l_inf(self) -> float:
        """The irreducible loss E_{x~p}[H(y|x)] under this set's own weights."""
        return float(self.w @ self.entropy)

    def padded(self, chunk: int, m: int | None = None):
        """Arrays padded with zero-weight rows to a whole number of eval chunks.

        The kernel wants a fixed chunk size, and truncating to a multiple of it would
        silently throw away part of a stratified set's weight -- so pad instead.
        """
        m = len(self) if m is None else min(m, len(self))
        pad = (-m) % chunk
        take = np.concatenate([np.arange(m), np.zeros(pad, dtype=np.int64)])
        wt = np.concatenate([self.w[:m], np.zeros(pad)])
        return self.hi[take], self.lo[take], self.probs[take], wt / wt.sum()

    def sample_y(self, seed: int = 0) -> np.ndarray:
        """One y ~ p(.|x) per eval token (for the plain sampled-CE estimator)."""
        cdf = np.cumsum(self.probs.astype(np.float64), axis=1)
        cdf[:, -1] = 1.0
        u = np.random.default_rng(seed).random(len(self))[:, None]
        return np.minimum((cdf < u).sum(1), D_OUT - 1).astype(np.int32)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def _cosine(step, total, lr_max):
    """lr_max -> lr_max/10, cosine."""
    lr_min = lr_max / 10.0
    c = 0.5 * (1.0 + jnp.cos(jnp.pi * step / total))
    return lr_min + (lr_max - lr_min) * c


def embedding_key(instance_seed: int):
    """The key that generates this problem instance's embeddings."""
    return jax.random.split(jax.random.key(instance_seed))[1]


def init_params(key, n: int, init_scales) -> jnp.ndarray:
    """(K, 512, n).  init_scale is the std of the initial logits (0 => zero init)."""
    scales = jnp.asarray(init_scales, dtype=jnp.float32)
    w = jax.random.normal(key, (len(scales), D_OUT, n), dtype=jnp.float32) / jnp.sqrt(n)
    return w * scales[:, None, None]


@functools.partial(jax.jit, static_argnames=("n", "total_steps"))
def _train_segment(w, m, v, t0, xs, lrs, key, n: int, total_steps: int,
                   b1=0.9, b2=0.999, eps=1e-8):
    hi_all, lo_all, y_all = xs
    batch = hi_all.shape[1]
    ar = jnp.arange(batch)

    def step(carry, x):
        w, m, v, t = carry
        hi, lo, y = x
        e = embed(key, hi, lo, n)  # (B, n)
        logits = jnp.einsum("kdn,bn->kbd", w, e)
        p = jax.nn.softmax(logits, axis=-1)
        g = p.at[:, ar, y].add(-1.0) / batch  # (K, B, d)
        gw = jnp.einsum("kbd,bn->kdn", g, e)  # (K, d, n)

        m = b1 * m + (1 - b1) * gw
        v = b2 * v + (1 - b2) * gw * gw
        t1 = t + 1
        mh = m / (1 - b1**t1)
        vh = v / (1 - b2**t1)
        lr = _cosine(t, total_steps, lrs)[:, None, None]
        w = w - lr * mh / (jnp.sqrt(vh) + eps)
        return (w, m, v, t1), None

    (w, m, v, t), _ = jax.lax.scan(step, (w, m, v, t0), (hi_all, lo_all, y_all))
    return w, m, v, t


@functools.partial(jax.jit, static_argnames=("n", "chunk"))
def _eval(w, hi, lo, probs, wt, key, n: int, chunk: int = 2048):
    """Exact expected cross-entropy  E_{y~p(.|x)}[-log p_hat(y|x)]  over the set.

    ``wt`` are normalised importance weights, so this is a weighted sum, not a mean.
    """
    nchunk = hi.shape[0] // chunk

    def body(i, acc):
        sl = jax.lax.dynamic_slice
        h = sl(hi, (i * chunk,), (chunk,))
        l = sl(lo, (i * chunk,), (chunk,))
        pr = sl(probs, (i * chunk, 0), (chunk, D_OUT))
        ww = sl(wt, (i * chunk,), (chunk,))
        e = embed(key, h, l, n)
        logits = jnp.einsum("kdn,bn->kbd", w, e)
        lp = jax.nn.log_softmax(logits, axis=-1)
        return acc + (ww * -(pr[None] * lp).sum(-1)).sum(-1)

    return jax.lax.fori_loop(0, nchunk, body, jnp.zeros((w.shape[0],), jnp.float32))


@functools.partial(jax.jit, static_argnames=("n", "chunk"))
def _eval_dual(w, hi, lo, probs, ys, wt, key, n: int, chunk: int = 2048):
    """Both loss estimators from one forward pass.

    exact:   E_{y~p(.|x)}[-log p_hat(y|x)]   (uses the known conditional; no y noise)
    sampled: -log p_hat(y|x) for one y ~ p(.|x)  (plain held-out cross-entropy)
    """
    nchunk = hi.shape[0] // chunk
    ar = jnp.arange(chunk)

    def body(i, acc):
        sl = jax.lax.dynamic_slice
        h = sl(hi, (i * chunk,), (chunk,))
        l = sl(lo, (i * chunk,), (chunk,))
        pr = sl(probs, (i * chunk, 0), (chunk, D_OUT))
        yy = sl(ys, (i * chunk,), (chunk,))
        ww = sl(wt, (i * chunk,), (chunk,))
        e = embed(key, h, l, n)
        lp = jax.nn.log_softmax(jnp.einsum("kdn,bn->kbd", w, e), axis=-1)
        return (acc[0] + (ww * -(pr[None] * lp).sum(-1)).sum(-1),
                acc[1] + (ww * -lp[:, ar, yy]).sum(-1))

    z = jnp.zeros((w.shape[0],), jnp.float32)
    tot = jax.lax.fori_loop(0, nchunk, body, (z, z))
    return tot[0], tot[1]


def evaluate(params, eval_set: "EvalSet", *, n: int, instance_seed: int = 0,
             y_seed: int = 0, chunk: int = 2048):
    """Final evaluation of trained params on an eval set.  Returns (exact, sampled)."""
    hi, lo, probs, wt = eval_set.padded(chunk)
    ys = eval_set.sample_y(y_seed)
    ys = np.concatenate([ys, np.zeros(len(wt) - len(ys), dtype=ys.dtype)])
    a, b = _eval_dual(params, jnp.asarray(hi), jnp.asarray(lo), jnp.asarray(probs),
                      jnp.asarray(ys), jnp.asarray(wt, dtype=jnp.float32),
                      embedding_key(instance_seed), n, chunk)
    return np.asarray(a), np.asarray(b), len(wt)


# --------------------------------------------------------------------------- #
# user-facing entry point
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    n: int
    steps: int
    lrs: np.ndarray
    init_scales: np.ndarray
    loss: np.ndarray  # (K,) final loss
    curve: np.ndarray  # (n_eval_points, K)
    curve_steps: np.ndarray
    train_flops: float
    eval_flops: float
    params: object = None

    def best(self):
        i = int(np.argmin(self.loss))
        return dict(lr=float(self.lrs[i]), init_scale=float(self.init_scales[i]),
                    loss=float(self.loss[i]))


def plan_cost(n: int, steps: int, n_cfg: int = 1, eval_tokens: int = 0,
              eval_points: int = 1) -> float:
    """Flops a `train_sweep` call will bill to the ledger.  Check before running!"""
    return train_flops(n, steps, BATCH, n_cfg) + eval_flops(n, eval_tokens, n_cfg) * eval_points


def train_sweep(
    n: int,
    steps: int,
    lrs,
    stream: Stream,
    eval_set: EvalSet,
    *,
    init_scales=None,
    instance_seed: int = 0,
    stream_offset: int = 0,
    eval_points: int = 1,
    eval_chunk: int = 2048,
    eval_tokens: int | None = None,
    tag: str = "untagged",
    return_params: bool = False,
) -> SweepResult:
    """Train ``len(lrs)`` configs of an n-dim model for ``steps`` steps, in parallel.

    ``lrs`` and ``init_scales`` are broadcast against each other to form the config
    axis, so you can sweep either or both.  The cost is billed to the flop ledger
    under ``tag``; the call refuses to run if it would blow the budget.
    """
    lrs = np.atleast_1d(np.asarray(lrs, dtype=np.float64))
    if init_scales is None:
        init_scales = np.zeros_like(lrs)
    init_scales = np.atleast_1d(np.asarray(init_scales, dtype=np.float64))
    lrs, init_scales = np.broadcast_arrays(lrs, init_scales)
    lrs, init_scales = lrs.ravel().copy(), init_scales.ravel().copy()
    k = len(lrs)

    ikey, ekey = jax.random.split(jax.random.key(instance_seed))
    w = init_params(ikey, n, init_scales)
    m = jnp.zeros_like(w)
    v = jnp.zeros_like(w)
    lrs_j = jnp.asarray(lrs, dtype=jnp.float32)

    ehi_np, elo_np, eprobs_np, ewt_np = eval_set.padded(eval_chunk, eval_tokens)
    m_eval = len(ewt_np)

    n_eval_pts = len(np.unique(np.linspace(0, steps, eval_points + 1).astype(int))) - 1
    cost = plan_cost(n, steps, k, m_eval, n_eval_pts)
    spent = ledger.total()["total"]
    if spent + cost > ledger.BUDGET:
        raise RuntimeError(
            f"refusing to run: would cost {cost:.4g} flops, only "
            f"{ledger.BUDGET - spent:.4g} left of the {ledger.BUDGET:.3g} budget")
    ehi = jnp.asarray(ehi_np)
    elo = jnp.asarray(elo_np)
    eprobs = jnp.asarray(eprobs_np)
    ewt = jnp.asarray(ewt_np, dtype=jnp.float32)

    bounds = np.unique(np.linspace(0, steps, eval_points + 1).astype(int))
    curve, curve_steps = [], []
    t = jnp.zeros((), jnp.int32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        xs = stream.batches(int(b - a), offset=stream_offset + int(a) * BATCH)
        w, m, v, t = _train_segment(w, m, v, t, xs, lrs_j, ekey, n, steps)
        loss = np.asarray(_eval(w, ehi, elo, eprobs, ewt, ekey, n, eval_chunk))
        curve.append(loss)
        curve_steps.append(int(b))

    tr = train_flops(n, steps, BATCH, k)
    ev = eval_flops(n, m_eval, k) * n_eval_pts
    ledger.log(tag, train=tr, eval=ev, n=n, steps=steps, n_cfg=k,
               instance_seed=instance_seed, eval_tokens=m_eval,
               loss=[float(x) for x in curve[-1]], lrs=[float(x) for x in lrs],
               init_scales=[float(x) for x in init_scales])
    return SweepResult(
        n=n, steps=steps, lrs=lrs, init_scales=init_scales,
        loss=curve[-1], curve=np.array(curve), curve_steps=np.array(curve_steps),
        train_flops=tr, eval_flops=ev, params=(w if return_params else None),
    )
