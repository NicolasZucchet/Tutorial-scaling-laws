"""Emergence: per-band top-1 accuracy behind the smooth loss of the (N, D) scan.

The loss the scan measures is an average over the whole Zipf tail, and it is smooth in
both N and D -- that is the whole point of a scaling law.  This module looks at the same
runs through a different lens: a *uniform* top-1 accuracy on a narrow band of context
indices.

    "does the model know the most likely next token for context i?"

for i in a band such as 20k-30k.  Two things change relative to :mod:`assocmem.grid`:

* **the metric is a threshold, not an average.**  A context contributes 0 or 1 depending
  on whether ``argmax_y W e_i`` equals ``argmax_y p(y|x=i)``, so a context that is
  half-learned counts as not learned.
* **the weighting is uniform inside the band**, not p(i).  A band is one slice of the
  step function the envelope calculation assumes, so it is not masked by the head.

Together those turn the smooth macro curve into a set of sigmoids that switch on one
after another, each band waiting for the capacity (or the data) to reach it.  Nothing
new is being trained: the model, the stream, the schedule and the learning-rate surface
are exactly the scan's, which is what makes the two pictures comparable.

Two caveats that belong next to the numbers:

* the ceiling is not 1 for the sampled-``y`` variant.  ``p(y*|x)`` averages ~0.42 over
  these bands, so ``acc_sampled`` saturates there while ``acc_top1`` (mode match)
  saturates at 1.  The chance level of ``acc_top1`` is ``1/512 = 0.002``.
* the training-time curves are *mid-schedule checkpoints* of one cosine-annealed run
  sized for the full stream, not a family of runs each annealed at its own D.  That is
  the usual way such a plot is made, and it slightly understates the accuracy of an
  early checkpoint.
"""

from __future__ import annotations

import functools
import time as _time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import data as D
from .data import D_OUT, embed
from .problem import CACHE

# Half-open bands of context index, as ranked by frequency, with their slide labels.
BANDS = ((1, 1_001), (2_000, 3_000), (5_000, 6_000), (10_000, 11_000),
         (20_000, 30_000))
LABELS = ("1-1k", "2-3k", "5-6k", "10-11k", "20-30k")


@dataclass
class BandEval:
    """Uniformly-weighted contexts from each band, with their true mode."""

    hi: np.ndarray  # (M,) uint32
    lo: np.ndarray  # (M,) uint32
    y_star: np.ndarray  # (M,) int32, argmax_y p(y|x)
    probs: np.ndarray  # (M, 512) float32, exact p(y|x)
    entropy: np.ndarray  # (M,) nats
    index: np.ndarray  # (M,) float64, the context id
    band_id: np.ndarray  # (M,) int32

    def __len__(self) -> int:
        return len(self.hi)

    @property
    def labels(self) -> list[str]:
        return list(LABELS)

    @property
    def sizes(self) -> list[int]:
        return [int((self.band_id == i).sum()) for i in range(len(BANDS))]


def _band_file(per_band: int, seed: int) -> Path:
    return CACHE / f"band_eval2_p{per_band}_s{seed}.npz"  # 2: new conditional


def build_band_eval(per_band: int = 4096, seed: int = 23) -> BandEval:
    """All contexts of a band if it is narrow, else a uniform sample of `per_band`."""
    f = _band_file(per_band, seed)
    if f.exists():
        z = np.load(f)
        return BandEval(**{k: z[k] for k in z.files})

    rng = np.random.default_rng(seed)
    idx, bid = [], []
    for k, (a, b) in enumerate(BANDS):
        tok = (np.arange(a, b, dtype=np.int64) if b - a <= per_band
               else np.sort(rng.choice(np.arange(a, b, dtype=np.int64), per_band,
                                       replace=False)))
        idx.append(tok)
        bid.append(np.full(len(tok), k, dtype=np.int32))

    index = np.concatenate(idx)
    probs = D.conditional(index)
    hi, lo = D.split_u32(index)
    be = BandEval(hi=hi, lo=lo, y_star=probs.argmax(1).astype(np.int32), probs=probs,
                  index=index.astype(np.float64),
                  entropy=-(probs * np.log(np.maximum(probs, 1e-45))).sum(1),
                  band_id=np.concatenate(bid))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: getattr(be, k) for k in be.__dataclass_fields__})
    return be


@functools.partial(jax.jit, static_argnames=("n", "chunk"))
def _argmax_chunked(w, hi, lo, key, n: int, chunk: int = 1024):
    """(K, M) argmax_y (W e_x)_y, one chunk of contexts at a time."""
    nchunk = hi.shape[0] // chunk

    def body(i, out):
        sl = jax.lax.dynamic_slice
        h = sl(hi, (i * chunk,), (chunk,))
        l = sl(lo, (i * chunk,), (chunk,))
        e = embed(key, h, l, n)
        pred = jnp.argmax(jnp.einsum("kdn,bn->kbd", w, e), axis=-1).astype(jnp.int32)
        return jax.lax.dynamic_update_slice(out, pred, (0, i * chunk))

    out = jnp.zeros((w.shape[0], nchunk * chunk), jnp.int32)
    return jax.lax.fori_loop(0, nchunk, body, out)


def band_accuracy(params, be: BandEval, *, n: int, instance_seed: int = 0,
                  chunk: int = 1024):
    """Per-band accuracy for every config.  Returns ``(top1, sampled)``, both (K, n_bands).

    ``top1[k, m]`` is the fraction of band-m contexts whose predicted token is the true
    mode (chance 1/512); ``sampled[k, m]`` is the same prediction scored against
    ``y ~ p(.|x)`` in expectation, i.e. the mean of ``p(pred|x)`` (ceiling ``E[p*]``).
    """
    from .train import embedding_key

    m = len(be)
    pad = (-m) % chunk

    def _pad(a):
        return np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)]) if pad else a

    pred = np.asarray(_argmax_chunked(
        params, jnp.asarray(_pad(be.hi)), jnp.asarray(_pad(be.lo)),
        embedding_key(instance_seed), n, chunk))[:, :m]

    nb = len(BANDS)
    cnt = np.bincount(be.band_id, minlength=nb).astype(float)
    top1, samp = [], []
    for p in pred:
        hit = (p == be.y_star).astype(float)
        top1.append(np.bincount(be.band_id, weights=hit, minlength=nb) / cnt)
        pp = be.probs[np.arange(m), p].astype(float)
        samp.append(np.bincount(be.band_id, weights=pp, minlength=nb) / cnt)
    return np.array(top1), np.array(samp)


def checkpoints(steps: int, n_log: int = 14, n_lin: int = 10, first: int = 100):
    """Step counts to evaluate at: log-spaced (early detail) union linear (for a linear x)."""
    lg = np.unique(np.round(np.geomspace(min(first, steps), steps, n_log)).astype(int))
    ln = np.unique(np.round(np.linspace(0, steps, n_lin + 1)).astype(int))[1:]
    return sorted(set(int(x) for x in np.concatenate([lg, ln]) if 0 < x <= steps))


def run_curve(h: int, steps: int, lr: float, stream, be: BandEval, se=None, *,
              seed: int = 0, ckpts=None, tag: str = "emergence") -> dict:
    """Train one (h, steps) run and record per-band accuracy along the way.

    The trainer is :mod:`assocmem.train`'s -- same hand-written Adam, same cosine
    schedule over ``steps``, same zero init, same stream -- driven segment by segment so
    that the intermediate weights can be probed.  The stratified excess loss of the scan
    is recorded at the same checkpoints when ``se`` is given, so the smooth macro curve
    and the sharp per-band curves come from the *same* run.
    """
    from . import ledger
    from .train import BATCH, _train_segment, embedding_key, eval_flops, init_params, \
        train_flops
    from .grid import excess_loss

    ckpts = list(checkpoints(steps)) if ckpts is None else list(ckpts)
    ikey = jax.random.split(jax.random.key(seed))[0]
    ekey = embedding_key(seed)
    w = init_params(ikey, h, [0.0])
    m = jnp.zeros_like(w)
    v = jnp.zeros_like(w)
    t = jnp.zeros((), jnp.int32)
    lrs_j = jnp.asarray([lr], dtype=jnp.float32)

    t0 = _time.time()
    rows, prev = [], 0
    for c in ckpts:
        xs = stream.batches(c - prev, offset=prev * BATCH)
        w, m, v, t = _train_segment(w, m, v, t, xs, lrs_j, ekey, h, steps)
        prev = c
        top1, samp = band_accuracy(w, be, n=h, instance_seed=seed)
        row = dict(steps=c, tokens=BATCH * c, top1=[float(x) for x in top1[0]],
                   sampled=[float(x) for x in samp[0]])
        if se is not None:
            ex, _, _ = excess_loss(w, se, n=h, instance_seed=seed)
            row["excess"] = float(ex[0])
        rows.append(row)

    ev_tok = len(be) + (len(se) if se is not None else 0)
    tr = train_flops(h, steps)
    ev = eval_flops(h, ev_tok) * len(ckpts)
    ledger.log(f"{tag}-h{h}", train=tr, eval=ev, n=h, steps=steps, n_cfg=1,
               instance_seed=seed, eval_tokens=ev_tok, lrs=[float(lr)],
               loss=[rows[-1].get("excess", float("nan"))])
    return dict(h=h, steps=steps, seed=seed, lr=float(lr), n_params=D_OUT * h,
                tokens=BATCH * steps, curve=rows, flops=float(tr + ev),
                seconds=round(_time.time() - t0, 1))
