"""Checks on the emergence experiment: the band eval set, the accuracy kernel, one run.

    PYTHONPATH=src uv run python tests/test_emergence.py     # ~15 s
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = pathlib.Path(__file__).resolve().parents[1]

from assocmem import ledger  # noqa: E402

# Never touch a real ledger from a test.
ledger.configure(path=pathlib.Path(tempfile.mkdtemp()) / "test.jsonl",
                 budget=float("inf"))

from assocmem import emergence as E  # noqa: E402
from assocmem.data import D_OUT, embed  # noqa: E402
from assocmem.train import embedding_key  # noqa: E402


def test_bands_are_what_they_claim():
    be = E.build_band_eval()
    assert len(E.BANDS) == len(E.LABELS) == 5
    for k, (a, b) in enumerate(E.BANDS):
        idx = be.index[be.band_id == k]
        assert len(idx) == len(np.unique(idx)), "contexts must not repeat within a band"
        assert idx.min() >= a and idx.max() < b, (k, idx.min(), idx.max())
        assert len(idx) == min(b - a, 4096), (k, len(idx))
    assert (be.y_star == be.probs.argmax(1)).all()
    print(f"  bands {be.labels} -> {be.sizes} contexts, uniform inside each")


def test_accuracy_kernel_matches_numpy():
    """The jitted argmax must agree with a plain numpy forward pass, tokens and all."""
    be = E.build_band_eval()
    h, seed = 96, 3
    w = jax.random.normal(jax.random.key(0), (2, D_OUT, h), jnp.float32) * 0.1
    top1, samp = E.band_accuracy(w, be, n=h, instance_seed=seed)

    e = np.asarray(embed(embedding_key(seed), jnp.asarray(be.hi), jnp.asarray(be.lo), h))
    pred = (np.asarray(w) @ e.T).argmax(1)  # (2, M)
    nb = len(E.BANDS)
    cnt = np.bincount(be.band_id, minlength=nb)
    for k in range(2):
        hit = (pred[k] == be.y_star).astype(float)
        ref = np.bincount(be.band_id, weights=hit, minlength=nb) / cnt
        assert np.abs(top1[k] - ref).max() < 1e-12, (top1[k], ref)
    # a random W is uninformed: accuracy at chance, 1/512, and the sampled variant
    # cannot exceed the ceiling E[p(y*|x)].
    assert top1.max() < 0.02, top1
    for k in range(nb):
        ceil = be.probs[be.band_id == k].max(1).mean()
        assert samp[:, k].max() <= ceil + 1e-6, (k, samp[:, k], ceil)
    print(f"  kernel == numpy forward pass             chance {top1.mean():.4f} "
          f"(1/{D_OUT} = {1 / D_OUT:.4f})")


def test_planted_memories_are_found():
    """Store 16 chosen contexts by hand; exactly those must be the ones predicted."""
    be = E.build_band_eval()
    h, seed = 512, 0
    pick = np.linspace(0, len(be) - 1, 16).astype(int)
    e = np.asarray(embed(embedding_key(seed), jnp.asarray(be.hi[pick]),
                         jnp.asarray(be.lo[pick]), h))
    w = np.zeros((1, D_OUT, h), np.float32)
    for j, i in enumerate(pick):  # Hebbian, on a handful of contexts only
        w[0, be.y_star[i]] += e[j]
    top1, _ = E.band_accuracy(jnp.asarray(w), be, n=h, instance_seed=seed)
    nb = len(E.BANDS)
    planted = np.bincount(be.band_id[pick], minlength=nb) / np.bincount(
        be.band_id, minlength=nb)
    assert np.abs(top1[0] - planted).max() < 0.01, (top1[0], planted)
    print(f"  16 planted memories recovered            per band "
          f"{np.round(top1[0], 4).tolist()}")


def test_checkpoints_are_sane():
    for steps in (100, 6400, 409_600):
        c = E.checkpoints(steps)
        assert c == sorted(set(c)) and c[0] >= 1 and c[-1] == steps, (steps, c)
        assert all(b > a for a, b in zip(c[:-1], c[1:]))
    print(f"  checkpoints monotone, end at steps       "
          f"{len(E.checkpoints(409_600))} of them at 409.6k steps")


def test_run_reproduces_the_scan():
    """Same trainer, same stream: the excess loss must match the (N, D) scan's cell."""
    from assocmem.grid import build_strat_eval
    from assocmem.problem import get_stream

    h, steps = 64, 1600
    cell = json.loads((ROOT / "results/grid.json").read_text())["cells"][f"h{h}/s{steps}/z0"]
    be, se = E.build_band_eval(), build_strat_eval()
    r = E.run_curve(h, steps, cell["lr_best"], get_stream(64 * steps), be, se,
                    ckpts=[steps // 2, steps])
    got = r["curve"][-1]["excess"]
    assert abs(got - cell["excess_best"]) < 5e-3, (got, cell["excess_best"])
    t = r["curve"][-1]["top1"]
    assert t[0] > t[-1], t  # the head is learned before the tail, always
    print(f"  one run reproduces grid cell h{h}/s{steps}    excess {got:.4f} "
          f"vs {cell['excess_best']:.4f}")


if __name__ == "__main__":
    print("emergence machinery")
    for f in (test_bands_are_what_they_claim, test_accuracy_kernel_matches_numpy,
              test_planted_memories_are_found, test_checkpoints_are_sane,
              test_run_reproduces_the_scan):
        f()
    print("ok")
