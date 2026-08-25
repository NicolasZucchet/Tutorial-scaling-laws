"""Fast sanity tests for the capacity experiment -- a couple of seconds in total.

Everything here stays at h <= 64 and a few hundred contexts, which is far below the
capacities the real sweep measures, so the file is cheap to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from assocmem.capacity import (D_TOK, TrainConfig, capacity, hebbian_accuracy,
                               instance, stores_all, train_to_saturation)


def test_instance_is_nested_and_on_the_sphere():
    """Bisection compares prefixes of one instance, so prefixes must be identical."""
    for h in (32, 64):
        e_small, y_small = instance(97, h, seed=3)
        e_big, y_big = instance(1001, h, seed=3)
        assert np.array_equal(np.asarray(e_small), np.asarray(e_big)[:97])
        assert np.array_equal(np.asarray(y_small), np.asarray(y_big)[:97])
        assert np.allclose(np.linalg.norm(np.asarray(e_big), axis=1), 1.0, atol=1e-5)
        assert np.asarray(y_big).min() >= 0 and np.asarray(y_big).max() < D_TOK
    # different seeds are different problems
    assert not np.array_equal(np.asarray(instance(50, 32, 0)[0]),
                              np.asarray(instance(50, 32, 1)[0]))


def test_hebbian_stores_a_few_and_breaks_on_many():
    """W = sum_i z_i e_i^T is exact when embeddings barely interfere, and fails later."""
    e, y = instance(4000, 64, seed=0)
    assert float(hebbian_accuracy(e[:32], y[:32])) == 1.0
    assert float(hebbian_accuracy(e[:4000], y[:4000])) < 0.5


def test_trained_beats_hebbian_at_the_same_n():
    """The whole point of the slide: one gradient step is much worse than many."""
    n, h = 600, 64  # above the Hebbian capacity (~2 h), below the trained one (~35 h)
    e, y = instance(n, h, seed=0)
    hebb = float(hebbian_accuracy(e, y))
    out = train_to_saturation(e, y, TrainConfig(lrs=(3.0,), segment=250, patience=4,
                                                max_steps=4000))
    assert hebb < 0.99, hebb  # Hebbian has already lost contexts by 9 h
    assert out.stored and out.best_acc == 1.0, (out.best_acc, out.steps)


def test_stores_all_agrees_with_a_direct_evaluation():
    e, y = instance(64, 32, seed=1)
    stored, acc, steps, capped = stores_all("hebbian", 64, 32, 1)
    assert steps == 0 and not capped
    assert (acc == float(hebbian_accuracy(e, y))) and stored == (acc >= 1.0)


def test_bisection_brackets_and_reports_a_verified_boundary():
    """The returned capacity must be a probed success, and `upper` a probed failure."""
    res = capacity("hebbian", 64, seed=0, precision=8, guess=16)
    probed = {t["n"]: t["stored"] for t in res.trace}
    assert probed[res.capacity] is True
    assert probed[res.upper] is False
    assert 0 < res.upper - res.capacity <= 8
    assert res.capacity > 64  # more contexts than dimensions, even for Hebbian
    # The 100 %-predicate is only *approximately* monotone in n -- adding a context can
    # repair another one, by strengthening its class -- so a bracket started from the
    # far side may settle on a neighbouring crossing.  Same ballpark, not same integer.
    far = capacity("hebbian", 64, seed=0, precision=8, guess=4000)
    assert 0.75 < far.capacity / res.capacity < 1.34, (far.capacity, res.capacity)
    probed_far = {t["n"]: t["stored"] for t in far.trace}
    assert probed_far[far.capacity] and not probed_far[far.upper]


def test_capped_probes_are_flagged():
    """A budget-limited verdict must announce itself rather than pass as saturation."""
    e, y = instance(3000, 64, seed=0)  # far past capacity, and a tiny step budget
    out = train_to_saturation(e, y, TrainConfig(lrs=(3.0,), segment=100, patience=99,
                                               max_steps=200))
    assert not out.stored and out.capped and out.steps == 200


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all good")
