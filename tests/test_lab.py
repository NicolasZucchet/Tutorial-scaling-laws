"""Fast sanity tests -- run `uv run python -m pytest tests` (or just execute this file).

Everything here uses n<=32 and <=40 steps, so the whole file costs ~1e8 flops: less
than 0.002 % of a tutorial budget, spent in a throwaway lab.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from assocmem import BudgetError, Lab, Sweep
from assocmem.data import D_OUT, conditional, sample_tokens, target_entropy, vocab_size


def test_problem_definition():
    v = vocab_size()
    assert 1e11 < v < 2e11, v
    tok = sample_tokens(200_000, seed=0)
    z = 5.5606919
    for i in (1, 2, 5, 100):  # empirical pmf matches i^-gamma / Z
        assert abs(np.mean(tok == i) - i**-1.2 / z) < 4e-3, i
    h = target_entropy(np.arange(1, 100_001))
    assert abs(np.exp(h).mean() - 16.0) < 0.5, np.exp(h).mean()  # E[exp H] = d/32
    p = conditional(np.arange(1, 501))
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
    ent = -(p * np.log(np.maximum(p, 1e-45))).sum(1)
    assert np.abs(ent - target_entropy(np.arange(1, 501))).max() < 1e-5


def test_sweep_expansion_and_validation():
    s = Sweep(c=[1e8, 2e8], n=[16, 32], lr=[0.05, 0.1])
    assert len(s) == 8 and len(s.groups()) == 4
    assert len(s + s) == 16
    for bad in (dict(c=[1e8], steps=[10], n=[16], lr=[0.1]),  # both time axes
                dict(n=[16], lr=[0.1]),                       # neither
                dict(c=[1e8], n=[16], lr=[-1.0])):            # bad lr
        try:
            Sweep(**bad)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass
    try:  # n so large the run would be < 1 step
        Sweep(c=[1e8], n=[10**6], lr=[0.1])
        raise AssertionError("should have rejected an under-1-step config")
    except ValueError as e:
        assert "steps" in str(e)


def test_flop_accounting_is_exact():
    s = Sweep(n=[32], steps=[10], lr=[0.1])
    cfg = s.configs[0]
    assert cfg.flops == 6 * (D_OUT * 32) * 64 * 10
    assert s.cost(eval_tokens=0) == cfg.flops


def test_lab_lifecycle():
    tmp = Path(tempfile.mkdtemp())
    try:
        lab = Lab("t", budget=4e9, rounds=2, root=tmp, quiet=True)
        s = Sweep(c=[1e8], n=[16, 32], lr=[0.05, 0.1])

        assert not s.estimate(lab, quiet=True).fits or lab.remaining > 0
        lab.run_round("smoke", s, smoke=True, plot=False)
        assert lab.rounds_used == 0, "a smoke run must not cost a round"

        r = lab.run_round("r1", s, plot=False)
        assert len(r) == 4 and lab.rounds_used == 1
        spent = lab.spent

        again = lab.run_round("r1", s, plot=False)  # cache: free, no round
        assert len(again) == 4 and lab.rounds_used == 1 and lab.spent == spent

        reloaded = Lab("t", root=tmp, quiet=True)  # state survives a restart
        assert reloaded.rounds_used == 1 and abs(reloaded.spent - spent) < 1

        try:  # over budget -> refused, with advice
            lab.run_round("huge", Sweep(c=[1e12], n=[64, 128, 256], lr=[0.05, 0.1]),
                          plot=False)
            raise AssertionError("should have refused")
        except BudgetError as e:
            assert "too big" in str(e) and "Options" in str(e)

        lab.run_round("r2", Sweep(c=[2e8], n=[16, 32], lr=[0.05]), plot=False)
        try:  # out of rounds
            lab.run_round("r3", Sweep(c=[1e8], n=[8], lr=[0.05]), plot=False)
            raise AssertionError("should have refused")
        except BudgetError as e:
            assert "rounds left" in str(e)

        lab.reset(confirm=True)
        assert lab.rounds_used == 0 and lab.spent == 0 and lab.rows == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fit_recovers_a_known_power_law():
    """Synthetic L(n, D) with a known optimum -> the fitter must recover its exponents.

    excess = 3 n^-0.2 + 3 D^-0.2 with D = C/(6*512*n) has the closed-form optimum
    n* = (C/3072)^0.5 and excess* = 6 (C/3072)^-0.1, plus a quadratic lr penalty
    centred on lr*(C) = 2.5 C^-0.2.
    """
    from assocmem.lab import fit_laws

    def lr_star(c):
        return 2.5 * c**-0.2

    l_inf, rows = 2.0, []
    ns = [2**k for k in range(8, 17)]  # 256 .. 65536, brackets every rung's optimum
    for c in (1e9, 1e10, 1e11, 1e12):
        for n in ns:
            d = c / (6 * D_OUT * n)
            loss = l_inf + 3.0 * n**-0.2 + 3.0 * d**-0.2
            for f in (1 / 1.7, 1.0, 1.7):  # lr grid centred on the truth
                lr = lr_star(c) * f
                rows.append(dict(c=c, n=n, steps=max(1, int(d / 64)), tokens=d, lr=lr,
                                 init=0.0, seed=0, loss=loss + 0.08 * np.log(f) ** 2))
    laws = fit_laws(rows, l_inf)
    assert not any("WARNING" in t for t in laws.notes), laws.notes
    assert abs(laws.n_law[1] - 0.5) < 0.03, laws.n_law
    assert abs(laws.loss_law[1] - 0.1) < 0.015, laws.loss_law
    assert abs(laws.lr_law[1] - (-0.2)) < 0.03, laws.lr_law
    for c in (1e10, 1e13):
        assert abs(laws.n_star(c) / (c / 3072) ** 0.5 - 1) < 0.15, c
        assert abs(laws.predict(c) - (l_inf + 6 * (c / 3072) ** -0.1)) < 0.05, c


def test_fit_warns_when_the_grid_misses_the_optimum():
    """A grid whose best n sits at an edge must be flagged, not silently believed."""
    from assocmem.lab import fit_laws

    l_inf, rows = 2.0, []
    for c in (1e9, 1e10, 1e11):
        for n in (16, 32, 64):  # far below n* = (C/3072)^0.5
            d = c / (6 * D_OUT * n)
            loss = l_inf + 3.0 * n**-0.2 + 3.0 * d**-0.2
            for f in (1 / 1.7, 1.0, 1.7):
                rows.append(dict(c=c, n=n, steps=max(1, int(d / 64)), tokens=d,
                                 lr=2.5 * c**-0.2 * f, init=0.0, seed=0,
                                 loss=loss + 0.08 * np.log(f) ** 2))
    notes = fit_laws(rows, l_inf).notes
    assert sum("WARNING" in t for t in notes) == 3, notes
    assert "Widen the n grid" in " ".join(notes)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all good")
