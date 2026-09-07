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
from assocmem.lab import FORM_URL
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
    s = Sweep(c=[1e8, 2e8], n=[D_OUT * 16, D_OUT * 32], lr=[0.05, 0.1])
    assert len(s) == 8 and len(s.groups()) == 4
    assert len(s + s) == 16
    assert s.configs[0].params == D_OUT * 16 and s.configs[0].width == 16
    for bad in (dict(c=[1e8], d=[640], n=[8192], lr=[0.1]),   # all three axes
                dict(n=[8192], lr=[0.1]),                     # only one
                dict(c=[1e8], n=[8192], lr=[-1.0])):          # bad lr
        try:
            Sweep(**bad)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass
    try:  # n so large the run would be under one batch of tokens
        Sweep(c=[1e8], n=[10**7], lr=[0.1])
        raise AssertionError("should have rejected an under-one-batch config")
    except ValueError as e:
        assert "tokens" in str(e), e
    try:  # d below one batch
        Sweep(d=[32], n=[8192], lr=[0.1])
        raise AssertionError("should have rejected d < BATCH")
    except ValueError as e:
        assert "batch" in str(e), e
    # d is snapped to whole batches, n to whole embedding dimensions
    s2 = Sweep(n=[8192], d=[1000], lr=[0.1])
    assert s2.configs[0].tokens == 1024, s2.configs[0].tokens
    assert Sweep(n=[8192], d=[1024], lr=[0.1]).configs[0].steps == 16
    assert Sweep(n=[8000], d=[1024], lr=[0.1]).configs[0].params == 8192

    # any two of (c, n, d) fix the third
    assert Sweep(c=[6 * 8192 * 1024], d=[1024], lr=[0.1]).configs[0].params == 8192
    assert Sweep(c=[6 * 8192 * 1024], n=[8192], lr=[0.1]).configs[0].tokens == 1024
    assert Sweep(n=[8192], d=[1024], lr=[0.1]).configs[0].c == 6 * 8192 * 1024

    # lr as a function of (c, n, d), and of c alone
    assert Sweep(n=[8192], d=[1024], lr=lambda c, n, d: 1e-6 * n).configs[0].lr == 8192e-6
    assert Sweep(n=[8192], d=[1024], lr=lambda c: 2.0 * c**-0.5).configs[0].lr > 0
    try:
        Sweep(n=[8192], d=[1024], lr=lambda c, n, d: -1.0)
        raise AssertionError("a negative lr from a function was allowed")
    except ValueError as e:
        assert "lr function" in str(e), e


def test_flop_accounting_is_exact():
    s = Sweep(n=[D_OUT * 32], d=[640], lr=[0.1])
    cfg = s.configs[0]
    assert cfg.flops == 6 * cfg.params * cfg.tokens
    assert cfg.flops == 6 * (D_OUT * 32) * 64 * 10
    assert s.cost(eval_tokens=0) == cfg.flops


def test_lab_lifecycle():
    tmp = Path(tempfile.mkdtemp())
    try:
        lab = Lab("t", budget=4e9, rounds=2, root=tmp, quiet=True)
        s = Sweep(c=[1e8], n=[D_OUT * 16, D_OUT * 32], lr=[0.05, 0.1])

        assert not s.estimate(lab, quiet=True).fits or lab.remaining > 0
        r = lab.run_round("r1", s, plot=False)
        assert len(r) == 4 and lab.rounds_used == 1
        spent = lab.spent

        again = lab.run_round("r1", s, plot=False)  # cache: free, no round
        assert len(again) == 4 and lab.rounds_used == 1 and lab.spent == spent

        reloaded = Lab("t", root=tmp, quiet=True)  # state survives a restart
        assert reloaded.rounds_used == 1 and abs(reloaded.spent - spent) < 1

        try:  # over budget -> refused, with advice
            lab.run_round("huge", Sweep(c=[1e12], n=[D_OUT * 64, D_OUT * 128], lr=[0.05, 0.1]),
                          plot=False)
            raise AssertionError("should have refused")
        except BudgetError as e:
            assert "too big" in str(e) and "Options" in str(e)

        lab.run_round("r2", Sweep(c=[2e8], n=[D_OUT * 16, D_OUT * 32], lr=[0.05]), plot=False)
        try:  # out of rounds
            lab.run_round("r3", Sweep(c=[1e8], n=[D_OUT * 8], lr=[0.05]), plot=False)
            raise AssertionError("should have refused")
        except BudgetError as e:
            assert "rounds left" in str(e)

        lab.reset(confirm=True)
        assert lab.rounds_used == 0 and lab.spent == 0 and lab.rows == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_isoflop_and_powerlaw_recover_a_known_optimum():
    """Synthetic L(n, D) with a known optimum -> the pieces a student fits must recover it.

    excess = 3 n^-0.2 + 3 D^-0.2 with D = C/(6*512*n) has the closed-form optimum
    n* = (C/3072)^0.5 and excess* = 6 (C/3072)^-0.1.  There is no canned `lab.fit()` any
    more, so what is tested is the route the notebook takes: IsoFLOP optima, then a power
    law through them, then a saturating power law for L*(C).
    """
    from assocmem.fit import powerlaw, saturating_powerlaw
    from assocmem.lab import Results

    l_inf, rows = 2.0, []
    ns = [2**k for k in range(8, 17)]  # 256 .. 65536, brackets every rung's optimum
    for c in (1e9, 1e10, 1e11, 1e12):
        for n in ns:
            d = c / (6 * D_OUT * n)
            loss = l_inf + 3.0 * n**-0.2 + 3.0 * d**-0.2
            rows.append(dict(c=c, n=n, steps=max(1, int(d / 64)), tokens=d, lr=0.05,
                             init=0.0, seed=0, loss=loss))
    iso = Results(rows).isoflop()
    assert len(iso) == 4 and not any(d["clipped"] for d in iso), iso
    for d in iso:
        assert abs(d["n_star"] / (d["c"] / 3072) ** 0.5 - 1) < 0.05, d

    a, b, r2 = powerlaw([d["c"] for d in iso], [d["n_star"] for d in iso])
    assert abs(b - 0.5) < 0.03 and r2 > 0.999, (a, b, r2)
    # the floor is fitted, not given: it has to come back out of the rungs
    li, A, alpha = saturating_powerlaw([d["c"] for d in iso], [d["loss_star"] for d in iso])
    assert abs(li - l_inf) < 0.1, (li, l_inf)
    assert abs(alpha - 0.1) < 0.015, alpha
    for c in (1e10, 1e13):
        assert abs(a * c**b / (c / 3072) ** 0.5 - 1) < 0.15, c
        assert abs(li + A * c**-alpha - (l_inf + 6 * (c / 3072) ** -0.1)) < 0.05, c


def test_isoflop_flags_a_grid_that_misses_the_optimum():
    """A grid whose best n sits at an edge must be flagged, not silently believed."""
    from assocmem.lab import Results

    l_inf, rows = 2.0, []
    for c in (1e9, 1e10, 1e11):
        for n in (16, 32, 64):  # far below n* = (C/3072)^0.5
            d = c / (6 * D_OUT * n)
            loss = l_inf + 3.0 * n**-0.2 + 3.0 * d**-0.2
            rows.append(dict(c=c, n=n, steps=max(1, int(d / 64)), tokens=d,
                             lr=0.05, init=0.0, seed=0, loss=loss))
    iso = Results(rows).isoflop()
    assert len(iso) == 3, iso
    assert all(d["clipped"] in ("high", "flat") for d in iso), [d["clipped"] for d in iso]


def test_joint_fit_forms_each_recover_their_own_truth():
    """All three parametric forms, each fitted to data generated from itself."""
    from assocmem.fit import joint_fit

    ns = np.array([2**k for k in range(11, 19)], float)
    ds = np.array([10**x for x in np.linspace(5, 8, 6)])
    n, d = (v.ravel() for v in np.meshgrid(ns, ds))

    chin = 2.0 + 3.0 * n**-0.2 + 3.0 * d**-0.25
    j = joint_fit(n, d, chin, c=6 * n * d, form="chinchilla")
    assert j.form == "chinchilla" and j.k == 1.0 and j.r2 > 0.999, j
    assert abs(j.alpha - 0.2) < 0.02 and abs(j.beta - 0.25) < 0.02, str(j)
    assert abs(j.l_inf - 2.0) < 0.05, str(j)

    kap = ((1e12 / n) ** (0.1 / 0.15) + 1e11 / d) ** 0.15
    k = joint_fit(n, d, kap, c=6 * n * d, form="kaplan")
    assert k.form == "kaplan" and k.r2 > 0.999, k
    assert abs(k.alpha - 0.1) < 0.01 and abs(k.beta - 0.15) < 0.01, str(k)
    assert abs(k.n_exponent - 0.6) < 0.02, k.n_exponent
    assert np.allclose(k.predict(n, d), kap, rtol=2e-3)

    # skaling: Chinchilla's floor and exponents with Kaplan's coupling, the coupling being
    # the fitted k -- so the whole point is that k comes back as something other than 1
    sk = 1.5 + (40.0 * n**-0.3 + 90.0 * d**-0.35) ** 0.6
    s = joint_fit(n, d, sk, c=6 * n * d, form="skaling")
    assert s.form == "skaling" and s.r2 > 0.9999, s
    assert abs(s.k - 0.6) < 0.02, str(s)
    assert abs(s.alpha - 0.3) < 0.02 and abs(s.beta - 0.35) < 0.02, str(s)
    assert abs(s.l_inf - 1.5) < 0.05, str(s)
    assert np.allclose(s.predict(n, d), sk, rtol=1e-3)
    # and it beats the additive form on its own data, which is the claim being made
    assert s.rmse < 0.1 * joint_fit(n, d, sk, form="chinchilla").rmse


def test_scaling_is_not_a_form_name():
    """'scaling' names two different papers' forms, so it has to be spelled out."""
    import pytest

    from assocmem.fit import joint_fit

    n = np.array([1e4, 1e5, 1e6, 1e4, 1e5, 1e6])
    d = np.array([1e5, 1e5, 1e5, 1e6, 1e6, 1e6])
    loss = 1.0 + 2.0 * n**-0.2 + 2.0 * d**-0.2
    with pytest.raises(ValueError, match="ambiguous"):
        joint_fit(n, d, loss, form="scaling")
    # the aliases that are not ambiguous still resolve
    assert joint_fit(n, d, loss, form="coupled").form == "skaling"
    assert joint_fit(n, d, loss, form="additive").form == "chinchilla"


def test_one_correct_answer_variant():
    """The demo variant: point-mass labels, zero irreducible loss, task variant intact."""
    from assocmem.data import conditional, sample_labels

    tok = sample_tokens(64, seed=1)
    p_det = conditional(tok, deterministic=True)
    assert np.allclose(p_det.max(1), 1.0), p_det.max(1)[:5]
    assert np.allclose(target_entropy(tok, deterministic=True), 0.0, atol=1e-12)
    # labels are the mode, and the same on every occurrence of a token
    y = sample_labels(np.repeat(tok, 3), seed=7, deterministic=True).reshape(-1, 3)
    assert np.all(y == y[:, :1]), "a deterministic label changed between occurrences"
    assert np.all(y[:, 0] == p_det.argmax(1))
    # the graded problem is untouched
    assert target_entropy(tok).mean() > 1.0

    d = Path(tempfile.mkdtemp())
    try:
        lab = Lab("det", budget=4e9, rounds=1, root=d, quiet=True, deterministic=True)
        assert not hasattr(lab, "l_inf"), "the lab must not hand out the irreducible loss"
        lab.run_round("r", Sweep(c=[2e8], n=[D_OUT * 16, D_OUT * 32], lr=[0.1]), plot=False)
        assert all(r["loss"] > 0 for r in lab.rows)
        # a lab remembers its variant, and refuses to mix the two
        try:
            Lab("det", budget=4e9, rounds=1, root=d, quiet=True, deterministic=False)
        except BudgetError as e:
            assert "variant" in str(e), e
        else:
            raise AssertionError("mixing variants in one lab was allowed")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_results_reductions_and_plot_runs():
    """envelope / lr_optima / plot_runs over a hand-made set of rows."""
    import matplotlib.pyplot as plt

    from assocmem import plot_runs
    from assocmem.lab import Results

    rows = []
    for c in (1e9, 4e9):
        for n in (D_OUT * 32, D_OUT * 64, D_OUT * 128):
            for lr in (0.05, 0.1, 0.2):
                # a parabola in log lr with its minimum at 0.1, and a floor falling in c
                loss = 2.0 - 0.2 * np.log10(c / 1e9) + 0.3 * (np.log(lr / 0.1)) ** 2
                rows.append(dict(key=f"{c}-{n}-{lr}", c=c, n=n, steps=100, tokens=6400,
                                 lr=lr, init=0.0, seed=0, loss=float(loss),
                                 flops=c, round=1))
    res = Results(rows)
    assert len(res.envelope()) == 6, len(res.envelope())
    assert {o["where"] for o in res.lr_optima()} == {"interior"}
    assert all(abs(o["lr_star"] - 0.1) < 1e-6 for o in res.lr_optima())

    ax = plot_runs(res, x="n", y="loss", color="c", show=False)
    assert ax is not None
    plt.close("all")
    # excess needs an l_inf from somewhere
    plot_runs(res, x="tokens_per_param", y="loss", color="c", excess=True, l_inf=1.5,
              show=False)
    plt.close("all")
    try:
        plot_runs(res, x="n", y="loss", excess=True, show=False)
    except ValueError as e:
        assert "l_inf" in str(e), e
    else:
        raise AssertionError("excess without l_inf was allowed")


def test_hero_evaluations_are_measured_but_not_charged():
    """`bill_eval=False` keeps scoring out of the budget without hiding it.

    The hero is scored on a much larger eval set than a screening run, twice, plus a
    learning curve -- charging that would quietly take a rung's worth of compute out of
    the run the student is actually buying.  So it is logged as `eval_free`: visible in
    the ledger and the report, absent from the total the budget is measured against.
    """
    from assocmem import ledger
    from assocmem.problem import get_evalset, get_stream
    from assocmem.train import train_sweep

    tmp = Path(tempfile.mkdtemp())
    try:
        ledger.configure(tmp / "ledger.jsonl", 1e13)
        stream, evals = get_stream(64 * 4), get_evalset(256)
        r = train_sweep(n=2, steps=4, lrs=[0.1], stream=stream, eval_set=evals,
                        eval_tokens=256, eval_points=2, tag="free", bill_eval=False)
        t = ledger.total()
        assert r.eval_flops > 0, r.eval_flops        # it was computed
        assert t["eval"] == 0.0, t                   # and not charged
        assert abs(t["eval_free"] - r.eval_flops) < 1, t
        assert abs(t["total"] - r.train_flops) < 1, t
        assert "uncharged" in ledger.report()

        # the default still bills it, which is what every screening round relies on
        r2 = train_sweep(n=2, steps=4, lrs=[0.2], stream=stream, eval_set=evals,
                         eval_tokens=256, eval_points=2, tag="billed")
        t2 = ledger.total()
        assert abs(t2["eval"] - r2.eval_flops) < 1, t2
        assert t2["eval_free"] == t["eval_free"], t2
    finally:
        ledger.configure(Path(__file__).resolve().parents[1] / "results" / "ledger.jsonl",
                         ledger.BUDGET)
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_of_a_hero_run():
    """lab.report() reads the hero record; the prefilled link carries the numbers.

    And carries no name: the form does not ask for one, so the scoreboard is a
    cloud of anonymous dots.

    No hero run is trained here -- one costs the whole budget.  The record is the
    dict lab.hero() writes, so what is tested is the arithmetic and the URL, which
    is where a typo would send the room's dots to the wrong form.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        lab = Lab("rep", budget=1e13, rounds=1, root=tmp, quiet=True)
        try:
            lab.report()
            raise AssertionError("should have refused: no hero run")
        except BudgetError as e:
            assert "no hero run" in str(e), e

        lab.hero_record = dict(predicted=2.5123456, loss=2.5312345,
                               c_train=5e12, c_eval_free=1e11)
        out = lab.report()
        assert "name" not in out, "the form asks for no name; the plot is anonymous"
        assert out["predicted"] == 2.5123 and out["actual"] == 2.5312, out
        # A fraction, not a percentage: 5e12 / 1e13, which is what the form's own
        # 0-to-1 validation accepts from a prefilled link.  The hero's own scoring
        # (c_eval_free) is deliberately NOT in there -- it is not charged.
        assert out["share"] == 0.5, out
        assert out["url"].startswith(FORM_URL), out

        out = lab.report(url="https://docs.google.com/forms/d/e/XYZ/viewform?edit",
                         fields={"predicted": "entry.2", "actual": "entry.3",
                                 "share": "entry.4"})
        assert out["url"] == (
            "https://docs.google.com/forms/d/e/XYZ/viewform"
            "?usp=pp_url&entry.2=2.5123&entry.3=2.5312&entry.4=0.5"), out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all good")
