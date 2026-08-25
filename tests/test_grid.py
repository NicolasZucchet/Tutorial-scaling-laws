"""Checks on the (N, D) scan machinery: the stratified evaluator and the fits.

    PYTHONPATH=src uv run python tests/test_grid.py     # ~20 s
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import grid as G  # noqa: E402
from assocmem import grid_fit as F  # noqa: E402
from assocmem.problem import get_evalset  # noqa: E402


def test_weights_sum_to_one():
    se = G.build_strat_eval()
    assert abs(se.weight.sum() - 1.0) < 1e-9
    print(f"  weights sum to 1                        {len(se)} tokens, "
          f"{len(se.bin_lo)} strata")


def test_l_inf_agrees_with_plain_sampling():
    """The importance weights must reproduce an unbiased quantity we know another way."""
    se = G.build_strat_eval()
    ref = float(get_evalset(65536).entropy.mean())  # 65k tokens drawn from p(x)
    assert abs(se.l_inf - ref) < 0.02, (se.l_inf, ref)
    print(f"  L_inf: stratified {se.l_inf:.4f} vs p(x)-sampled {ref:.4f}")


def test_tail_mass_matches_brute_force():
    for k in (10, 1000, 100_000):
        exact = 1.0 - sum((i + 1.0) ** -1.2 for i in range(int(k))) / 5.560669
        assert abs(G.tail_mass(k) - exact) < 2e-4, (k, G.tail_mass(k), exact)
    print("  tail_mass matches a brute-force sum      k = 10 / 1e3 / 1e5")


def test_fit_recovers_planted_exponents():
    """A fit that cannot recover exponents from noise-free synthetic data is useless."""
    rng = np.random.default_rng(0)
    n = 512 * np.array([32, 64, 128, 256, 512] * 5, float)
    d = 64 * np.repeat([100, 400, 1600, 6400, 25_600], 5).astype(float)
    for form, q in (("additive", 1.0), ("power_mean", 3.0)):
        t1, t2 = 9.0 * n**-0.2, 7.0 * d**-0.1667
        y = (t1 + t2) if form == "additive" else (t1**q + t2**q) ** (1 / q)
        law = F.fit_law(n, d, y, form=form)
        assert abs(law.a_exp - 0.2) < 0.01, (form, law.a_exp)
        assert abs(law.b_exp - 0.1667) < 0.01, (form, law.b_exp)
        print(f"  {form:<11s} recovers a={law.a_exp:.4f} b={law.b_exp:.4f} "
              f"from planted (0.2000, 0.1667)")


def test_misspecified_form_biases_exponents():
    """The point of the whole exercise: the wrong form fits well and lies anyway."""
    n = 512 * np.array([32, 64, 128, 256, 512] * 5, float)
    d = 64 * np.repeat([100, 400, 1600, 6400, 25_600], 5).astype(float)
    q = 3.0
    y = ((9.0 * n**-0.2) ** q + (7.0 * d**-0.1667) ** q) ** (1 / q)
    add = F.fit_law(n, d, y, form="additive")
    assert add.rmse_log < 0.05, add.rmse_log  # looks like a good fit ...
    assert abs(add.b_exp - 0.1667) > 0.05, add.b_exp  # ... but the exponent is wrong
    print(f"  additive form on power-mean data: rel.rms {100 * add.rmse_log:.1f} % "
          f"but b = {add.b_exp:.3f} not 0.167")


def test_isoflop_groups_are_really_isoflop():
    store = {"cells": {}, "meta": {}}
    for h, s in ((32, 1600), (128, 400), (512, 100), (64, 6400)):
        store["cells"][f"h{h}/s{s}/z0"] = dict(
            h=h, steps=s, seed=0,
            excess_star=1.0 + 0.05 * (np.log(h) - np.log(128)) ** 2)
    profs = F.isoflop_profiles(store)
    assert len(profs) == 1 and profs[0]["hs"] == [32, 128, 512], profs
    assert abs(profs[0]["n_star"] / 512 - 128) < 1, profs[0]["n_star"] / 512
    want = 6 * 512 * 32 * 1600 * 64
    assert abs(profs[0]["c"] - want) < 1, (profs[0]["c"], want)
    print(f"  IsoFLOP grouping picks h={profs[0]['hs']} at C={profs[0]['c']:.2e}")


def test_isoflop_detail_recovers_planted_minima():
    """Plant parabolas with a known vertex on exact anti-diagonals and read them back.

    The deck's compute-optimal numbers all come out of this function, so it has to
    recover a vertex that sits *between* two measured widths, and the three power laws
    through the vertices, from noise-free data.
    """
    a_pref, a_exp = 1.4, 0.45  # N*(C)
    e_pref, e_exp = 12.0, 0.09  # L*(C) - L_inf
    curv = 0.35
    store = {"cells": {}}
    for j in range(6):
        k = 51_200 * 4**j
        c = 6.0 * F.D_OUT * 64 * k
        n_star = a_pref * c**a_exp
        for h in (m * 2**i for m in (1, 5, 25) for i in range(16)):
            if k % h or not (n_star / 3.2 <= F.D_OUT * h <= n_star * 3.2):
                continue
            e = e_pref * c**-e_exp + curv * np.log(F.D_OUT * h / n_star) ** 2
            store["cells"][f"h{h}/s{k // h}/z0"] = dict(
                h=h, steps=k // h, seed=0, excess_star=e, lr_star=0.01, lr_edge="")

    det = F.isoflop_detail(store)
    assert det["n_fit"] == 6 and det["n_edge"] == 0, (det["n_fit"], det["n_edge"])
    for pr in det["profiles"]:
        want = a_pref * pr["c"] ** a_exp
        assert abs(pr["n_star"] / want - 1) < 0.02, (pr["c"], pr["n_star"], want)
    laws = det["laws"]
    assert abs(laws["n_of_c"]["exp"] - a_exp) < 0.01, laws["n_of_c"]
    assert abs(laws["l_of_c"]["exp"] + e_exp) < 0.01, laws["l_of_c"]
    # the envelope exponent is the ratio of the other two, by construction
    assert abs(laws["frontier"]["exp"] + e_exp / a_exp) < 0.02, laws["frontier"]
    print(f"  isoflop_detail recovers N*~C^{laws['n_of_c']['exp']:.3f} "
          f"L*~C^{laws['l_of_c']['exp']:.4f} envelope N*^{laws['frontier']['exp']:.3f} "
          f"from planted ({a_exp}, -{e_exp}, {-e_exp / a_exp:.3f})")


def test_isoflop_detail_flags_edge_minima():
    """A profile whose vertex sits outside the widths measured must not enter the fits."""
    store = {"cells": {}}
    for j in range(6):
        k = 51_200 * 4**j
        for h in (32, 128, 512):
            if k % h:
                continue
            # monotone decreasing in h: the vertex is beyond the largest width
            e = 1.5 - 0.1 * np.log(h) + 0.004 * np.log(h) ** 2
            store["cells"][f"h{h}/s{k // h}/z0"] = dict(
                h=h, steps=k // h, seed=0, excess_star=e, lr_star=0.01, lr_edge="")
    det = F.isoflop_detail(store)
    assert det["n_fit"] == 0 and det["n_edge"] == 6, (det["n_fit"], det["n_edge"])
    assert all(p["edge"] for p in det["profiles"])
    assert all(det["laws"][k]["exp"] is None for k in det["laws"])  # nothing to fit
    print(f"  edge minima flagged and excluded          {det['n_edge']}/6 profiles")


if __name__ == "__main__":
    print("grid machinery")
    for f in (test_weights_sum_to_one, test_l_inf_agrees_with_plain_sampling,
              test_tail_mass_matches_brute_force, test_fit_recovers_planted_exponents,
              test_misspecified_form_biases_exponents,
              test_isoflop_groups_are_really_isoflop,
              test_isoflop_detail_recovers_planted_minima,
              test_isoflop_detail_flags_edge_minima):
        f()
    print("ok")
