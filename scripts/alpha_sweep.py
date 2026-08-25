"""Do the predicted exponents track alpha?  Repeat the fit-region grid at several tails.

The envelope calculation says the two loss exponents are pure functions of the tail
exponent alpha -- `alpha - 1` on the model axis and `1 - 1/alpha` on the data axis -- and
therefore that the compute-optimal split is too.  This runs the same grid as
`scaling_grid.py --stage A` at several alpha and reads each exponent off the corner where
its constraint binds, so nothing about the functional form is assumed.

    PYTHONPATH=src uv run python scripts/alpha_sweep.py                  # ~7 min per alpha
    PYTHONPATH=src uv run python scripts/alpha_sweep.py --alpha 1.5      # just one
    PYTHONPATH=src uv run python scripts/alpha_sweep.py --report         # collect + print

Each alpha gets its own stream, stratified eval set and grid file; alpha = 1.2 reuses the
main scan in results/grid.json rather than re-running it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import ledger  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/alpha"
SUMMARY = ROOT / "results/alpha_sweep.json"
ledger.configure(path=ROOT / "results/grid_ledger.jsonl", budget=float("inf"))

from assocmem import grid as G  # noqa: E402
from assocmem import grid_fit as F  # noqa: E402
from assocmem.data import D_OUT, GAMMA  # noqa: E402
from assocmem.fit import powerlaw  # noqa: E402
from assocmem.problem import get_evalset, get_stream  # noqa: E402

ALPHAS = (1.1, 1.3, 1.5, 1.8)
HS = (32, 64, 128, 256, 512)
STEPS = (100, 400, 1600, 6400, 25_600)
SEEDS = (0, 1, 2)

# The model-axis exponent can only be read where the *model* is the binding constraint,
# which needs D^(1/alpha) >> capacity, i.e. D >> (41h)^alpha.  At D = 1.64e6 that margin
# is 129x at alpha = 1.2 but only 1.2x at alpha = 1.8, so the corner measurement there is
# masked by the data constraint rather than wrong.  --extend adds the two smallest widths
# at 16x and 64x more data to test exactly that.
#
# That is still not enough at alpha = 1.8: 26.2M tokens (the whole cached stream) is only
# 5x past the capacity of h = 64, and the measured slope collapses onto a single function
# of the margin D^(1/alpha)/(41h), so what is missing is *margin*, not tokens.  Margin is
# linear in 1/h and only D^(1/alpha) in the data, so the two smallest widths buy at 1/4
# the compute what another 16x of tokens would -- tokens we do not have.  Hence widths
# below the main grid here.
EXT_HS = (8, 16, 32, 64)
EXT_STEPS = (102_400, 409_600)
EXT_SEEDS = (0,)


def store_path(alpha: float) -> pathlib.Path:
    if alpha == GAMMA:
        return ROOT / "results/grid.json"  # the main scan; do not re-run it
    return OUTDIR / f"grid_g{alpha:.3f}.json"


def load(alpha: float) -> dict:
    f = store_path(alpha)
    return json.loads(f.read_text()) if f.exists() else dict(meta={}, cells={})


def save(alpha: float, store: dict) -> None:
    f = store_path(alpha)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.partial")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.rename(f)


def margin(alpha: float, h: int, d: float) -> float:
    """How far past the capacity bound the data reaches: D^(1/alpha) / (41 h)."""
    return d ** (1.0 / alpha) / (F.CAP_PER_H * h)


def run_alpha(alpha: float, extend: bool = False) -> dict:
    print(f"\n=== alpha = {alpha}{' (extended D)' if extend else ''} " + "=" * 32)
    t0 = time.time()
    se = G.build_strat_eval(gamma=alpha)
    store = load(alpha)
    store["meta"] = dict(alpha=alpha, l_inf=se.l_inf,
                         l=float(np.log(D_OUT) - se.l_inf),
                         bin_lo=[float(x) for x in se.bin_lo], m_eval=len(se))
    hs, sts, seeds = ((EXT_HS, EXT_STEPS, EXT_SEEDS) if extend
                      else (HS, STEPS, SEEDS))
    stream = get_stream(64 * max(sts), gamma=alpha)
    ev = get_evalset(4096)  # only used for the trainer's own cheap progress number
    print(f"  L_inf = {se.l_inf:.4f}   l = {store['meta']['l']:.4f}   "
          f"eval = {len(se)} weighted tokens over {len(se.bin_lo)} strata")
    if extend:
        for h in hs:
            print(f"  h={h}: capacity margin {margin(alpha, h, 64 * max(sts)):.1f}x "
                  f"at D={64 * max(sts):.2e} (was "
                  f"{margin(alpha, h, 64 * max(STEPS)):.1f}x)")
    todo = [(h, s, z) for h in hs for s in sts for z in seeds
            if f"h{h}/s{s}/z{z}" not in store["cells"]]
    if not todo:
        print("  all cells present")
    for h, s, z in todo:
        c = G.run_cell(h, s, se, stream, ev, seed=z, tag=f"alpha{alpha}")
        store["cells"][f"h{h}/s{s}/z{z}"] = c
        save(alpha, store)
    save(alpha, store)
    print(f"  {len(store['cells'])} cells in {time.time() - t0:.0f} s")
    return store


def subgrid(store: dict) -> dict:
    """Restrict to HS x STEPS, so every alpha is compared on the same grid.

    alpha = 1.2 is the main scan and carries stage-B cells the other alpha do not, and a
    corner exponent read at a different corner is not the same measurement.
    """
    return dict(meta=store["meta"],
                cells={k: c for k, c in store["cells"].items()
                       if c["h"] in HS and c["steps"] in STEPS})


def summarise(alpha: float, store: dict) -> dict:
    """Corner exponents and IsoFLOP exponents, plus what theory predicts for them."""
    store = subgrid(store)
    ca, cb = F.corner_exponents(store)
    profs = [p for p in F.isoflop_profiles(store) if not p["edge"]]
    out = dict(alpha=alpha, n_cells=len(store["cells"]),
               a=ca["a"], a_at=ca["at"], b=cb["b"], b_at=cb["at"],
               a_th=F.a_th(alpha), b_th=F.b_th(alpha), l_inf=store["meta"]["l_inf"],
               excess_max=max(c["excess_star"] for c in store["cells"].values()),
               excess_min=min(c["excess_star"] for c in store["cells"].values()),
               n_profiles=len(profs))
    if len(profs) >= 3:
        cs = [p["c"] for p in profs]
        _, pn, r2n = powerlaw(cs, [p["n_star"] for p in profs])
        _, pl, r2l = powerlaw(cs, [p["excess_star"] for p in profs])
        out.update(pn=pn, r2n=r2n, pl=pl, r2l=r2l,
                   pn_th=F.b_th(alpha) / (F.a_th(alpha) + F.b_th(alpha)),
                   pl_th=-F.a_th(alpha) * F.b_th(alpha) / (F.a_th(alpha) + F.b_th(alpha)))
    out["margin"] = margin(alpha, HS[1], 64 * max(STEPS))
    return out


# How far past capacity the *wider* model of a pair has to be before its slope counts as
# a measurement of the model constraint rather than of the data one.  The slope is
# empirically a function of that margin alone -- pairs at equal margin agree to ~0.01
# whatever (h, D) produced them -- and it is within 0.02 of its plateau by 10x.
MARGIN_MIN = 10.0


def extended_corner(alpha: float, full: dict) -> dict | None:
    """The model-axis exponent, read from the widest pair that is still model-limited.

    Widest, not smallest: capacity per unit width is itself still drifting at the bottom
    of the grid (33.5 contexts/h at h = 32 against a 41.3 asymptote), which tilts the
    slope upwards, so among the pairs that clear MARGIN_MIN the widest is the least
    biased.  Whichever of EXT_HS an alpha actually has is enough -- the small widths were
    only run where the margin demanded them (alpha = 1.8).
    """
    cells = {k: c for k, c in full["cells"].items() if c["h"] in EXT_HS}
    widths: dict[int, set[int]] = {}
    for c in cells.values():
        widths.setdefault(c["steps"], set()).add(c["h"])
    steps = sorted(s for s, hs in widths.items() if len(hs) >= 2)
    if not steps or max(steps) <= max(STEPS):
        return None
    d = 64 * max(steps)
    _, dn, _ = F.local_slopes(dict(meta=full["meta"], cells=cells))
    pairs = sorted((h1, h2) for (h1, h2, st) in dn if st == max(steps))
    ok = [(h1, h2) for h1, h2 in pairs if margin(alpha, h2, d) >= MARGIN_MIN]
    if not ok:
        return None
    h1, h2 = ok[-1]
    narrower = [(a, b) for a, b in pairs if b <= h1]
    return dict(a=dn[(h1, h2, max(steps))], at=(h1, h2, d), d=d,
                margin=margin(alpha, h2, d),
                # the next pair down, kept so the deck can say what pushing further gives
                a_small=(dn[(narrower[-1][0], narrower[-1][1], max(steps))]
                         if narrower else None),
                at_small=narrower[-1] if narrower else None)


def report() -> None:
    rows = []
    for alpha in sorted(set(ALPHAS) | {GAMMA}):
        store = load(alpha)
        if store["cells"]:
            rows.append(summarise(alpha, store))
    if not rows:
        print("no alpha grids yet")
        return
    SUMMARY.write_text(json.dumps(rows, indent=1))
    print(f"\n{'alpha':>6s} {'a meas':>8s} {'a=A-1':>8s} {'ratio':>6s} "
          f"{'b meas':>8s} {'b=1-1/A':>8s} {'ratio':>6s} "
          f"{'N* meas':>8s} {'N* th':>7s} {'excess range':>15s}")
    for r in rows:
        pn = f"{r['pn']:8.4f}" if "pn" in r else f"{'-':>8s}"
        pnt = f"{r['pn_th']:7.4f}" if "pn" in r else f"{'-':>7s}"
        print(f"{r['alpha']:6.2f} {r['a']:8.4f} {r['a_th']:8.4f} "
              f"{r['a'] / r['a_th']:6.2f} {r['b']:8.4f} {r['b_th']:8.4f} "
              f"{r['b'] / r['b_th']:6.2f} {pn} {pnt} "
              f"{r['excess_min']:6.3f}..{r['excess_max']:6.3f}")
    print()
    for r in rows:
        ext = extended_corner(r["alpha"], load(r["alpha"]))
        if ext:
            small = ("" if ext["a_small"] is None else
                     f"; {ext['at_small'][0]}->{ext['at_small'][1]} gives "
                     f"{ext['a_small']:.4f}")
            print(f"  alpha={r['alpha']:.2f}: model axis at h={ext['at'][0]}->{ext['at'][1]}, "
                  f"D={ext['d']:.2e} (margin {ext['margin']:.1f}x, was {r['margin']:.1f}x):  "
                  f"a = {ext['a']:.4f} vs {r['a']:.4f} before, "
                  f"predicted {r['a_th']:.4f}  ->  ratio "
                  f"{ext['a'] / r['a_th']:.2f} (was {r['a'] / r['a_th']:.2f}){small}")
            r["a_ext"], r["a_ext_d"] = ext["a"], ext["d"]
            r["a_ext_margin"], r["a_ext_at"] = ext["margin"], list(ext["at"])
            r["a_ext_small"] = ext["a_small"]
            r["a_ext_at_small"] = list(ext["at_small"]) if ext["at_small"] else None
    SUMMARY.write_text(json.dumps(rows, indent=1))
    print(f"\n  a is read at (h1->h2, D) per row: "
          + ", ".join(f"{r['alpha']}: {r['a_at']}" for r in rows))
    print(f"  b is read at (h, D1->D2) per row: "
          + ", ".join(f"{r['alpha']}: {r['b_at']}" for r in rows))
    print(f"\nwrote {SUMMARY.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="*", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--extend", action="store_true",
                    help="add small-h / large-D cells, to un-mask the model axis")
    args = ap.parse_args()
    if not args.report:
        for alpha in (args.alpha if args.alpha else ALPHAS):
            if alpha == GAMMA and not args.extend:
                print(f"alpha = {GAMMA} is the main scan (results/grid.json); skipping")
                continue
            run_alpha(alpha, extend=args.extend)
        print(f"\n{ledger.report().splitlines()[0]}")
    report()


if __name__ == "__main__":
    main()
