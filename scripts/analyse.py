"""Fit the scaling laws from whatever screening rounds have been run so far.

    uv run python scripts/analyse.py round1 [round2 ...]

Prints (a) per-rung IsoFLOP optima, (b) the power laws n*(C), L*(C), and
(c) a joint fit L(n, D) = L_inf + A n^-alpha + B D^-beta with its own optimum.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import sys
from pathlib import Path

import numpy as np

from assocmem import fit

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERO_C = float(sys.argv[-1]) if sys.argv[-1].replace("e", "").replace("+", "").replace(
    ".", "").isdigit() else 5.5e12


def collect(names):
    rows, ent = [], None
    for nm in names:
        d = json.loads((RESULTS / f"{nm}.json").read_text())
        rows += d["rows"]
        ent = d.get("eval_entropy", ent)
    return rows, ent


def main(names, hero_c=5.5e12, quiet=False):
    rows, l_inf_emp = collect(names)
    cs = sorted({r["c"] for r in rows})
    iso = {}
    for c in cs:
        ns = sorted({r["n"] for r in rows if r["c"] == c})
        loss = [min(r["loss"] for r in rows if r["c"] == c and r["n"] == n) for n in ns]
        f = fit.isoflop_optimum(ns, loss)
        iso[c] = f
        if not quiet:
            print(f"C={c:9.3g}  n*={f.n_star:8.1f}  L*={f.loss_star:.4f}"
                  f"   grid n={ns} L={np.round(loss, 4)}")

    cc = np.array(cs)
    nstar = np.array([iso[c].n_star for c in cs])
    lstar = np.array([iso[c].loss_star for c in cs])
    a, b, r2 = fit.powerlaw(cc, nstar)
    print(f"\nn*(C)  = {a:.4g} * C^{b:.4f}     (r2={r2:.4f})")
    if len(cs) >= 3:
        li, aa, al = fit.saturating_powerlaw(cc, lstar, l_inf0=l_inf_emp or 2.3)
        print(f"L*(C)  = {li:.4f} + {aa:.4g} * C^-{al:.4f}   [3-param fit]")
    else:
        li = l_inf_emp
    ae, be, r2e = fit.powerlaw(cc, lstar - (l_inf_emp or 0.0))
    print(f"L*(C)  = {l_inf_emp:.4f} + {ae:.4g} * C^-{-be:.4f}  "
          f"[L_inf pinned to eval-set entropy, r2={r2e:.4f}]")

    # ---- joint fit on the two best learning rates per (C, n) cell -------------
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["c"], r["n"]), []).append(r)
    nn, dd, ll = [], [], []
    for v in cells.values():
        for r in sorted(v, key=lambda r: r["loss"])[:2]:
            nn.append(r["n"]); dd.append(r["tokens"]); ll.append(r["loss"])
    out = {}
    for name, flx in [("free", None), ("pinned", l_inf_emp)]:
        j = fit.joint_fit(nn, dd, ll, fix_l_inf=flx)
        out[name] = j
        print(f"\njoint[{name}]: L = {j.l_inf:.4f} + {j.a:.4g} n^-{j.alpha:.4f}"
              f" + {j.b:.4g} D^-{j.beta:.4f}   rmse={j.rmse:.4f}   n* ~ C^{j.n_exponent:.4f}")
        for c in list(cc) + [3.6e11, hero_c]:
            ns_, ds_, ls_ = j.optimum(float(c))
            print(f"    C={c:9.3g}: n*={ns_:7.0f} D*={ds_:10.4g} steps={ds_ / 64:8.0f} L={ls_:.4f}")
    return iso, out


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a[0].isdigit()]
    main(names or ["round1"], hero_c=HERO_C)
