"""Turn the three screening rounds into the hero-run recipe.

Steps
  1. IsoFLOP optimum (n*, L*) per rung, with a small correction for rungs whose lr
     grid was clipped away from lr*.
  2. power laws  n*(C), lr*(C), L*(C).
  3. leave-one-out check of the L*(C) extrapolation.
  4. size the hero run to the remaining budget and predict its loss.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np

from assocmem import fit, ledger

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
ROUNDS = ["round1", "round2", "round3"]
EVAL_RESERVE = 3.4e11  # kept back for the hero's own evaluations


def load_rows():
    rows, ent = [], None
    for nm in ROUNDS:
        d = json.loads((RESULTS / f"{nm}.json").read_text())
        rows += d["rows"]
        ent = d.get("eval_entropy", ent)
    return rows, ent


def lr_curvature(rows, c, n):
    """d^2 L / d(log lr)^2 from the round-3 lr parabola."""
    sel = sorted([r for r in rows if r["c"] == c and r["n"] == n], key=lambda r: r["lr"])
    x = np.log([r["lr"] for r in sel]); y = np.array([r["loss"] for r in sel])
    return float(np.polyfit(x, y, 2)[0])


def main():
    rows, l_inf = load_rows()
    cs = sorted({r["c"] for r in rows})

    # ---- 1. lr* law from the two rungs where the grid bracketed the optimum -----
    curv = lr_curvature(rows, 3.0e11, 560)
    lr_anchors = [(4.0e9, 0.0764), (3.0e11, 0.0273)]
    a_lr, p_lr, _ = fit.powerlaw([c for c, _ in lr_anchors], [v for _, v in lr_anchors])
    print(f"lr*(C) = {a_lr:.4g} * C^{p_lr:.4f}      (curvature d2L/dlog(lr)^2 = {curv:.4f})")

    # ---- 2. IsoFLOP optima, with an lr-clipping correction ----------------------
    print(f"\n{'C':>10} {'n*':>8} {'L* raw':>8} {'lr used':>8} {'lr*':>7} {'pen':>6} {'L* cor':>8}")
    nstar, lstar, lstar_raw = [], [], []
    for c in cs:
        ns = sorted({r["n"] for r in rows if r["c"] == c})
        cells = [min((r for r in rows if r["c"] == c and r["n"] == n), key=lambda r: r["loss"])
                 for n in ns]
        loss = [cc["loss"] for cc in cells]
        f = fit.isoflop_optimum(ns, loss)
        # lr penalty at the cell nearest the optimum
        near = min(cells, key=lambda cc: abs(np.log(cc["n"] / f.n_star)))
        lr_opt = a_lr * c**p_lr
        pen = curv * np.log(near["lr"] / lr_opt) ** 2
        nstar.append(f.n_star); lstar_raw.append(f.loss_star); lstar.append(f.loss_star - pen)
        print(f"{c:10.3g} {f.n_star:8.1f} {f.loss_star:8.4f} {near['lr']:8.4g} "
              f"{lr_opt:7.4f} {pen:6.4f} {f.loss_star - pen:8.4f}")

    cs_a, nstar, lstar = np.array(cs), np.array(nstar), np.array(lstar)

    # ---- 3. power laws ---------------------------------------------------------
    a_n, b_n, r2_n = fit.powerlaw(cs_a, nstar)
    print(f"\nn*(C)  = {a_n:.4g} * C^{b_n:.4f}   (r2={r2_n:.5f})")
    aL, bL, r2L = fit.powerlaw(cs_a, lstar - l_inf)
    print(f"L*(C)  = {l_inf:.4f} + {aL:.4g} * C^-{-bL:.4f}   (r2={r2L:.5f}) "
          f"[L_inf = eval-set entropy]")
    li3, a3, al3 = fit.saturating_powerlaw(cs_a, lstar, l_inf0=l_inf)
    print(f"L*(C)  = {li3:.4f} + {a3:.4g} * C^-{al3:.4f}          [3-param, L_inf free]")

    # ---- 4. leave-one-out: fit on the 4 cheapest rungs, predict the 5th ---------
    aL4, bL4, _ = fit.powerlaw(cs_a[:-1], lstar[:-1] - l_inf)
    an4, bn4, _ = fit.powerlaw(cs_a[:-1], nstar[:-1])
    print(f"\nLOO check (fit rungs 1-4, predict rung 5 @ C={cs_a[-1]:.3g}):")
    print(f"   L*: predicted {l_inf + aL4 * cs_a[-1] ** bL4:.4f}  actual {lstar[-1]:.4f}  "
          f"(err {l_inf + aL4 * cs_a[-1] ** bL4 - lstar[-1]:+.4f})")
    print(f"   n*: predicted {an4 * cs_a[-1] ** bn4:7.1f}  actual {nstar[-1]:7.1f}")

    # ---- 5. hero sizing -------------------------------------------------------
    remaining = ledger.total()["remaining"]
    c_hero = remaining - EVAL_RESERVE
    n_hero = a_n * c_hero**b_n
    n_hero = int(round(n_hero / 8) * 8)
    steps = fit.steps_for(c_hero, n_hero)
    c_actual = fit.FLOPS_PER_ND * n_hero * steps * 64
    lr_hero = a_lr * c_actual**p_lr
    pred_pinned = l_inf + aL * c_actual**bL
    pred_free = li3 + a3 * c_actual**-al3
    print(f"\n=== HERO ===")
    print(f"  budget left {remaining:.4g}  ->  train {c_actual:.4g} (+{EVAL_RESERVE:.3g} eval)")
    print(f"  n      = {n_hero}   (N = {512 * n_hero:,} params)")
    print(f"  steps  = {steps}   (D = {steps * 64:,} tokens, {steps * 64 / (512 * n_hero):.2f} tok/param)")
    print(f"  lr_max = {lr_hero:.4f}  -> lr_min = {lr_hero / 10:.5f}, cosine, zero init")
    print(f"  predicted loss = {pred_pinned:.4f} nats  [pinned]   {pred_free:.4f} nats  [free]")

    out = dict(lr_law=dict(a=a_lr, p=p_lr), curvature=curv, l_inf=l_inf,
               rungs=[dict(c=float(c), n_star=float(n), l_star=float(l))
                      for c, n, l in zip(cs_a, nstar, lstar)],
               l_star_raw=[float(x) for x in lstar_raw],
               n_law=dict(a=a_n, b=b_n, r2=r2_n), l_law=dict(a=aL, b=bL, r2=r2L),
               l_law_free=dict(l_inf=li3, a=a3, alpha=al3),
               hero=dict(n=n_hero, steps=steps, lr=lr_hero, c_train=c_actual,
                         predicted_loss=pred_pinned, predicted_loss_free=pred_free))
    (RESULTS / "final_fit.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n-> results/final_fit.json")
    return out


if __name__ == "__main__":
    main()
