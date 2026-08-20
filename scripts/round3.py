"""Screening round 3 -- top rung (C=3e11) with a proper lr parabola.

Rounds 1-2 gave  n*(C) = 0.00231 C^0.469  (r2=0.998) and excess loss ~ C^-0.095
(r2=0.9998), but lr* sat on the low edge of the grid at the two largest rungs, so
those L* are over-estimates.  lr* drifts as roughly C^-0.21, predicting ~0.030 here.

At every rung so far the n-ranking was identical at every lr, so we spend the
budget as: a 3-point lr parabola at the predicted n*, plus two n-probes at the
centre lr to get n*(3e11) directly.
Cap: 1.6e12 flops.
"""

import numpy as np

from _common import (EVAL_TOKENS, STREAM_TOKENS, fit, get_evalset, get_stream,
                     isoflop_rung, ledger, preflight, save, train_sweep)

CAP = 1.6e12
C = 3.0e11
N_CENTRE = 560  # = 0.00231 * C^0.469
LRS = [0.016, 0.028, 0.048]
LR_MID = 0.028
N_PROBES = [280, 1120]

jobs = [(N_CENTRE, fit.steps_for(C, N_CENTRE), len(LRS))]
jobs += [(n, fit.steps_for(C, n), 1) for n in N_PROBES]
preflight(jobs, CAP)

stream = get_stream(STREAM_TOKENS)
evals = get_evalset(EVAL_TOKENS)

rows = isoflop_rung(C, [N_CENTRE], LRS, tag="R3-lr", stream=stream, evals=evals)
rows += isoflop_rung(C, N_PROBES, [LR_MID], tag="R3-isoflop", stream=stream, evals=evals)

lr_row = sorted([r for r in rows if r["n"] == N_CENTRE], key=lambda r: r["lr"])
x = np.log([r["lr"] for r in lr_row])
y = [r["loss"] for r in lr_row]
co = np.polyfit(x, y, 2)
lr_star = float(np.exp(-co[1] / (2 * co[0]))) if co[0] > 0 else float(lr_row[int(np.argmin(y))]["lr"])
print(f"\nlr parabola at n={N_CENTRE}, C={C:.3g}:  lr* = {lr_star:.4f}  "
      f"(grid {[r['lr'] for r in lr_row]} -> {np.round(y, 4)})")

save("round3", dict(rows=rows, lr_star=lr_star, c=C,
                    eval_entropy=float(evals.entropy.mean())))
print("\n" + ledger.report())
