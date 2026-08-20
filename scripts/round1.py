"""Screening round 1 -- learning-rate landscape + first two IsoFLOP rungs.

Goals
  (a) where is lr* and how does it move with (n, steps)?
  (b) does init scale matter (zero init vs random logits)?
  (c) first two points of the n*(C) curve.
Cap: 6e11 flops.
"""

import numpy as np

from _common import (EVAL_TOKENS, STREAM_TOKENS, fit, get_evalset, get_stream,
                     isoflop_rung, ledger, preflight, save, train_sweep)

CAP = 6e11
LRS_WIDE = np.geomspace(6e-3, 1.0, 7)
LRS_MID = np.geomspace(6e-3, 0.6, 5)
RUNGS = [
    dict(c=4.0e9, ns=[32, 64, 128, 256, 512], lrs=LRS_WIDE),
    dict(c=1.2e10, ns=[64, 128, 256, 512, 1024], lrs=LRS_MID),
]
INIT_C, INIT_N = 4.0e9, 128
INIT_SCALES = [0.0, 0.3, 1.0, 3.0]
INIT_LRS = np.geomspace(2e-2, 0.5, 3)

jobs = [(n, fit.steps_for(r["c"], n), len(r["lrs"])) for r in RUNGS for n in r["ns"]]
jobs.append((INIT_N, fit.steps_for(INIT_C, INIT_N), len(INIT_SCALES) * len(INIT_LRS)))
preflight(jobs, CAP)

stream = get_stream(STREAM_TOKENS)
evals = get_evalset(EVAL_TOKENS)
print(f"eval set: {len(evals)} tokens, mean entropy (irreducible) "
      f"= {evals.entropy.mean():.4f} nats\n")

rows = []
for r in RUNGS:
    rows += isoflop_rung(r["c"], r["ns"], r["lrs"], tag="R1-isoflop",
                         stream=stream, evals=evals)

# init-scale probe: full (init_scale x lr) grid in one vmapped sweep
g_lr, g_is = np.meshgrid(INIT_LRS, INIT_SCALES, indexing="ij")
res = train_sweep(n=INIT_N, steps=fit.steps_for(INIT_C, INIT_N), lrs=g_lr.ravel(),
                  init_scales=g_is.ravel(), stream=stream, eval_set=evals,
                  eval_tokens=EVAL_TOKENS, tag="R1-init")
init_rows = [dict(n=INIT_N, c=INIT_C, lr=float(a), init_scale=float(b), loss=float(l))
             for a, b, l in zip(res.lrs, res.init_scales, res.loss)]
print("\ninit-scale probe (n=%d):" % INIT_N)
for s in INIT_SCALES:
    sel = [r for r in init_rows if r["init_scale"] == s]
    print(f"  init_scale={s:<4} " + "  ".join(f"lr={r['lr']:.3g}:{r['loss']:.4f}" for r in sel))

save("round1", dict(rows=rows, init=init_rows, eval_entropy=float(evals.entropy.mean())))
print("\n" + ledger.report())
