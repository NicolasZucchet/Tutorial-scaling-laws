"""Screening round 2 -- two more IsoFLOP rungs (30x span in total), narrowed lr.

Round 1 said: init scale is irrelevant (use 0), lr* ~ 0.06-0.08 at every scale.
So here we keep 3 lrs at C=4e10 (to check whether lr* drifts) and 2 at C=1.2e11,
and centre the n grids on the round-1 prediction n* = 0.00786 C^0.415.
Cap: 1.45e12 flops.
"""

import numpy as np

from _common import (EVAL_TOKENS, STREAM_TOKENS, fit, get_evalset, get_stream,
                     isoflop_rung, ledger, preflight, save, train_sweep)

CAP = 1.45e12
RUNGS = [
    dict(c=4.0e10, ns=[96, 192, 384, 768], lrs=[0.045, 0.075, 0.125]),
    dict(c=1.2e11, ns=[160, 320, 640], lrs=[0.05, 0.09]),
    dict(c=1.2e11, ns=[1280], lrs=[0.07]),  # guard against n* being under-predicted
]

jobs = [(n, fit.steps_for(r["c"], n), len(r["lrs"])) for r in RUNGS for n in r["ns"]]
preflight(jobs, CAP)

stream = get_stream(STREAM_TOKENS)
evals = get_evalset(EVAL_TOKENS)

rows = []
for r in RUNGS:
    rows += isoflop_rung(r["c"], r["ns"], r["lrs"], tag="R2-isoflop",
                         stream=stream, evals=evals)

save("round2", dict(rows=rows, eval_entropy=float(evals.entropy.mean())))
print("\n" + ledger.report())
