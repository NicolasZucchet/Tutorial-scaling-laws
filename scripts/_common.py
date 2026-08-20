"""Shared helpers for the screening / hero scripts."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import sys
from pathlib import Path

import numpy as np

from assocmem import fit, ledger
from assocmem.problem import get_evalset, get_stream
from assocmem.train import plan_cost, train_sweep

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_TOKENS = 4096  # screening eval set size
STREAM_TOKENS = 600_000


def save(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"-> results/{name}.json")


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def preflight(jobs, cap: float) -> float:
    """jobs: list of (n, steps, n_cfg).  Prints the bill and aborts if over `cap`."""
    tot = sum(plan_cost(n, s, k, EVAL_TOKENS, 1) for n, s, k in jobs)
    st = ledger.total()
    print(f"planned: {tot:.4g} flops over {len(jobs)} sweeps "
          f"({sum(k for _, _, k in jobs)} configs)")
    print(f"already spent: {st['total']:.4g} | budget {ledger.BUDGET:.3g} "
          f"| after this round: {st['total'] + tot:.4g} "
          f"({100 * (st['total'] + tot) / ledger.BUDGET:.1f}%)")
    if tot > cap:
        sys.exit(f"ABORT: {tot:.4g} exceeds this round's cap {cap:.4g}")
    if st["total"] + tot > ledger.BUDGET:
        sys.exit("ABORT: over total budget")
    return tot


def isoflop_rung(c: float, ns, lrs, *, tag: str, stream, evals, init_scales=None,
                 instance_seed: int = 0):
    """Train every (n, lr) at fixed compute `c`; returns a list of row dicts."""
    rows = []
    for n in ns:
        steps = fit.steps_for(c, n)
        r = train_sweep(n=n, steps=steps, lrs=lrs, init_scales=init_scales,
                        stream=stream, eval_set=evals, eval_tokens=EVAL_TOKENS,
                        instance_seed=instance_seed, tag=tag)
        for lr, isc, l in zip(r.lrs, r.init_scales, r.loss):
            rows.append(dict(c=c, n=n, steps=steps, tokens=steps * 64,
                             lr=float(lr), init_scale=float(isc), loss=float(l)))
        b = r.best()
        print(f"  C={c:.3g} n={n:5d} steps={steps:6d}  best lr={b['lr']:.4g} "
              f"loss={b['loss']:.4f}   all={np.round(r.loss, 3)}")
    return rows


__all__ = ["RESULTS", "EVAL_TOKENS", "STREAM_TOKENS", "save", "load", "preflight",
           "isoflop_rung", "get_stream", "get_evalset", "train_sweep", "fit", "ledger"]
