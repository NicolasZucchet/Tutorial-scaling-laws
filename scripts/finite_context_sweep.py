"""Model scaling under online sampling from exactly 10,000 possible contexts.

Contexts are sampled online from a Zipf distribution truncated and renormalised on
1..10,000.  Every occurrence receives a fresh next-token draw from its fixed conditional
distribution.  The cached stream is only a reproducible realisation of this online
process.  Evaluation computes the loss exactly by weighting every context by its
renormalised frequency; no held-out Monte Carlo sample is needed.

    PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/finite_context_sweep.py
    PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/finite_context_sweep.py --report
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import data as D  # noqa: E402
from assocmem import grid as G  # noqa: E402
from assocmem import ledger  # noqa: E402
from assocmem.train import EvalSet, Stream  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/finite_support_sweep.json"
CACHE = ROOT / "results/cache/finite_shared_k10000_d6553600_s0.npz"
LEDGER = ROOT / "results/finite_support_ledger.jsonl"

K = 10_000
ALPHA = 1.2
HS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
STEPS = (102_400,)
SEEDS = (0,)
MAX_TOKENS = 64 * max(STEPS)

ledger.configure(path=LEDGER, budget=float("inf"))


def finite_eval() -> tuple[G.StratEval, EvalSet]:
    # The same evaluator as slide 22.  Setting head=max_context makes all 10k contexts
    # exact; max_context is the only distributional change.
    se = G.build_strat_eval(head=K, gamma=ALPHA, max_context=K)
    ev = EvalSet(se.hi[:4096], se.lo[:4096], se.probs[:4096], se.entropy[:4096])
    return se, ev


def finite_stream() -> Stream:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tokens = D.sample_tokens(MAX_TOKENS, seed=19_000, gamma=ALPHA,
                                 max_context=K)
        # A fresh y is drawn for every occurrence, exactly as in online sampling.
        labels = D.sample_labels(tokens, seed=20_000)
        hi, lo = D.split_u32(tokens)
        tmp = CACHE.with_name(CACHE.name + ".partial.npz")
        np.savez(tmp, hi=hi, lo=lo, y=labels)
        tmp.rename(CACHE)
    z = np.load(CACHE)
    return Stream(z["hi"], z["lo"], z["y"])


def load() -> dict:
    if OUT.exists():
        store = json.loads(OUT.read_text())
        if store.get("meta", {}).get("shared_pipeline") is True:
            return store
    return {"meta": {}, "cells": {}}


def save(store: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.rename(OUT)


def key(h: int, steps: int, seed: int) -> str:
    return f"h{h}/s{steps}/z{seed}"


def run() -> dict:
    se, ev = finite_eval()
    stream = finite_stream()
    store = load()
    store["meta"] = {
        "sampling": "online-truncated-zipf",
        "evaluation": "exact-renormalized-frequencies",
        "shared_pipeline": True,
        "support": K,
        "alpha": ALPHA,
        "l_inf": se.l_inf,
        "hs": list(HS),
        "steps": list(STEPS),
        "tokens": [64 * s for s in STEPS],
    }
    todo = [(h, s, z) for h in HS for s in STEPS for z in SEEDS
            if key(h, s, z) not in store["cells"]]
    print(f"online Zipf sampling over exactly {K:,} possible contexts")
    print(f"exact frequency-weighted eval over all {K:,} contexts; "
          f"{len(todo)} cells remaining")
    for h, steps, seed in todo:
        t0 = time.time()
        cell = G.run_cell(
            h, steps, se, stream, ev, seed=seed,
            eval_tokens=4096, seg_steps=102_400,
            tag="finite-support-k10000",
        )
        store["cells"][key(h, steps, seed)] = cell
        save(store)
        print(
            f"h={h:4d} N={D.D_OUT*h:7,d} D={64*steps:9,d} "
            f"loss={cell['excess_star']:.6f} lr={cell['lr_star']:.5f} "
            f"edge={cell['lr_edge'] or '-':>4s} ({time.time()-t0:.1f}s)"
        )
    return store


def report(store: dict) -> None:
    if not store.get("cells"):
        print("no results yet")
        return
    print(f"\n{'h':>5s} {'parameters':>11s} " + " ".join(f"D={64*s:>8,d}" for s in STEPS))
    for h in HS:
        vals = []
        for steps in STEPS:
            cell = store["cells"].get(key(h, steps, 0))
            vals.append("       -" if cell is None else f"{cell['excess_star']:10.6f}")
        print(f"{h:5d} {D.D_OUT*h:11,d} " + " ".join(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    store = load() if args.report else run()
    report(store)


if __name__ == "__main__":
    main()
