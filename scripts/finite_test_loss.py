"""Test loss of the finite-support models, measured on the *full* context pool.

Slide "What if data was finite?" plots two model-scaling series at the same budget
(D = 2.62e7 draws / 409,600 steps):

    "first 2k contexts"     trained *and* evaluated on the Zipf truncated to K = 2,000
    "infinite context pool" trained and evaluated on the whole Zipf

The truncated series is the one that bends away from a power law, and the reason is
that it runs out of things to learn.  This script measures the other half of that
story: take exactly the same truncated-pool runs and score them against the *full*
distribution, the one the untruncated series is trained on.  That is a genuine
train/test split -- the model has never seen a context beyond rank K, and the full
distribution puts 19 % of its mass there at K = 2,000 -- so the curve plateaus at that
missing mass while the training curve keeps falling.

`--support` picks the pool, and must name one `finite_context_sweep.py` has already
run; outputs are keyed by it the same way.

Each cell is retrained at the learning rate that won the grid in
`scripts/finite_context_sweep.py` (one rate, not the whole grid -- the grid's only job
was to pick it) on the byte-identical stream, and the run is checked against the loss
recorded there before its test loss is kept.  A cell whose training loss does not
reproduce is reported and dropped rather than plotted, because a silent mismatch would
mean the two curves no longer come from the same models.

    PYTHONPATH=src uv run python scripts/finite_test_loss.py
    PYTHONPATH=src uv run python scripts/finite_test_loss.py --support 10000
    PYTHONPATH=src uv run python scripts/finite_test_loss.py --hs 8 16    # a subset
    PYTHONPATH=src uv run python scripts/finite_test_loss.py --report
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from assocmem import data as D  # noqa: E402
from assocmem import grid as G  # noqa: E402
from assocmem import ledger  # noqa: E402
from assocmem.train import EvalSet, train_sweep  # noqa: E402

import finite_context_sweep as FCS  # noqa: E402  (same eval set, same stream)

STEPS = 409_600
SEED = 0
TOL = 2e-4  # reproduction tolerance on the training loss, in nats

TRAIN = OUT = LEDGER = None  # set by set_support, below


def set_support(k: int) -> None:
    """Point this script and `finite_context_sweep` at pool `k`.

    Both are keyed by K so that pools do not overwrite each other; K = 10 000 keeps the
    unsuffixed names its results were first recorded under (`FCS._suffix`).
    """
    global TRAIN, OUT, LEDGER
    FCS.set_support(k)           # also re-points *its* ledger, so ours goes second
    sfx = FCS._suffix(k)
    TRAIN = FCS.out_path(k)
    OUT = ROOT / f"results/finite_support_test{sfx}.json"
    LEDGER = ROOT / f"results/finite_test_ledger{sfx}.jsonl"
    ledger.configure(path=LEDGER, budget=float("inf"))


set_support(FCS.K)


def train_cells() -> dict:
    store = json.loads(TRAIN.read_text())
    return {c["h"]: c for c in store["cells"].values()
            if c["steps"] == STEPS and c["seed"] == SEED}


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"meta": {}, "cells": {}}


def save(store: dict) -> None:
    tmp = OUT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.rename(OUT)


def run(hs: list[int]) -> dict:
    cells = train_cells()
    unknown = [h for h in hs if h not in cells]
    if unknown:
        raise SystemExit(f"no {STEPS}-step run recorded for h={unknown}; "
                         f"run scripts/finite_context_sweep.py first")

    se_train, ev = FCS.finite_eval()           # truncated to K, as trained
    se_test = G.build_strat_eval()             # the whole Zipf: the test set
    eval_tokens, eval_chunk = FCS.eval_shape(len(se_train))
    stream = FCS.finite_stream(64 * STEPS)

    store = load()
    store["meta"] = {
        "steps": STEPS,
        "tokens": 64 * STEPS,
        "seed": SEED,
        "support": FCS.K,
        "alpha": FCS.ALPHA,
        "train_eval": f"truncated-{FCS.K}, exact renormalised frequencies",
        "test_eval": "full Zipf, stratified (assocmem.grid.build_strat_eval defaults)",
        "l_inf_train": se_train.l_inf,
        "l_inf_test": se_test.l_inf,
        "tail_mass_beyond_support": G.tail_mass(FCS.K, FCS.ALPHA),
        "source": str(TRAIN.relative_to(ROOT)),
    }

    todo = [h for h in hs if str(h) not in store["cells"]]
    print(f"scoring the truncated-pool runs against the full pool "
          f"({len(todo)} of {len(hs)} cells to do)")
    print(f"mass beyond context {FCS.K:,}: "
          f"{100 * store['meta']['tail_mass_beyond_support']:.2f} %")
    for h in todo:
        cell = cells[h]
        lr = cell["lr_best"]
        t0 = time.time()
        r = train_sweep(h, STEPS, [lr], stream, ev, instance_seed=SEED,
                        eval_points=max(1, STEPS // 102_400),
                        eval_tokens=eval_tokens, eval_chunk=eval_chunk,
                        tag=f"finite-test-h{h}", return_params=True)
        ex_train, _, _ = G.excess_loss(r.params, se_train, n=h, instance_seed=SEED)
        ex_test, per_bin, _ = G.excess_loss(r.params, se_test, n=h, instance_seed=SEED)
        drift = float(ex_train[0]) - cell["excess_best"]
        ok = abs(drift) <= TOL
        store["cells"][str(h)] = dict(
            h=h, n_params=D.D_OUT * h, lr=lr,
            excess_train=float(ex_train[0]), excess_train_recorded=cell["excess_best"],
            drift=drift, reproduced=ok,
            excess_test=float(ex_test[0]),
            per_bin_test=[float(v) for v in per_bin[0]],
            seconds=round(time.time() - t0, 1),
        )
        save(store)
        flag = "" if ok else "  <-- DOES NOT REPRODUCE"
        print(f"h={h:5d} N={D.D_OUT*h:8,d} train={float(ex_train[0]):.6f} "
              f"(recorded {cell['excess_best']:.6f}, drift {drift:+.2e}) "
              f"test={float(ex_test[0]):.6f}{flag} ({time.time()-t0:.1f}s)")
    return store


def report(store: dict) -> None:
    if not store.get("cells"):
        print("no results yet")
        return
    print(f"\n{'h':>5s} {'parameters':>11s} {'train':>10s} {'test':>10s} "
          f"{'test/train':>11s}  reproduced")
    for h in sorted(map(int, store["cells"])):
        c = store["cells"][str(h)]
        print(f"{h:5d} {c['n_params']:11,d} {c['excess_train']:10.6f} "
              f"{c['excess_test']:10.6f} {c['excess_test']/c['excess_train']:11.2f}"
              f"  {'yes' if c['reproduced'] else 'NO'}")
    bad = [h for h in store["cells"] if not store["cells"][h]["reproduced"]]
    if bad:
        print(f"\n{len(bad)} cell(s) did not reproduce their recorded training loss: "
              f"h={sorted(map(int, bad))}.  Do not plot these.")
    print("\nchart series (paste into figures/finite-chart.md):")
    for h in sorted(map(int, store["cells"])):
        c = store["cells"][str(h)]
        if c["reproduced"]:
            print(f"        - {{x: {c['n_params']}, y: {c['excess_test']:.6f}}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", type=int, nargs="+", default=list(FCS.HS))
    ap.add_argument("--support", type=int, default=FCS.K,
                    help="context pool to score (default %(default)s)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    set_support(args.support)
    store = load() if args.report else run(args.hs)
    report(store)


if __name__ == "__main__":
    main()
