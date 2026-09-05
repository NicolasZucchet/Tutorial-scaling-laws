"""Model scaling under online sampling from a fixed, finite pool of possible contexts.

Contexts are sampled online from a Zipf distribution truncated and renormalised on
1..K, K = `--support` (the slide runs at K = 2,000; K = 10,000 is the earlier pass and
is still on disk).  Every occurrence receives a fresh next-token draw from its fixed
conditional distribution.  The cached stream is only a reproducible realisation of this online
process.  Evaluation computes the loss exactly by weighting every context by its
renormalised frequency; no held-out Monte Carlo sample is needed.

The stream is stored as a chain of equal 6.55M-token chunks, chunk `i` drawn with its
own seed, so a longer stream is always the shorter one plus an appended chunk (the same
discipline as `stream_master`/`stream_ext` -- `sample_tokens` sizes its rejection chunk
from the requested length, so re-drawing at a new length would silently change the
realisation and with it every number already recorded).  For K = 10 000, chunk 0 is
byte-for-byte the 6.55M file the first pass of that sweep used.

Results, the stream cache and the ledger are all keyed by K, so pools do not overwrite
each other and a re-run of a pool already done is a no-op.

    PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/finite_context_sweep.py
    PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/finite_context_sweep.py --support 10000
    PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/finite_context_sweep.py --steps 102400
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
CACHE = ROOT / "results/cache"

K = 2_000  # the pool the slide plots; --support overrides it
ALPHA = 1.2
HS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
STEPS = (409_600,)
SEEDS = (0,)
CHUNK_TOKENS = 6_553_600  # = 64 * 102_400, the length of the first pass


def _suffix(k: int) -> str:
    """K = 10 000 keeps the unsuffixed names it was recorded under; others are keyed."""
    return "" if k == 10_000 else f"_k{k}"


def out_path(k: int) -> pathlib.Path:
    return ROOT / f"results/finite_support_sweep{_suffix(k)}.json"


def ledger_path(k: int) -> pathlib.Path:
    return ROOT / f"results/finite_support_ledger{_suffix(k)}.jsonl"


# Set for the default pool at import time so `import finite_context_sweep as FCS` gets a
# working module; `main` re-points it when --support asks for another pool.
OUT = out_path(K)
LEDGER = ledger_path(K)
ledger.configure(path=LEDGER, budget=float("inf"))


def set_support(k: int) -> None:
    """Re-point the module (and its ledger) at pool `k`.  Call before `run`/`load`."""
    global K, OUT, LEDGER
    K, OUT, LEDGER = k, out_path(k), ledger_path(k)
    ledger.configure(path=LEDGER, budget=float("inf"))


def finite_eval(k: int = None) -> tuple[G.StratEval, EvalSet]:
    # The same evaluator as slide 22.  Setting head=max_context makes all K contexts
    # exact; max_context is the only distributional change.
    k = K if k is None else k
    se = G.build_strat_eval(head=k, gamma=ALPHA, max_context=k)
    ev = EvalSet(se.hi[:4096], se.lo[:4096], se.probs[:4096], se.entropy[:4096])
    return se, ev


def eval_shape(n_eval: int) -> tuple[int, int]:
    """(eval_tokens, eval_chunk) for the in-training monitor over `n_eval` contexts.

    `_eval` averages over whole chunks and drops the remainder, so a pool smaller than
    the default 2048 chunk would monitor a NaN.  Pools of 4096 or more keep the original
    4096/2048 exactly.  The monitor never feeds a plotted number -- those come from the
    exact frequency-weighted `excess_loss`, which pads -- but a NaN in the ledger is
    still a NaN.
    """
    tokens = min(4096, n_eval)
    return tokens, (2048 if tokens >= 2048 and tokens % 2048 == 0 else tokens)


def _chunk(i: int, k: int = None) -> pathlib.Path:
    """One 6.55M-token chunk of the online draw, appended never regrown."""
    k = K if k is None else k
    # "v2": the labels are drawn from p(y|x), so a chunk written under the old
    # per-token-logit conditional must never be read back by the new one -- the
    # model would learn one conditional and be scored against another.
    f = CACHE / f"finite_shared_v2_k{k}_d{CHUNK_TOKENS}_s{i}.npz"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tokens = D.sample_tokens(CHUNK_TOKENS, seed=19_000 + i, gamma=ALPHA,
                                 max_context=k)
        # A fresh y is drawn for every occurrence, exactly as in online sampling.
        labels = D.sample_labels(tokens, seed=20_000 + i)
        hi, lo = D.split_u32(tokens)
        tmp = f.with_name(f.name + ".partial.npz")
        np.savez(tmp, hi=hi, lo=lo, y=labels)
        tmp.rename(f)
    return f


def finite_stream(tokens: int, k: int = None) -> Stream:
    n = -(-tokens // CHUNK_TOKENS)
    zs = [np.load(_chunk(i, k)) for i in range(n)]
    cat = lambda k: np.concatenate([z[k] for z in zs]) if n > 1 else zs[0][k]
    return Stream(cat("hi"), cat("lo"), cat("y"))


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


def run(steps_list: tuple[int, ...]) -> dict:
    se, ev = finite_eval()
    eval_tokens, eval_chunk = eval_shape(len(se))
    stream = finite_stream(64 * max(steps_list))
    store = load()
    store["meta"] = {
        "sampling": "online-truncated-zipf",
        "evaluation": "exact-renormalized-frequencies",
        "shared_pipeline": True,
        "support": K,
        "alpha": ALPHA,
        "l_inf": se.l_inf,
        "hs": list(HS),
        "steps": sorted({*store["meta"].get("steps", []), *steps_list}),
        "series": list(steps_list),
    }
    store["meta"]["tokens"] = [64 * s for s in store["meta"]["steps"]]
    todo = [(h, s, z) for h in HS for s in steps_list for z in SEEDS
            if key(h, s, z) not in store["cells"]]
    print(f"online Zipf sampling over exactly {K:,} possible contexts")
    print(f"exact frequency-weighted eval over all {K:,} contexts; "
          f"{len(todo)} cells remaining")
    for h, steps, seed in todo:
        t0 = time.time()
        cell = G.run_cell(
            h, steps, se, stream, ev, seed=seed,
            eval_tokens=eval_tokens, eval_chunk=eval_chunk, seg_steps=102_400,
            tag=f"finite-support-k{K}",
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
    steps_list = sorted({c["steps"] for c in store["cells"].values()})
    print(f"\n{'h':>5s} {'parameters':>11s} "
          + " ".join(f"D={64*s:>10,d}" for s in steps_list))
    prev = {s: None for s in steps_list}
    for h in HS:
        vals = []
        for steps in steps_list:
            cell = store["cells"].get(key(h, steps, 0))
            if cell is None:
                vals.append("           -     ")
                continue
            e = cell["excess_star"]
            sl = "     -" if prev[steps] is None else f"{np.log2(e / prev[steps]):+6.3f}"
            prev[steps] = e
            vals.append(f"{e:10.6f} ({sl})")
        print(f"{h:5d} {D.D_OUT*h:11,d} " + " ".join(vals))
    print("\n(the number in brackets is the local log-log slope against the rung below)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--support", type=int, default=K,
                    help="size of the context pool (default %(default)s)")
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS),
                    help="one model-scaling series per step count (D = 64 * steps)")
    args = ap.parse_args()
    set_support(args.support)
    store = load() if args.report else run(tuple(args.steps))
    report(store)


if __name__ == "__main__":
    main()
