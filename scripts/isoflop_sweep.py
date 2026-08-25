"""A sweep laid out for the IsoFLOP figure: six model sizes per compute budget.

The (N, D) grid was laid out to measure exponents on the N and D axes, so its IsoFLOP
anti-diagonals carry only three or four points each.  A parabola through three points is
not a fit, it is an interpolation, and the points are wherever the grid happened to put
them -- at C = 10^13 the cheapest one sits a factor of 80 below N*, which is far enough
off-optimum to bend the parabola and move its vertex.

So: the same six budgets, but six widths each, log-spaced over [N*/3, 3N*] around the N*
the first pass measured.  Two constraints make this exact rather than approximate:

* C = 6 * (512 h) * (64 * steps), so a budget fixes the product ``h * steps = K``.  Every
  width must therefore divide K, and K = 2^(11+2j) * 25 here, so the admissible widths
  are 2^a, 5*2^a and 25*2^a -- a ladder in steps of 1.25, dense enough that snapping a
  target ratio to it costs at most 12 %.  No interpolation, no rounding of steps.
* the learning rate is taken from the surface fitted to the first pass and bracketed
  three ways, then refined until the optimum is interior, exactly as the grid's stage B
  does.  An lr that is systematically off *as a function of h* would tilt the parabola
  rather than merely raise it.

    PYTHONPATH=src uv run python scripts/isoflop_sweep.py --plan   # table + cost, free
    PYTHONPATH=src uv run python scripts/isoflop_sweep.py          # ~15 min
    PYTHONPATH=src uv run python scripts/isoflop_sweep.py --budgets 0 1   # a slice

Checkpointed to results/isoflop_grid.json after every cell, so it is resumable, and
billed to its own ledger (results/isoflop_ledger.jsonl, no budget).  The cells use the
same schema as results/grid.json, so `assocmem.grid_fit.isoflop_detail` reads either.
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
GRID = ROOT / "results/grid.json"
OUT = ROOT / "results/isoflop_grid.json"
LEDGER = ROOT / "results/isoflop_ledger.jsonl"

# Its own ledger, no budget: this is not the student exercise, and the closed 1e13
# ledger in results/ledger.jsonl must not be touched.
ledger.configure(path=LEDGER, budget=float("inf"))

from assocmem import grid as G  # noqa: E402
from assocmem.data import D_OUT  # noqa: E402
from assocmem.grid_fit import isoflop_detail, lr_surface  # noqa: E402
from assocmem.problem import get_evalset, get_stream  # noqa: E402

BATCH = 64
K0 = 51_200  # h*steps on the cheapest anti-diagonal of the grid; C = 6*512*64*K
N_BUDGETS = 6  # ... and the budgets go up by 4, as the grid's do
PER_BUDGET = 6
SPAN = 3.0  # widths cover [N*/SPAN, N*·SPAN]
SEED = 0

# h must divide K = 2^(11+2j) * 25.
ADMISSIBLE = sorted({m * 2**a for m in (1, 5, 25) for a in range(16)})


def key(h: int, steps: int, seed: int) -> str:
    return f"h{h}/s{steps}/z{seed}"


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else dict(meta={}, cells={})


def save(store: dict) -> None:
    tmp = OUT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.rename(OUT)


def plan(n_star_of_c) -> list[dict]:
    """The cell list: for each budget, six admissible widths around N*(C)."""
    out = []
    for j in range(N_BUDGETS):
        k = K0 * 4**j
        c = 6.0 * D_OUT * BATCH * k
        h_star = n_star_of_c(c) / D_OUT
        hs: list[int] = []
        for r in SPAN ** np.linspace(-1.0, 1.0, PER_BUDGET):
            cand = [h for h in ADMISSIBLE if k % h == 0 and h not in hs]
            hs.append(min(cand, key=lambda h: abs(np.log(h / (h_star * r)))))
        hs = sorted(hs)
        steps = [k // h for h in hs]
        assert all(h * s == k for h, s in zip(hs, steps)), (j, hs, steps)
        out.append(dict(c=c, k=k, h_star=h_star, hs=hs, steps=steps))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the cells and the cost")
    ap.add_argument("--budgets", nargs="*", type=int, help="indices 0..5 to run")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-lr", type=int, default=3, help="lr bracket width")
    args = ap.parse_args()

    grid = load(GRID)
    det = isoflop_detail(grid)
    law = det["laws"]["n_of_c"]

    def n_star_of_c(c):
        return law["pref"] * c ** law["exp"]

    lr_fn = lr_surface(grid)
    if lr_fn is None:
        raise SystemExit("no lr surface: results/grid.json has no bracketed cells")
    rows = plan(n_star_of_c)
    want = range(N_BUDGETS) if not args.budgets else args.budgets

    store = load(OUT)
    todo, flops = [], 0.0
    for j in want:
        r = rows[j]
        for h, s in zip(r["hs"], r["steps"]):
            if key(h, s, args.seed) in store["cells"]:
                continue
            todo.append((h, s, r["c"]))
            flops += 6.0 * D_OUT * h * BATCH * s * args.n_lr

    print(f"N*(C) = {law['pref']:.3f} C^{law['exp']:.4f}, from the first pass"
          f"   lr* = exp({lr_fn.coef[0]:.2f}) h^{lr_fn.coef[1]:.3f} D^{lr_fn.coef[2]:.3f} "
          f"({100 * lr_fn.rms:.0f} % rms)")
    for j in want:
        r = rows[j]
        ratios = "  ".join(f"{h * D_OUT / n_star_of_c(r['c']):.2f}" for h in r["hs"])
        print(f"  C = {r['c']:.3e}   N* = {n_star_of_c(r['c']):9.0f}   "
              f"h = {r['hs']}\n{'':>22s}steps = {r['steps']}   N/N* = {ratios}")
    tok = max((64 * s for _, s, _ in todo), default=0)
    print(f"\n{len(todo)} cells to run, {args.n_lr} lrs each: {flops:.3e} flops "
          f"(~{flops / 4.0e11 / 60:.0f} min at 400 GFLOP/s), "
          f"{tok:,} stream tokens needed")
    if args.plan or not todo:
        print("nothing to do" if not todo and not args.plan else "(plan only)")
        return

    store["meta"] = dict(k0=K0, per_budget=PER_BUDGET, span=SPAN, seed=args.seed,
                         n_star_law=law, lr_surface=dict(coef=list(lr_fn.coef),
                                                         rms=lr_fn.rms, n=lr_fn.n),
                         budgets=[r["c"] for r in rows])
    se = G.build_strat_eval()
    t0 = time.time()
    stream = get_stream(tok)
    ev = get_evalset(4096)
    print(f"stream loaded in {time.time() - t0:.0f} s\n")
    print(f"  {'cell':>16s} {'N/N*':>6s} {'lr*':>8s} {'excess':>8s} {'edge':>5s} "
          f"{'flops':>9s} {'s':>6s}")

    t0 = time.time()
    for h, s, c in todo:
        g = lr_fn(h, s)
        lrs = [g * 2.0**e for e in np.linspace(-1, 1, args.n_lr)]
        cell = G.run_cell(h, s, se, stream, ev, seed=args.seed, lrs=lrs, refine=3,
                          tag="isoflop")
        cell["c_target"] = c
        store["cells"][key(h, s, args.seed)] = cell
        save(store)
        print(f"  {key(h, s, args.seed):>16s} {h * D_OUT / n_star_of_c(c):6.2f} "
              f"{cell['lr_star']:8.5f} {cell['excess_star']:8.4f} "
              f"{cell['lr_edge'] or '-':>5s} {cell['flops']:9.2e} {cell['seconds']:6.1f}",
              flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min.  "
          f"{len(store['cells'])} cells in {OUT.relative_to(ROOT)}")
    print(ledger.report().splitlines()[0])


if __name__ == "__main__":
    main()
