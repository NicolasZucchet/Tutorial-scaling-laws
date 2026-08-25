"""The (N, D) scan: measured scaling law vs the back-of-the-envelope prediction.

At alpha = 1.2 the envelope calculation on the slides predicts, with *no fitted
parameters at all*,

    L - L_inf  ~  l * [ sum_{i > cap(N)} p(i)  +  sum_{i > D^(1/alpha)} p(i) ]
               ~  A N^-0.2  +  B D^-(1/6)

where l = log d - E[H] = 3.775 nats is the cost of a context the model does not know,
cap(N) ~ 41 h is the measured capacity, and the exponents come from the Zipf tail:
(alpha - 1) = 0.2 for the model axis and (1 - 1/alpha) = 1/6 for the data axis.

This script measures the left-hand side on a grid, fits a Chinchilla-style
L = L_inf + A N^-a + B D^-b to the *cheap corner only*, and then extrapolates to
points up to 400x more expensive to see whether the fit and the theory hold.

Two things make this a sharper test than the usual scaling-law study: L_inf is known
exactly (it is the mean conditional entropy, 2.4637 nats) so it is fixed rather than
fitted, and training is single-pass over a nested stream, which is exactly the
assumption behind the D^(1/alpha) coverage argument.

    PYTHONPATH=src uv run python scripts/scaling_grid.py --stage A     # ~8 min
    PYTHONPATH=src uv run python scripts/scaling_grid.py --stage B     # ~35 min
    PYTHONPATH=src uv run python scripts/scaling_grid.py --report      # fit + tables
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

OUT = pathlib.Path("results/grid.json")
LEDGER = pathlib.Path("results/grid_ledger.jsonl")

# This scan is *not* the student exercise, so it gets its own ledger and no budget.
# The 1e13-flop ledger in results/ledger.jsonl is closed and must not be touched.
ledger.configure(path=LEDGER, budget=float("inf"))

from assocmem import grid as G  # noqa: E402
from assocmem.problem import TOTAL_TOKENS, get_evalset, get_stream  # noqa: E402

# D = 64 * steps, in powers of four, topping out at the full stream.
STEPS = (100, 400, 1600, 6400, 25_600, 102_400, 409_600)
HS = (32, 64, 128, 256, 512, 1024, 2048, 8192)

# Stage A: the cheap corner the law is *fitted* on.  C <= 5.0e11 per run.
A_HS = (32, 64, 128, 256, 512)
A_STEPS = (100, 400, 1600, 6400, 25_600)
A_SEEDS = (0, 1, 2)

# Stage B, held out from the fit, one seed, lr centred on the stage-A lr surface.
# Three jobs, which is why the list is not a rectangle:
#   * the N-limited corner (small h, large D), where the model axis is asymptotic and
#     the local slope should approach alpha - 1 = 0.2.  Cheap.
#   * the D-limited corner (large h), where the data axis should approach 1 - 1/alpha.
#   * completing IsoFLOP anti-diagonals at high C.  h and steps move in powers of two
#     and four, so cells with equal h*steps cost exactly the same; a profile needs
#     three h a factor of four apart, e.g. h=[128, 512, 2048] at steps=[25600, 6400,
#     1600].  These give a measured N*(C) that never passes through a joint fit.
B_CELLS = ((2048, 1600), (32, 102_400), (64, 102_400), (32, 409_600),
           (128, 102_400), (64, 409_600), (256, 102_400), (2048, 6400),
           (128, 409_600), (512, 102_400), (256, 409_600), (1024, 25_600),
           (2048, 25_600), (8192, 6400), (512, 409_600), (1024, 102_400),
           (2048, 102_400), (1024, 409_600), (2048, 409_600))
B_SEEDS = (0,)


def key(h: int, steps: int, seed: int) -> str:
    return f"h{h}/s{steps}/z{seed}"


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return dict(meta={}, cells={})


def save(store: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.rename(OUT)


def run(cells, seeds, store, se, stream, ev, lr_fn=None, refine=3) -> dict:
    todo = [(h, s, z) for (h, s) in cells for z in seeds
            if key(h, s, z) not in store["cells"]]
    if not todo:
        print("  nothing to do (all cells present)")
        return store
    plan = sum(6 * 512 * h * 64 * s * 5 for h, s, _ in todo)
    print(f"  {len(todo)} cells, ~{plan:.2e} flops planned "
          f"(~{plan / 4.5e11 / 60:.0f} min at 450 GFLOP/s)\n")
    print(f"  {'cell':>18s} {'lr*':>8s} {'excess':>8s} {'edge':>5s} "
          f"{'flops':>9s} {'s':>6s}")
    for h, s, z in todo:
        lrs = None
        if lr_fn is not None:
            g = lr_fn(h, s)
            lrs = [g * 2.0**e for e in (-1, 0, 1)]
        c = G.run_cell(h, s, se, stream, ev, seed=z, lrs=lrs, refine=refine)
        store["cells"][key(h, s, z)] = c
        save(store)
        print(f"  {key(h, s, z):>18s} {c['lr_star']:8.4f} {c['excess_star']:8.4f} "
              f"{c['lr_edge'] or '-':>5s} {c['flops']:9.2e} {c['seconds']:6.1f}")
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("A", "B"), help="which stage to run")
    ap.add_argument("--cells", nargs="*", type=int, metavar="H STEPS",
                    help="run one explicit cell: --cells 1024 409600")
    ap.add_argument("--report", action="store_true", help="fit and print, no training")
    args = ap.parse_args()

    store = load()
    se = G.build_strat_eval()
    store["meta"] = dict(l_inf=se.l_inf, l=float(np.log(512) - se.l_inf),
                         bin_lo=[float(x) for x in se.bin_lo], m_eval=len(se),
                         steps=list(STEPS), hs=list(HS))
    if args.report:
        from assocmem.grid_fit import FORMS, isoflop_profiles, report
        out = report(store)
        if out:  # a machine-readable summary, for the alpha sweep to collect
            summary = dict(
                alpha=1.2, l_inf=se.l_inf, l_assumed=store["meta"]["l"],
                l_measured=out["l_measured"], n_cells=len(store["cells"]),
                laws={f: vars(out["laws"][f]) for f in FORMS},
                isoflop=isoflop_profiles(store))
            pathlib.Path("results/grid_fit.json").write_text(
                json.dumps(summary, indent=1))
            print("\nwrote results/grid_fit.json")
        return

    cells = seeds = None
    if args.cells:
        cells, seeds = [tuple(args.cells[:2])], (0,)
        need = 64 * cells[0][1]
    elif args.stage == "A":
        cells, seeds, need = [(h, s) for h in A_HS for s in A_STEPS], A_SEEDS, 64 * max(A_STEPS)
    elif args.stage == "B":
        cells, seeds, need = list(B_CELLS), B_SEEDS, 64 * max(s for _, s in B_CELLS)
    else:
        ap.error("give --stage A|B, --cells H STEPS, or --report")

    print(f"L_inf = {se.l_inf:.4f} nats   l = {np.log(512) - se.l_inf:.4f} nats   "
          f"eval = {len(se)} weighted tokens")
    print(f"stream: need {need:,} of {TOTAL_TOKENS:,} tokens")
    t0 = time.time()
    stream = get_stream(need)
    ev = get_evalset(4096)
    print(f"loaded in {time.time() - t0:.0f} s\n")

    lr_fn = None
    if args.stage == "B" or args.cells:
        from assocmem.grid_fit import lr_surface
        lr_fn = lr_surface(store)
        if lr_fn is None:
            print("  no stage-A cells yet: falling back to the lr prior")

    t0 = time.time()
    store = run(cells, seeds, store, se, stream, ev, lr_fn=lr_fn)
    save(store)
    print(f"\ndone in {time.time() - t0:.0f} s.  {len(store['cells'])} cells in {OUT}")
    print(ledger.report().splitlines()[0])


if __name__ == "__main__":
    main()
