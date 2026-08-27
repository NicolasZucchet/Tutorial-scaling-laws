"""The Chinchilla runs and the Chinchilla paper's own published numbers.

Part I's three compute-optimal slides (Pareto front, IsoFLOP, parametric fit) all read
the *same* point cloud, so it is fetched once, written to results/, and never touched
again by the figure script.

Where the runs come from -- this matters, and it is stated on every figure the deck
draws from them.  DeepMind never released the per-run logs behind Figure 2 or the
IsoFLOP sweeps behind Figure 3.  What is public is a reconstruction of **Figure 4**:
Epoch AI's replication attempt [@besiroglu2024chinchilla] read the scatter straight out
of the paper's own SVG, recovering (model size, training FLOPs, loss) for 245 of the
paper's 400-odd runs -- the loss from the point's fill colour, so it is quantised at
the colour map's resolution.

    https://github.com/epoch-research/analyzing-chinchilla
    data/svg_extracted_data.csv

Two cleanups happen here and nowhere else:

  * the extracted model sizes carry sub-pixel jitter (142 distinct values for the
    paper's 50 architectures), so each is snapped to the nearest size in Table A9 --
    every point moves by less than 0.6 % in N;
  * the dataset size is not in the figure at all and is recovered as D = C / 6N, the
    deck's own compute identity.

So: a reconstruction, a subset, and one derived column.  It is not the original data,
it is the best public approximation of it, and it reproduces the paper's headline
exponents to within a couple of percent -- which is the whole point of showing it.

    uv run python scripts/chinchilla_data.py           # from the cached JSON
    uv run python scripts/chinchilla_data.py --fetch   # re-download the CSV

Writes results/chinchilla_runs.json (the point cloud plus its provenance) and
results/chinchilla_paper.json (the paper's Table 2 exponents and its Approach-3
parametric constants, transcribed from the arXiv HTML).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/chinchilla_runs.json"
PAPER = ROOT / "results/chinchilla_paper.json"

CSV_URL = ("https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/"
           "main/data/svg_extracted_data.csv")
REPO_URL = "https://github.com/epoch-research/analyzing-chinchilla"
PAPER_URL = "https://arxiv.org/html/2203.15556v1"

# Table A9 of Hoffmann et al. 2022: the 50 architectures, in millions of parameters.
# Only the parameter counts are needed -- the extracted model sizes are snapped to
# them, so the jitter from reading pixels off a figure does not split one architecture
# into five.
TABLE_A9_M = (44, 57, 74, 90, 106, 117, 140, 163, 175, 196, 217, 251, 278, 306, 425,
              489, 509, 552, 587, 632, 664, 724, 816, 893, 1018, 1143, 1266, 1424,
              1429, 1593, 1609, 1731, 1794, 2007, 2283, 2298, 2639, 2980, 3530, 3802,
              4084, 4516, 6796, 9293, 11452, 12295, 12569, 13735, 14940, 16183)

# The nine training-FLOP budgets of Approach 2 ("a fixed set of 9 different training
# FLOP counts, ranging from 6e18 to 3e21", Section 3.2).  The exact values are not
# printed in the paper; these are the nine the extracted runs cluster on.
ISOFLOP_BUDGETS = (6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21)

# ---------------------------------------------------------------- the paper's numbers
#
# Transcribed from the arXiv HTML of 2203.15556v1, not recalled.  Table 2 reports the
# exponents of N_opt ~ C^a and D_opt ~ C^b with 10th/90th percentiles from a bootstrap
# over 80 % resamples; the Kaplan row is the comparison the table itself draws.
TABLE_2 = [
    {"approach": "1. Pareto front",
     "paper": "Minimum over training curves",
     "a": 0.50, "a_ci": [0.488, 0.502], "b": 0.50, "b_ci": [0.501, 0.512]},
    {"approach": "2. IsoFLOP",
     "paper": "IsoFLOP profiles",
     "a": 0.49, "a_ci": [0.462, 0.534], "b": 0.51, "b_ci": [0.483, 0.529]},
    {"approach": "3. Parametric fit",
     "paper": "Parametric modelling of the loss",
     "a": 0.46, "a_ci": [0.454, 0.455], "b": 0.54, "b_ci": [0.542, 0.543]},
    {"approach": "Kaplan et al. (2020)",
     "paper": "Kaplan et al. (2020)",
     "a": 0.73, "a_ci": None, "b": 0.27, "b_ci": None},
]

# Section D.2, equation (10): L(N, D) = E + A/N^alpha + B/D^beta, fitted with a Huber
# loss on the log loss.  In the deck's Part II notation (slide 44) this is
# L - L_inf = A N^-a + B D^-b, so E is L_inf, alpha is a and beta is b.
PARAMETRIC = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}


# ---------------------------------------------------------------- fetch and clean


def fetch_csv() -> str:
    with urllib.request.urlopen(CSV_URL, timeout=60) as fh:
        return fh.read().decode("utf-8")


def clean(text: str) -> dict:
    """Snap model sizes to Table A9, derive D, and drop nothing else."""
    rows = list(csv.DictReader(io.StringIO(text)))
    n = np.array([float(r["Model Size"]) for r in rows])
    c = np.array([float(r["Training FLOP"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])

    sizes = np.array(TABLE_A9_M, dtype=float) * 1e6
    j = np.argmin(np.abs(np.log10(n)[:, None] - np.log10(sizes)[None, :]), axis=1)
    snapped = sizes[j]
    drift = float(np.abs(np.log10(n / snapped)).max())
    if drift > 0.01:
        raise SystemExit(f"a model size moved by {drift:.3f} decades when snapped to "
                         "Table A9: the extraction or the table has changed")

    order = np.lexsort((c, snapped))
    return {
        "source": {
            "runs": "reconstruction of Figure 4 of Hoffmann et al. 2022 (Chinchilla)",
            "by": "Epoch AI, Besiroglu et al. 2024, 'Chinchilla Scaling: A replication "
                  "attempt' -- read out of the paper's own SVG",
            "repo": REPO_URL,
            "csv": CSV_URL,
            "paper_numbers": PAPER_URL,
            "caveats": [
                "245 of the paper's 400-odd runs; Figures 2 and 3 were never released",
                "loss is read from the marker colour, so it is quantised",
                "model sizes snapped to Table A9 (max drift "
                f"{100 * (10**drift - 1):.2f} % in N)",
                "dataset size is derived, D = C / 6N, not measured",
            ],
        },
        "n_runs": len(rows),
        "table_a9_m": list(TABLE_A9_M),
        "isoflop_budgets": list(ISOFLOP_BUDGETS),
        "runs": [{"n": float(snapped[i]), "c": float(c[i]),
                  "d": float(c[i] / (6 * snapped[i])), "loss": float(loss[i])}
                 for i in order],
    }


def paper_numbers() -> dict:
    a, b = PARAMETRIC["alpha"], PARAMETRIC["beta"]
    return {
        "source": PAPER_URL,
        "note": "Table 2 and Section D.2 of Hoffmann et al. 2022, transcribed from the "
                "arXiv HTML.",
        "table_2": TABLE_2,
        "parametric": PARAMETRIC,
        # What the Approach-3 constants imply for the compute-optimal frontier.
        # Minimising E + A N^-alpha + B D^-beta under C = 6ND gives
        #     N* = (alpha A / beta B)^(1/(alpha+beta)) (C/6)^(beta/(alpha+beta)),
        # and D* = C/6N*, so both exponents and both prefactors are fixed by the five
        # constants -- no extra fitting.  The deck draws the ridge from these.
        "parametric_implied": {
            "n_exp": b / (a + b),
            "d_exp": a / (a + b),
            "l_exp": -a * b / (a + b),
            "n_pref": (a * PARAMETRIC["A"] / (b * PARAMETRIC["B"])) ** (1 / (a + b))
                      / 6 ** (b / (a + b)),
            "d_pref": (b * PARAMETRIC["B"] / (a * PARAMETRIC["A"])) ** (1 / (a + b))
                      / 6 ** (a / (a + b)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="re-download the extracted CSV instead of trusting the "
                         "cached results/chinchilla_runs.json")
    args = ap.parse_args()

    if args.fetch or not RUNS.exists():
        print(f"downloading {CSV_URL}")
        store = clean(fetch_csv())
        RUNS.write_text(json.dumps(store, indent=1))
        print(f"wrote {RUNS.relative_to(ROOT)}  ({store['n_runs']} runs)")
    else:
        store = json.loads(RUNS.read_text())
        print(f"{RUNS.relative_to(ROOT)} already has {store['n_runs']} runs "
              "(--fetch to refresh)")

    p = paper_numbers()
    imp = p["parametric_implied"]
    PAPER.write_text(json.dumps(p, indent=1))
    print(f"wrote {PAPER.relative_to(ROOT)}")
    print(f"  parametric fit implies N* ~ C^{imp['n_exp']:.4f}, "
          f"D* ~ C^{imp['d_exp']:.4f}, L-L_inf ~ C^{imp['l_exp']:.4f}")
    # A cross-check on the prefactors: the two must satisfy 6 N* D* = C.
    c = 1e21
    n_s, d_s = imp["n_pref"] * c ** imp["n_exp"], imp["d_pref"] * c ** imp["d_exp"]
    assert abs(6 * n_s * d_s / c - 1) < 1e-9, (n_s, d_s)
    print(f"  at C = 1e21 it puts N* = {n_s / 1e9:.2f}B, D* = {d_s / 1e9:.0f}B "
          f"({d_s / n_s:.0f} tokens per parameter)")

    runs = store["runs"]
    n = np.array([r["n"] for r in runs])
    print(f"  N {n.min():.3g}..{n.max():.3g} over "
          f"{len(set(n.tolist()))} distinct model sizes")


if __name__ == "__main__":
    main()
