# Scaling laws for an associative memory — a tutorial

Train the best `p̂(·|x) = softmax(W eₓ)` you can, with `W ∈ R^(512×n)`, Zipf(1.2) inputs and
random fixed conditional outputs — under **10¹³ flops** and **3 screening rounds**, with **one
shot** at the final run. The whole exercise costs about a minute of laptop compute.

## For students

```bash
uv sync --extra notebook
uv run assocmem-prepare        # once: builds ~250 MB of cached data (~1 min)
uv run jupyter lab notebooks/tutorial.ipynb
```

The notebook is the tutorial. You write about ten lines:

```python
from assocmem import Lab, Sweep

lab = Lab("me", budget=1e13, rounds=3)

s1 = Sweep(c=[4e9, 1.2e10], n=[64, 128, 256, 512], lr=[0.0125, 0.025, 0.05, 0.1, 0.2])
s1.estimate(lab)                        # free: cost, share of budget, ETA, what to cut
r1 = lab.run_round("R1", s1)            # spends one round, prints a table, plots itself
# ... two more rounds ...
laws = lab.fit()                        # n*(C), lr*(C), L*(C) with r², plotted
lab.hero(laws)                          # sizes to the remainder, commits a prediction, runs
```

Everything is billed automatically and **you cannot overspend** — the library refuses and tells
you what to cut. Re-running an identical sweep is free and does not burn a round, `smoke=True`
runs a sweep at 1 % of the steps to check it works, and `lab.reset(confirm=True)` starts over.
State lives in `runs/<name>/` and survives a kernel restart.

### The API in full

| | |
|---|---|
| `Sweep(c=[...], n=[...], lr=[...])` | cartesian product; `c` is flops per run, so steps are derived from `n` — that puts every run at equal compute, which is what an IsoFLOP profile needs. Use `steps=[...]` instead of `c` for a direct sweep. `Sweep + Sweep` concatenates. |
| `sweep.estimate(lab)` | free. Flops, % of budget, wall-clock ETA, per-rung breakdown |
| `lab.run_round(name, sweep)` | trains everything; costs one round. `smoke=True` → free of rounds |
| `results.best() / .table() / .isoflop() / .plot() / .df` | slice, print, fit, draw |
| `lab.fit()` | the three power laws + `.recipe(C)`, `.predict(C)`, `.summary()`, `.plot()` |
| `lab.hero(laws)` | one shot, sized to the leftover budget. Prints the prediction *before* training |
| `lab.status() / .remaining / .rounds_left / .reset()` | where you stand |

`lab.fit()` warns you when a rung's optimum falls outside the widths you tried, or when a
learning-rate grid never bracketed its minimum — the two ways a scaling-law fit quietly lies.

## For instructors

* `REPORT.md` + `figures/final.png` — the reference solution: **3.2609 nats**, produced by
  `scripts/expert_run.py`.
* `notebooks/tutorial_reference_run.ipynb` — the notebook executed with its deliberately
  mediocre default sweeps, so you know what students see out of the box (**3.2765**).
* `scripts/round{1,2,3}.py` + `final_fit.py` + `hero.py` — the first, over-tuned attempt
  (**3.2993**), kept because the comparison is the lesson.
* Every student gets the **same** problem instance, so all three are directly comparable.

Three attempts on the same problem, differing mainly in how much went to screening:

| attempt | tuning | hero compute | hero loss |
|---|---|---|---|
| careful (`scripts/round*.py`) | 47 % | 5.01·10¹² | 3.2993 |
| notebook defaults | 26 % | 6.97·10¹² | 3.2765 |
| expert (`scripts/expert_run.py`) | **8 %** | **8.83·10¹²** | **3.2609** |

All three hero runs collapse onto one power law `L − L∞ = 9.48·C^-0.083` with residuals of
0.0001 nats: the hero loss is set by the compute it gets, almost regardless of the recipe. So
the punchline to steer toward is that **tuning precision is worth less than the compute it
costs** — 10¹² of screening costs ~0.008 nats, while being 20 % wrong in `n` costs 0.0014. The
whole achievable spread is ~0.05 nats, and the zero-tuning ceiling is 3.2552.

A second, sharper finding students can reach: don't *fit* the loss exponent, **derive** it from
the width exponent. With capacity `∝ n^c` and excess `∝ K^-(γ-1)`, `b = 1/(1+cγ)` gives
`α = (γ−1)(1−b)/γ`. That rule predicted all three hero runs to within +0.003…+0.010 nats,
versus +0.032…+0.053 for their own fitted power laws.

### Running the reference solution

```bash
export PYTHONPATH=src
uv run python scripts/round1.py     # then round2, round3
uv run python scripts/final_fit.py  # laws -> hero recipe
uv run python scripts/hero.py
uv run python scripts/figures.py
uv run python -m assocmem.ledger    # audit the flop budget
```

These write to `results/ledger.jsonl`, which is already closed at 99.9 % of the reference
budget; delete it to re-run from scratch. Student labs are accounted separately, in
`runs/<name>/ledger.jsonl`.

```bash
uv run python tests/test_lab.py     # ~2 s: problem definition, budget guards, fit recovery
```

## Layout

| | |
|---|---|
| `src/assocmem/lab.py` | `Lab`, `Sweep`, `Results`, `Laws` — the student-facing API |
| `src/assocmem/plots.py` | auto-composed round / law / hero figures |
| `src/assocmem/data.py` | exact Zipf sampler, entropy-matched conditionals, sphere embeddings — all hashed from the token id, so the 1.1×10¹¹-token vocabulary is never materialised |
| `src/assocmem/train.py` | hand-written-gradient Adam trainer, `lax.scan` over steps, vmapped over configs; flop accounting |
| `src/assocmem/fit.py` | IsoFLOP parabolas, power laws, joint `L(n, D)` fit |
| `src/assocmem/problem.py`, `prepare.py` | cached stream + eval sets |
| `src/assocmem/ledger.py` | the flop ledger (`python -m assocmem.ledger` to audit) |

## Notes on the implementation

* **Flops** are counted as `6ND` for training and `2NM` for every evaluation pass. `6ND` is
  conservative here: the true cost is `4ND`, since the embeddings are inputs and need no
  gradient.
* **Speed.** A round's configs are grouped by `(n, steps)` and trained in one vmapped call, so
  the tiny 512×n matmuls batch into one big one — ~400 GFLOP/s accounted on an M4 Pro, which
  makes the entire 10¹³ budget about 25 s of arithmetic. Configs in a group share one data
  stream (common random numbers), so *comparisons* between them are much less noisy than the
  absolute losses.
* **Evaluation** uses the exact expected cross-entropy `−Σ_y p(y|x) log p̂(y|x)` rather than a
  sampled `y`: same population loss, far less variance. `lab.hero` cross-checks against the
  plain sampled-`y` CE and an independent eval set.
