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

## The capacity experiment (slide "Capacity, in theory and in practice")

How many contexts can `softmax(W e)` actually store at **100 % accuracy**, and how much
of that does the Hebbian one-step solution get? `d = 256` next tokens, embedding
dimension `h` swept over 32…2048, 3 seeds, and for each `(model, h, seed)` a bisection
on the number of contexts `n`, narrowed until fewer than 8 contexts separate the largest
success from the smallest failure.

```bash
export PYTHONPATH=src
uv run python scripts/capacity_sweep.py                  # everything (~16 h on CPU)
uv run python scripts/capacity_sweep.py --h 128 256      # one slice
uv run python scripts/capacity_sweep.py --chart-only     # redraw from the JSON
uv run python scripts/capacity_sweep.py --chart-only --write-slide   # + the deck's figure
uv run python tests/test_capacity.py                     # ~5 s
```

The run is checkpointed after every cell, so it is resumable and re-running skips what is
already done. Outputs:

| | |
|---|---|
| `results/capacity.json` | every probe: `n`, accuracy, steps, whether the step backstop bound it |
| `results/capacity_chart.md` | two colloquium ` ```chart ` blocks + a table, for reading outside the deck |
| `figures/capacity-chart.md` | with `--write-slide`: the deck's figure — the slide version of the first chart, its legend and `assets/capacity-chart.js`. The slide holds only `<!-- figure: capacity-chart -->`, so this script owns the plot outright |

Method notes, because a "100 % accuracy" number is only as good as its protocol:

* **Frequencies are absent on purpose.** A 100 %-accuracy criterion does not care how
  often a context appears, so contexts are unweighted and the Zipf law of
  `assocmem.data` plays no role here.
* **Nested instances.** The `n`-context problem is a *prefix* of the `n'`-context one,
  so every probe in a bisection is asking about the same contexts.
* **Trained to saturation, not to a budget.** Adam on the full-batch cross-entropy stops
  when the accuracy reaches 100 % or when it stops improving — never on a step count.
  Near the boundary the accuracy creeps for thousands of steps, so this matters: at
  `h = 32` a 1k-step budget reports 1089 contexts, 2k reports 1102, and 4k, 8k and 32k
  report 1107, 1107, 1111. The 8k backstop is therefore converged to ~0.4 %, and probes
  that hit it are flagged `capped` in the JSON.
* **The learning rate is part of the measurement.** `lr = 3` was picked because it
  dominates: `lr = 1` sometimes fails to close the last 0.1 % that `lr = 3` closes, and
  `lr = 10`/`30` plateau *lower* on the failing side. A hinge (Crammer-Singer) objective
  and a closed-form ridge solution were both tried and are much worse, so plain
  cross-entropy is the best witness available.
* **Cost is set by the top of the range.** The measured run took 15.6 h, of which
  `h = 2048` alone was 12 h: the near-boundary probes are the slow ones, and precision 8
  out of ~85 000 contexts puts most of a 16-probe bisection there.
* **Monotonicity is approximate.** Adding a context can repair another one by
  strengthening its class, so a few successes can sit above the first failure and the
  bisection reports *a* crossing, not the first. An exhaustive scan finds this negligible
  for `h >= 64` and real at `h = 32`, which is where the seed spread is largest.

## The (N, D) scan (slide "Results")

Does the back-of-the-envelope calculation on the slides actually predict the toy model's
scaling law? At `alpha = 1.2` it predicts, with **no fitted parameters at all**,

    L - L_inf  ~  l * (mass of contexts the model cannot know)
               ~  A N^-(alpha-1)  and  B D^-(1-1/alpha)   =  A N^-0.2  and  B D^-0.1667

so the scan measures the left-hand side over a grid in model size and data, fits a law to
the **cheap corner only**, and extrapolates up to 400x further out.

```bash
export PYTHONPATH=src
uv run python scripts/build_extension.py   # once: extends the stream to 26.2M tokens
uv run python scripts/scaling_grid.py --stage A    # the fit region, 75 runs, ~6 min
uv run python scripts/scaling_grid.py --stage B    # held-out targets, 19 runs, ~57 min
uv run python scripts/scaling_grid.py --report     # the fits and the tables
uv run python scripts/grid_figures.py              # figures/grid_law.png, grid_map.png
uv run python tests/test_grid.py tests/test_stream.py
```

Grid: `h` in 32...8192 (so `N = 512h` from 1.6e4 to 4.2e6), `D = 64 * steps` in powers of
four from 6.4e3 to 2.6e7, single pass, learning rate optimised at every cell -- the grid
is refined until the optimum is *interior*, since an edge optimum biases the loss upward
by an amount that varies with `h` and `D`, which corrupts an exponent rather than merely
shifting it. Stage A is `h <= 512, D <= 1.6e6` with three seeds (94 cells, 5.7 min);
stage B holds out the far corners and the small-`h`/large-`D` and large-`h`/small-`D`
edges, with learning rates from a surface fitted on stage A (`lr* ~ h^-0.09 D^-0.41`,
20 % rms) and a three-point bracket to confirm. Seed-to-seed spread is ~0.003 nats.

Three things make this a sharper test than a normal scaling-law study:

* **`L_inf` is known**, not fitted: it is the mean conditional entropy, 2.46 nats. The
  scan reports the *excess* `L - L_inf` computed per token as `CE_i - H_i`, a paired
  difference, so the usual trade-off between `L_inf` and the exponents never arises.
* **Training is single-pass over a nested stream**, which is exactly the assumption
  behind the `D^(1/alpha)` coverage argument. (The flip side: a context seen once and a
  context never seen cannot be told apart, so a measured data exponent near `1-1/alpha`
  is consistent with the coverage story without isolating it.)
* **The evaluation is stratified, not sampled.** With `alpha = 1.2` the top 100 contexts
  carry 65 % of the mass, so a set drawn from `p(x)` estimates the tail -- the part the
  law is about -- from a handful of draws, with a standard error as large as the effect.
  `assocmem.grid` instead takes every context below 4096 with its exact weight and
  log-stratifies the rest, 87 strata over 34.6k tokens. That also yields the per-stratum
  loss curve, which is the direct picture of the step function the envelope assumes.

What it finds, at `alpha = 1.2`:

| | measured | predicted | |
|---|---|---|---|
| model exponent, where `N` binds | 0.2116 | `alpha-1` = 0.2000 | +5.8 % |
| data exponent, where `D` binds | 0.1637 | `1-1/alpha` = 0.1667 | -1.8 % |
| `N*(C)`, from IsoFLOP profiles | `C^0.4491` | `C^0.4545` | -1.2 % |
| `L*(C) - L_inf`, from IsoFLOP profiles | `C^-0.0913` | `C^-0.0909` | +0.4 % |

The first two are finite differences between neighbouring cells, the last two are six
IsoFLOP profiles spanning `C = 1e10 ... 1e13` with `r2` of 0.9994 and 0.9999, measured by
the dedicated sweep below. Nothing in the right-hand column was fitted: it is `alpha` and
arithmetic.

with two lessons that only the grid can show:

* **The two constraints compose as a `min`, not a sum.** A context is learned when it is
  *both* within capacity *and* has been seen, so the envelope predicts a kink,
  `max(A N^-a, B D^-b)`, not the additive Chinchilla form. Fitting the additive form to
  the same data gives a 3.2 % relative error -- a fit that looks fine -- and exponents
  that are 0.79x and 2.40x the predicted ones, hence `N* ~ C^0.72` instead of `C^0.45`.
  A power mean, `[(A N^-a)^q + (B D^-b)^q]^(1/q)` with `q` fitted (it comes out 3.4),
  fits 8x better, recovers the predicted exponents to 1 % and 12 %, and extrapolates to
  19 held-out cells -- out to `C = 1.6e14`, 63x past the top of the fit region -- with a
  **1.3 %** rms error, against 6.9 % for the additive form and 6.8 % for the pure `max`.
* **The exponents are right and the prefactor is not.** The per-stratum curve shows the
  assumed step is a sigmoid spread over ~2.5 decades of context index, and the model
  behaves as if it knows 15-50x fewer contexts than `min(capacity(N), D^(1/alpha))`,
  so the envelope under-predicts the loss by 40-50 % everywhere on the grid. A
  constant factor on the cutoff moves the prefactor and leaves the exponent alone, which
  is exactly what is observed. The cost of an unknown context is also not the assumed
  `l = log d - L_inf = 3.78` nats but 4.42: an unknown context is worse than uninformed,
  because the weights that store the frequent contexts actively mispredict it.

Caveats worth stating with the numbers: the capacity constant used by the envelope
(41 contexts per unit `h`) was measured at `d = 256` and this problem has `d = 512`, and
the local exponents are corner values -- away from the corners each axis is masked by the
other bottleneck, so its local slope is smaller (see panels (c) and (d)).

The scan is billed to its own ledger, `results/grid_ledger.jsonl`, with no budget: it is
not the student exercise, and the closed 1e13 ledger in `results/ledger.jsonl` is never
touched. Total cost 1.47e15 flops, about 150x the student budget, and ~63 min of wall clock.

### The alpha sweep

Both exponents are supposed to be pure functions of the tail, so the same fit-region grid
is repeated at several `alpha`, and each exponent is read off the corner where its own
constraint binds -- no functional form assumed.

```bash
export PYTHONPATH=src
uv run python scripts/alpha_sweep.py                # 1.1, 1.3, 1.5, 1.8; ~7 min each
uv run python scripts/alpha_sweep.py --extend       # small-h / large-D cells; ~8 min each
uv run python scripts/alpha_sweep.py --report       # -> results/alpha_sweep.json
uv run python scripts/alpha_figures.py              # -> figures/alpha_sweep.png
```

| alpha | data axis, measured | `1-1/alpha` | model axis, measured | `alpha-1` |
|---|---|---|---|---|
| 1.1 | 0.0873 | 0.0909 | 0.1124 | 0.1000 |
| 1.2 | 0.1637 | 0.1667 | 0.2116 | 0.2000 |
| 1.3 | 0.2284 | 0.2308 | 0.2944 | 0.3000 |
| 1.5 | 0.3483 | 0.3333 | 0.5077 | 0.5000 |
| 1.8 | 0.4907 | 0.4444 | 0.7263 | 0.8000 |

The data axis holds across the whole range. The model axis needs a caveat that turned out
to be the most useful thing the sweep produced: **it can only be read where the model is
what binds**, i.e. where `D^(1/alpha)` runs well past the capacity, and the margin needed
grows steeply with `alpha`. At `D = 1.6e6` the margin at `h = 64` is 129x for
`alpha = 1.2` but only 1.2x for `alpha = 1.8`, and the measured exponent tracks the margin
rather than the prediction: ratios of 1.01, 0.98, 0.90, 0.67 as the margin falls.

`--extend` tests exactly that, by adding `h` in {32, 64} at 16x and 64x more data:

| alpha | `a` at `D=1.6e6` | margin | `a` at `D=2.6e7` | margin | `alpha-1` |
|---|---|---|---|---|---|
| 1.5 | 0.4509 | 5.3x | **0.5077** | 33.6x | 0.5000 |
| 1.8 | 0.5397 | 1.1x | **0.7263** | 5.0x | 0.8000 |

So the shortfall was the data constraint masking the model axis, not the prediction
failing -- and `alpha = 1.8` is still only at a 5x margin, so its 0.73 is a lower bound.
The compute-optimal exponents track `b/(a+b)` and `ab/(a+b)` across the sweep as well; see
`figures/alpha_sweep.png`.

Each `alpha` gets its own stream, stratified eval set and grid file, under `results/alpha/`
(`alpha = 1.2` reuses the main scan). The same append-only discipline applies: a stream is
generated once at a canonical length and extended by a separate file, never regrown.

### The IsoFLOP construction (slide "Are the toy model predictions good enough?")

The last two rows of that table are IsoFLOP profiles, and the deck shows the construction
rather than just the answer: at each compute budget, sweep the model size, fit a parabola
in `log N`, mark the minimum, then fit the minima.

```bash
export PYTHONPATH=src
uv run python scripts/isoflop_sweep.py --plan          # the cell list and its cost, free
uv run python scripts/isoflop_sweep.py                 # 36 cells, 2.5e14 flops, ~8 min
uv run python scripts/isoflop_slide.py                 # fits -> results/isoflop_fits.json
uv run python scripts/isoflop_slide.py --write-slide   # -> figures/isoflop-figure.md + slides.md
uv run python scripts/grid_slide.py --write-slide      # -> figures/results-alpha.md + slides.md
```

The `(N, D)` grid's anti-diagonals are already *exact* IsoFLOP lines — `h` in powers of
two and `steps` in powers of four, so equal `h*steps` costs equal flops — but they carry
only three or four points each, wherever the grid happened to put them: at `C = 1e13` its
cheapest point sits a factor of 80 below `N*`. A parabola through three points is an
interpolation, not a fit, and a badly placed third point bends it.

`scripts/isoflop_sweep.py` therefore re-measures the same six budgets with **six widths
each**, log-spaced over `[N*/3, 3N*]` around the `N*` the first pass found. Two things
keep it exact rather than approximate:

* a budget fixes the product `h * steps = K`, and `K = 2^(11+2j) * 25` here, so the
  admissible widths are `2^a`, `5*2^a` and `25*2^a` — a ladder in steps of 1.25, dense
  enough that snapping a target ratio to it costs at most 12 %. Every width divides `K`,
  so no step count is rounded and no loss is interpolated.
* every cell gets a three-point learning-rate bracket from the surface fitted on the
  first pass (`lr* ~ h^-0.028 D^-0.461`, 27 % rms), refined until the optimum is
  interior. An lr that is wrong *as a function of h* tilts a parabola rather than merely
  raising it. All 36 came out interior.

`assocmem.grid_fit.isoflop_detail` then returns the points, the parabola, the minimum and
the three power laws the minima define:

| | measured | predicted | | anti-diagonals |
|---|---|---|---|---|
| `L* - L∞` vs `C` | `C^-0.0913` | `-ab/(a+b)` = `C^-0.0909` | r² 0.9999 | `C^-0.0916` |
| `N*` vs `C` | `C^0.4491` | `b/(a+b)` = `C^0.4545` | r² 0.9994 | `C^0.4626` |
| `L* - L∞` vs `N*` — the envelope | `N*^-0.2032` | `-(alpha-1)` = `N*^-0.2000` | r² 0.9997 | `N*^-0.1979` |

The last column is what the grid's own anti-diagonals gave, for comparison: `L*` barely
moves (a parabola is flat at its base, so a poorly sampled one still finds the right
depth) while `N*` shifts by up to 6 % at the top two budgets, which is what pulls the
exponent from `+1.8 %` off the prediction to `-1.2 %`.

The third row is the one worth knowing: the two compute exponents share a denominator, so
their ratio is exactly `-(alpha - 1)`. **The envelope of the IsoFLOP minima carries the
model-size exponent**, with no compute axis anywhere in it — measured `-0.203` against a
predicted `-0.200`.

Profiles whose parabola minimum falls outside the widths measured are kept in the JSON
flagged `edge` and excluded from all three fits: an edge minimum is a bound, not a
measurement. The dedicated sweep has none; the grid's anti-diagonals had one
(`C = 5.2e12`).

The figure is a **generated SVG**, not a ` ```chart ` block, because its two reveals (the
envelope, then `N*(C)`) are `<g class="fragment">` groups — the same mechanism the deck's
hand-drawn figures use. Two things to know before editing it: colloquium numbers
fragments by substituting *every* literal `data-colloquium-fragment="1"`, so that value is
a marker and the reveal order is document order (anything else is left permanently
visible); and a `1180x470` viewBox renders 613 px tall at full slide width, which
overflows the slide and silently pushes the figure off the bottom — hence `1180x400`.

Outputs: `results/isoflop_grid.json` (36 cells, same schema as `grid.json`, resumable),
`results/isoflop_fits.json` (every profile and every fit, `source` records which store it
came from), `results/isoflop_chart.md`, and its own ledger `results/isoflop_ledger.jsonl`.

### A note on the training stream

`data.sample_tokens` sizes its rejection-sampling chunk from the requested length, so its
RNG desynchronises with `m`: **the first 4M tokens of a 26M draw are not the first 4M
tokens of a 4M draw.** Raising a single `MASTER_TOKENS` and regenerating would silently
change the problem instance under every number in `REPORT.md` and `results/ledger.jsonl`,
with no error anywhere. So `stream_master_s0.npz` is frozen and is the canonical prefix;
longer streams are that file concatenated with a separately stored, independently seeded
extension (`stream_ext_s0.npz`). `tests/test_stream.py` asserts the base fingerprint and
the nesting, and `build_extension` refuses to clobber.

## The emergence experiment (slide "Digression: emergent behavior under the hood")

The scan above reports one number, the excess loss averaged over the whole Zipf tail, and it
is a clean power law. Ask the *same* runs a **threshold** question instead — for a band of
context ranks, does the model predict the **most likely next token**, counted uniformly
inside the band? — and the same nine models look like five sigmoids switching on one after
another.

```bash
export PYTHONPATH=src
uv run python scripts/emergence.py --run          # 9 models, h = 32...8192, full stream (~67 min)
uv run python scripts/emergence.py --report       # the switch-on tables
uv run python scripts/emergence.py --figures      # figures/emergence.png, six panels
uv run python scripts/emergence.py --write-slide  # figures/emergence-chart.md, the deck's figure
uv run python tests/test_emergence.py             # ~15 s
```

Bands `1-1k`, `2-3k`, `5-6k`, `10-11k`, `20-30k` (every context of the four narrow ones, 4096
sampled uniformly from the last, 8096 in total), 23 checkpoints per run, `D` = 26.2M tokens,
one seed. Nothing is trained differently from the scan: same hand-written Adam, same frozen
stream, same cosine schedule, same `lr*` surface as stage B — so the excess loss of these runs
reproduces the scan's own cells to **≤ 0.001 nats**. The two pictures are one experiment seen
through two metrics, which is the whole point.

Where each band switches on, against the capacity yardstick (`capacity = 41h` contexts, so the
band's median rank is reached at `N = 512 * median / 41`):

| band | `N`(10 %) | `N`(50 %) | `N`(90 %) | width | `N` predicted | ratio |
|---|---|---|---|---|---|---|
| 1-1k | — | 3.3·10⁴ | 1.2·10⁵ | — | 6.3·10³ | 5.3 |
| 2-3k | 7.4·10⁴ | 1.8·10⁵ | 5.5·10⁵ | 0.87 dec | 3.1·10⁴ | 5.7 |
| 5-6k | 1.6·10⁵ | 4.0·10⁵ | 1.9·10⁶ | 1.09 dec | 6.9·10⁴ | 5.8 |
| 10-11k | 3.2·10⁵ | 9.2·10⁵ | — | — | 1.3·10⁵ | 7.0 |
| 20-30k | 9.1·10⁵ | 3.1·10⁶ | — | — | 3.1·10⁵ | 9.8 |

and over training, at the largest model (`h` = 8192, `N` = 4.2·10⁶, whose capacity is far past
rank 30k so only coverage can bind):

| band | final | `D`(50 %) | `D` for "seen once" | times seen at `D`(50 %) |
|---|---|---|---|---|
| 1-1k | 0.926 | 2.3·10⁴ | 9.7·10³ | 2.4 |
| 2-3k | 0.927 | 2.3·10⁵ | 6.7·10⁴ | 3.4 |
| 5-6k | 0.901 | 6.3·10⁵ | 1.7·10⁵ | 3.7 |
| 10-11k | 0.870 | 1.8·10⁶ | 3.7·10⁵ | 4.9 |
| 20-30k | 0.615 | 1.1·10⁷ | 1.1·10⁶ | 10.3 |

Three things the pair of pictures says that the loss alone does not:

* **The switches are ordered and they are not steps.** Each band needs 5–10x more parameters
  than the capacity yardstick allows, and it takes ~1 decade of `N` to go from 10 % to 90 %.
  That is the same softness the scan sees as a sigmoid over ~2.5 decades of context index —
  measured here directly, on the contexts themselves.
* **"Seen once" is not enough; a handful of times is.** The coverage argument behind the
  `D^(1-1/alpha)` exponent assumes a context is learned when `p(i) D >= 1`. Measured, the
  crossing sits at 2–10 occurrences. A constant factor on the cutoff moves the prefactor and
  leaves the exponent alone, which is exactly what the scan reports.
* **Nothing jumps in the average.** Over the same nine models the excess loss slides smoothly
  from 1.5526 to 0.5488 nats, and over the same single run its curve is featureless — while
  ten thousand individual contexts go from chance to 0.6.

Method notes, because "accuracy" needs a protocol as much as "capacity" does:

* **Top-1 against the mode, not against a sampled `y`.** A context counts as known when
  `argmax_y W e_x` equals `argmax_y p(y|x)`. Chance is `1/512 = 0.002`. The ceiling is **not**
  1: these conditionals are broad (`exp(H)` averages 16 tokens, `p(y*|x)` averages 0.46), so
  near-ties flip and the head bands saturate around 0.95. The same prediction scored against
  `y ~ p(.|x)` is recorded as `sampled` in the JSON, and saturates at `E[p*]` instead.
* **Uniform inside the band, on purpose.** A threshold metric does not care how often a
  context appears; weighting by `p(i)` would let the head answer for the tail, which is
  exactly the averaging the experiment is trying to undo.
* **Mid-schedule checkpoints.** The training-time curves are checkpoints of one cosine-annealed
  run sized for the full stream, not a family of runs each annealed at its own `D` — the usual
  way such a plot is made, and it slightly understates an early checkpoint.
* **The two largest models are the least tuned.** `h` = 4096 and 8192 take `lr*` from the
  stage-A surface extrapolated (27 % rms), and the head-band accuracy peaks at `h` = 1024
  (0.953) and dips to 0.926 at `h` = 8192 while the loss keeps falling monotonically. Either
  the bigger model really does trade a little head precision for tail coverage, or its
  learning rate is slightly off; the switch-on structure is far larger than the effect.

Cost 1.33·10¹⁵ flops and 67 min of wall clock, billed to `results/emergence_ledger.jsonl`
(no budget, like the scan; the closed 10¹³ student ledger is untouched). Outputs:
`results/emergence.json` (every checkpoint of every run, plus the `--report` analysis),
`results/emergence_chart.md`, `figures/emergence-chart.md`, `figures/emergence.png`.

## The finite-context-pool sweep (slide "What if data was finite?")

The scan above draws contexts from the full, effectively infinite Zipf tail. Truncate the
tail at `K = 10 000` contexts, renormalise, and the model-size curve stops being a power
law: once a model can hold most of a *bounded* pool there is nothing left to buy.

```bash
export PYTHONPATH=src
uv run python scripts/finite_context_sweep.py                  # 9 rungs at D = 2.62e7, ~74 min
uv run python scripts/finite_context_sweep.py --steps 102400    # the original 4x-cheaper series
uv run python scripts/finite_context_sweep.py --report          # both series, with local slopes
```

`h` in 8…2048 (so `N = 512h` from 4.1e3 to 1.05e6), `D = 2.62e7` online draws and 409 600
steps at **every** rung, one seed, learning rate optimised per cell with the same
interior-optimum refinement as the scan (all nine interior, in both series). The run is
checkpointed after every cell, so it is resumable and re-running skips what is done. The
stream is a chain of equal 6.55M-token chunks, chunk `i` drawn with its own seed, so a
longer stream is the shorter one plus an appended chunk — the same append-only discipline
as `stream_master`/`stream_ext` and for the same reason (`sample_tokens` sizes its
rejection chunk from the requested length, so re-drawing at a new length silently changes
the realisation). Chunk 0 is byte-for-byte the file the first pass used.

Two things make this *finite support*, not a finite dataset: contexts are sampled online
from the truncated Zipf, and every occurrence gets a fresh next-token draw from its fixed
conditional. There is no held-out set and no train/test gap — evaluation weights all
10 000 contexts by their exact renormalised frequencies. So the curve below is what
running out of *things to learn* looks like, never what overfitting looks like.

| `h` | `N` | excess loss | local slope | at `D` = 6.55e6 | its slope |
|---|---|---|---|---|---|
| 8 | 4 096 | 1.693699 | — | 1.694101 | — |
| 16 | 8 192 | 1.324286 | −0.355 | 1.325302 | −0.354 |
| 32 | 16 384 | 1.032124 | −0.360 | 1.033968 | −0.358 |
| 64 | 32 768 | 0.783258 | −0.398 | 0.786548 | −0.395 |
| 128 | 65 536 | 0.566136 | −0.468 | 0.573029 | −0.457 |
| 256 | 131 072 | 0.380699 | −0.572 | 0.394475 | −0.539 |
| 512 | 262 144 | 0.228371 | −0.737 | 0.253310 | −0.639 |
| 1024 | 524 288 | 0.127639 | −0.839 | 0.163384 | −0.633 |
| 2048 | 1 048 576 | 0.079962 | **−0.675** | 0.120958 | −0.434 |

Over the same range the infinite-pool curve — same protocol, same `D`, from the scan's own
`steps = 409 600` column — is close to a straight `N^-0.20` (slopes −0.212, −0.205, −0.202,
−0.198, −0.197, −0.182). The finite-pool curve is steeper from the very first rung
(`N^-0.36`) and then *accelerates*, to `N^-0.84` — four times the infinite-pool slope.
That acceleration is the point of the slide.

**The last rung still relaxes, to −0.675, and that is a limitation of the measurement
rather than a property of the problem.** The paragraph below is the evidence; the short
version is that the relaxation is residual under-convergence whose size grows with `N`,
so the bottom of this curve is an upper bound that gets looser as it goes right.

### Why 409 600 steps, and why the top rung is still an upper bound

The first version of this sweep gave every rung 102 400 steps (`D = 6.55e6`) and its top
rung relaxed to −0.434. Quadrupling the budget moved that point down 34 % (0.120958 →
0.079962) and its slope to −0.675, but did not remove the relaxation. What it did do is
show exactly what the relaxation is.

**The objective is convex in `W`**, so each rung's exact optimum — infinite data, no
schedule, no step budget — can be computed directly on the true conditionals by full-batch
Adam on `sum_i p(i) CE(p(.|i), softmax(W e_i))`. That is a per-rung yardstick the sweep can
be held against, and it is not close to flattering at the top:

| `h` | measured | converged optimum | gap | measured slope | optimum slope | gap at 102 400 |
|---|---|---|---|---|---|---|
| 8 | 1.693699 | 1.693454 | +0.01 % | — | — | +0.0 % |
| 16 | 1.324286 | 1.323826 | +0.03 % | −0.355 | −0.355 | +0.1 % |
| 32 | 1.032124 | 1.031228 | +0.09 % | −0.360 | −0.360 | +0.3 % |
| 64 | 0.783258 | 0.781221 | +0.26 % | −0.398 | −0.401 | +0.7 % |
| 128 | 0.566136 | 0.562518 | +0.64 % | −0.468 | −0.474 | +1.9 % |
| 256 | 0.380699 | 0.373205 | +2.01 % | −0.572 | −0.592 | +5.7 % |
| 512 | 0.228371 | 0.212071 | +7.69 % | −0.737 | −0.815 | +19.4 % |
| 1024 | 0.127639 | 0.095693 | +33.4 % | −0.839 | −1.148 | +70.7 % |
| 2048 | 0.079962 | 0.032715 | **+144 %** | **−0.675** | **−1.548** | +270 % |

Three things follow, and they are the whole explanation:

* **The bias grows monotonically with `N`.** Every rung gets the same 409 600 steps, so the
  biggest model is the furthest from its own optimum: +0.01 % at `h = 8` rising to +144 %
  at `h = 2048`. A model-scaling curve whose points are progressively looser upper bounds
  is a curve that is *flattened at its top end*, which is exactly the observed relaxation.
* **The converged curve does steepen monotonically**, −0.355 → −1.548, with no relaxation
  anywhere and no sign of an asymptote — it is already falling faster than `N^-1` by
  `h = 1024`. So the *bend is real*; only its measurement at equal compute is not yet
  resolved at the last rung.
* **More compute shrinks the bias but slowly.** The `h = 2048` gap went from +270 % at
  102 400 steps to +144 % at 409 600: 4x compute roughly halves it. Another uniform 4x
  (1 638 400 steps, `D = 1.05e8`, ~6.7e15 flops and ~5 h of wall clock) would leave it
  near +70 % and the last slope still short of −1. Closing it to a few percent needs
  something like 10^2x, which this experiment cannot afford — so the honest statement is
  that **equal-compute training cannot resolve the bottom of this curve, and the convex
  solve is the only thing here that sees its true shape.** Left for the author to decide.

Two supporting measurements, from the diagnosis that led to the re-run:

* **Doubling only the optimisation, on identical data.** Two epochs over the *same* 6.55M
  draws — no new draws at all — buys 4.2 % at `h = 512` and 14.4 % at `h = 2048`; with 2x
  *fresh* draws instead, 6.0 % and 19.1 %. Optimisation alone is about three quarters of
  each gain, and both are ~3.2x larger at the top rung. A rung that had run out of samples
  could not improve by re-reading samples it already had.
* **Sampling noise is real, and it is not what bends this curve.** The best add-β estimator
  built on the actual empirical counts of the draws scores 0.0550 nats at `D = 6.55e6` and
  **0.0225 nats at `D = 2.62e7`**, so no model of any size can go below that at this `D`
  and the bottom of the curve is an upper bound for that reason too. But the top point sits
  at **3.55x** that floor (and 2.44x its own converged optimum), so sampling is not what
  limits it — under-convergence is. At the old budget the top rung's per-stratum excess was
  a nearly uniform **2.05x** the floor in *every* stratum from rank 100 to 10 000, which is
  what an unconverged run looks like; a capacity ceiling bites the rare strata far harder
  than the head.

Cost 1.67e15 flops and 74 min of wall clock for the `D = 2.62e7` series (the top two rungs
are half of it), plus 6.2e14 for the diagnosis, billed to
`results/finite_support_ledger.jsonl` — its own ledger, no budget, and the closed 10¹³
student ledger is untouched; 2.84e15 in that ledger all told. The convex solves are
analysis rather than training and are not billed: ~1.5e15 flops, ~20 min. Output:
`results/finite_support_sweep.json`, which keeps **both** series (cells are keyed by step
count; `meta.series` records which one the last run produced) with every cell's
learning-rate grid and per-stratum losses. The slide's chart block is **hand-written** in
`figures/finite-chart.md`: there is no `--write-slide` and no BEGIN/END markers, so the
numbers there are copied from this JSON by hand, from the `excess_star` field. Both of its
series are at `D = 2.62e7`. `assets/finite-chart.js` styles its markers.

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
| `src/assocmem/capacity.py` | the capacity experiment: Hebbian vs trained, bisection on `n` |
| `src/assocmem/grid.py`, `grid_fit.py` | the (N, D) scan: stratified evaluator, one grid cell, the fits |
| `src/assocmem/emergence.py` | the emergence experiment: per-band eval set, accuracy kernel, one checkpointed run |
| `scripts/scaling_grid.py`, `alpha_sweep.py` | the two scans; `grid_figures.py`, `alpha_figures.py` draw them |
| `scripts/capacity_sweep.py` | the capacity sweep; `--write-slide` writes `figures/capacity-chart.md`, the plot on "Capacity, in theory and in practice" |
| `scripts/grid_slide.py` | turns the scans into the deck's Results slides: `--write-slide` writes `figures/results-alpha.md` (the exponents-vs-α chart) and the inline `results-fit` block in `slides.md` |
| `scripts/isoflop_sweep.py` | six widths per compute budget, centred on `N*`: the sweep the IsoFLOP figure is fitted to |
| `scripts/isoflop_slide.py` | the IsoFLOP construction: parabolas, minima, `N*(C)`, as a generated SVG with two reveals; `--write-slide` writes `figures/isoflop-figure.md` and the inline `isoflop-exponents` sentence in `slides.md` |
| `assets/results-chart.js` | the alpha result slide's line and marker styling |
| `scripts/emergence.py` | the emergence sweep, its six-panel figure, and the deck's accuracy-vs-`N` chart: `--write-slide` writes `figures/emergence-chart.md` (legend and styling script included), the figure on "Digression: emergent behavior under the hood" |
| `assets/emergence-chart.js` | the emergence chart's marker styling and log tick labels, loaded from the `<script src>` tag at the end of `figures/emergence-chart.md` |
| `scripts/finite_context_sweep.py` | the finite-context-pool sweep: online truncated Zipf, exact frequency-weighted eval |
| `assets/finite-chart.js` | the finite-data chart's filled markers (colloquium draws line points hollow by default) |
| `assets/pc-chart.js` | the two practice-capacity charts (digitised from Allen-Zhu & Li and Morris et al.): filled markers, dashed yardsticks, decade-only ticks written `1k` / `1M`. One file for the pair — the pass is shared and the tick plugin must be registered once — loaded from the `<script src>` tag at the end of `figures/pc-bitstrings.md`, which sits after both chart blocks |
| `slides.md` | the deck (colloquium). Build it with **`uv run python scripts/build_slides.py`**, not `colloquium build` — see below |
| `figures/*.html` | the deck's six hand-drawn SVG figures, one file each: `embed-fig`, `zipf-fig`, `w-build-fig`, `sphere-fig`, `loss-step-fig`, `scaling-twin-fig`. `slides.md` refers to each by a one-line `<!-- figure: <key> -->` placeholder and holds no SVG itself |
| `figures/pc-facts.md`, `figures/pc-bitstrings.md`, `figures/finite-chart.md` | the three *hand-written* chart figures — digitised or copied numbers, no generator script — each with its `cap-legend`. `slides.md` keeps only the placeholder, the surrounding prose and, on the finite-data slide, the `cap-cue` marker that belongs to the step sequence |
| `figures/capacity-chart.md`, `figures/isoflop-figure.md`, `figures/results-alpha.md`, `figures/emergence-chart.md` | the same placeholder mechanism for the four *plotted* figures, each written by its generator (`capacity_sweep.py`, `isoflop_slide.py`, `grid_slide.py`, `emergence.py` with `--write-slide`) together with its legend and its `assets/*.js` styling tag. Generated files: edit the script, not these |
| `scripts/build_slides.py` | expands those placeholders and runs colloquium: `build_slides.py` → `slides.html`, `--check` verifies every placeholder resolves and no figure is orphaned, and `--serve [-p 8090]` watches the slides, figures, bibliography, and assets while serving the expanded deck at `/slides.html`. Colloquium has no include directive, so `colloquium build slides.md` on its own renders the figures empty |
| `assets/slides.css` | the deck's stylesheet, pulled in by the `<link>` on the title slide; font URLs are relative to `assets/`, so it needs to sit next to `slides.html` |
| `assets/capacity-chart.js`, `assets/scaling-slider.js` | the deck's two scripts: capacity-chart markers and tick labels, and the α slider on the twin-axis scaling-law slide |

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
