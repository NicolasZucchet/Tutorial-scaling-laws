# Scaling laws for a linear-softmax associative memory

`p̂(·|x) = softmax(W eₓ)`, `W ∈ R^(512×n)`, no bias. Zipf(γ=1.2) inputs cut at `p=1e-14`
(⇒ V = 1.111·10¹¹ tokens, Z = 5.5607); fixed random conditionals with `E[exp H] = 16`.
Adam, batch 64, cosine `η → η/10`. **Budget 10¹³ flops** — `6ND` for training plus `2NM` for
every evaluation pass — of which **99.8 % used**. Free knobs: the width `n` (so `N = 512n`),
the number of steps, the peak lr `η`. All numbers in **nats**. Figure: `figures/final.png`.

## Recipe

Three screening rounds, each an IsoFLOP rung: at fixed compute `C`, sweep `n`, take the best
loss per `n`, fit a parabola in `log n` → `(n*, L*)`. Four rungs, `2·10⁹` → `10¹¹` (50× span),
then extrapolate 88× to the hero run.

| law | fit | quality |
|---|---|---|
| optimal width | `n* = 1.23·10⁻³ · C^0.4949` | r² = 0.9999 |
| optimal peak lr | `η* = 45.0 · C^-0.2802` | 4 bracketed lr parabolas, one per rung |
| loss | `L* = L∞ + 14.4 · C^-0.0995`, `L∞ = 2.468` | r² = 0.9997 |

**Three tricks make screening 6× cheaper than the obvious design, at no cost in law quality.**
(1) `η*` turned out independent of `n` at every rung, so a 3-point lr parabola costs 2 extra
runs per rung, not 2 per `n` — which is why all four rungs got one. (2) A rung costs
(#configs × C), so ladder *span* is far cheaper bought at the bottom (`2·10⁹`) than at the top;
extrapolating 88× instead of 17× hurts the prediction, not the score. (3) Keep every screening
run ≥ ~200 steps — below that the cosine schedule is a different regime and `η*` stops
extrapolating, which sets the floor rung at ~`2·10⁹`. Init scale is irrelevant (0, 0.3, 1, 3
within 0.005 nats): use `W = 0`, the correct uniform prior.

Equivalently `N ∝ C^0.495`, `D ∝ C^0.505` — an almost exactly even split, at **0.57 tokens per
parameter**. That is the opposite regime from LLM scaling (~20 tokens/param): here capacity, not
data, is scarce, because a token only has to be *seen* a few times to be memorised.

## Hero-run parameters

| | |
|---|---|
| width `n` / params `N` | **3136** / **1 605 632** (`W`: 512×3136) |
| steps / tokens `D` | **14 315** / **916 160** (batch 64) |
| peak lr → final lr | **0.01062 → 0.001062**, cosine, no warmup |
| optimizer / init | Adam (0.9, 0.999, 1e-8), no weight decay / **W = 0** |
| train flops | **8.826·10¹²** (+3.68·10¹¹ for its evaluations) |

## Expected loss

**3.256 nats**, from the 3-parameter fit (`L∞` free: `2.776 + 21.6·C^-0.1276`). The 2-parameter
fit with `L∞` pinned to the measured irreducible loss says 3.208 — I quote the free fit because
it has been the better extrapolator in every attempt (see below). Uncertainty ±0.03.

## Actual loss

**3.2609 nats** on 65 536 held-out tokens `x ~ p(x)`, using the exact expected cross-entropy
`−Σ_y p(y|x) log p̂(y|x)` — an unbiased, lower-variance estimator of the population loss.
Cross-checks: **3.2492** with the plain sampled-`y` cross-entropy on the same tokens, **3.2605**
on an independent 32 768-token set. Irreducible loss (mean conditional entropy) is 2.4632, so the
excess is **0.7977 nats**: perplexity 26.1 against an irreducible 11.7 and a uniform 512.

**Prediction error +0.005 nats (free fit) / +0.053 (pinned fit).**

## Two findings

**1. Compute spent tuning is the dominant cost.** Three attempts, same problem instance, same
eval set, differing mainly in how much of the budget went to screening:

| attempt | tuning | law quality | hero compute | hero loss |
|---|---|---|---|---|
| careful — 5 rungs, lr parabola only at the top | 47 % | r² 0.9992 | 5.01·10¹² | 3.2993 |
| notebook defaults — 2 lrs per rung, no top parabola | 26 % | r² 0.9983 | 6.97·10¹² | 3.2765 |
| **this run** — cheap wide ladder, lr parabola per rung | **8 %** | r² 0.9997 | **8.83·10¹²** | **3.2609** |

All three hero runs — different `n`, different `η`, from independently fitted laws — collapse
onto a single power law `L − L∞ = 9.48·C^-0.0830` with residuals of **0.0001 nats**. The hero
loss is set by the compute it receives and is almost indifferent to the recipe details, which is
why the cheapest screening wins. With zero-cost tuning the ceiling would be 3.2552; this run is
**+0.0057** from it, versus +0.0441 for the careful attempt. Since `α ≈ 0.083`, 10¹² of screening
costs ~0.008 nats, while being 20 % wrong in `n` costs 0.0014 and 30 % wrong in `η` costs 0.006 —
so the right strategy is *get `η` right, ignore `n`, and screen as little as the laws allow.*

**2. Do not fit the loss exponent — derive it from the width exponent.** The screening ladder
measures `α = 0.0995`, but the true large-`C` value is 0.0830, so the pinned power law
over-predicts by 0.05 nats at 88×. The two exponents are not independent: if capacity `∝ n^c` and
the unlearned Zipf tail gives excess `∝ K^-(γ-1)`, then `b = 1/(1+cγ)` and

  **α = (γ−1)(1−b)/γ**

Measured `b = 0.4949` implies `c = 0.85` (interference makes capacity grow slightly sublinearly
in `n`) and `α = 0.0842` — against 0.0830 traced by the three hero runs. Anchored at the top rung,
this rule predicts **3.2582** (error **+0.0027**), and applied retrospectively to the two earlier
attempts it gives +0.0096 and +0.0076 — a 6–20× smaller error than their fitted power laws,
using no extra compute.

## Compute ledger

| item | rungs | flops | share |
|---|---|---|---|
| round 1 | `2·10⁹`, `10¹⁰` — 5 widths + lr parabola each | 9.18·10¹⁰ | 0.9 % |
| round 2 | `3·10¹⁰` — 4 widths + lr parabola | 1.86·10¹¹ | 1.9 % |
| round 3 | `10¹¹` — 3 widths + lr parabola | 5.07·10¹¹ | 5.1 % |
| hero run | train + all evaluations | 9.19·10¹² | 91.9 % |
| **total** | 25 screening runs + 1 hero | **9.98·10¹²** | **99.8 %** |

Remaining levers, both untested: the hero's own evaluations cost 3.68·10¹¹ (0.003 nats' worth),
and Adam's `β₂`/`ε` were left at defaults throughout — the task pins "Adam" but not its
hyper-parameters, so that is the one unexplored axis that could plausibly move the loss by more
than the 0.006 nats still separating this run from the zero-tuning ceiling.
