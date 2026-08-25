---
title: "Scaling law tutorial"
author: "Nicolas Zucchet"
date: "2026-08-18"
theme: default
aspect_ratio: "16:9"
bibliography: refs.bib

fonts:
  heading: "Playfair Display"

footer:
  left: "Nicolas Zucchet"
  center: "Scaling law tutorial"
  right: "auto"

# The deck's CSS lives in assets/slides.css, pulled in by the <link> at the
# end of the title slide: colloquium inlines `custom_css` verbatim, so keeping
# 270 lines of stylesheet here would bury the slides.
---

# Scaling laws in large language models and toy models

Nicolas Zucchet -- [nzucchet@stanford.edu](mailto:nzucchet@stanford.edu)

September 8th 2026

<!-- The deck's stylesheet, for every slide: colloquium has no hook for an
     external sheet, and a <link> in the body is honoured all the same. -->

<link rel="stylesheet" href="assets/slides.css">

---

## Agenda

### Part I: scaling laws in practice
- Some **history** and why scaling laws have been (and are) **central to the development of LLMs**
- **What scaling laws are** and what do they tell us
- How scaling laws guide **model and learning recipe** development

### Part II: understanding where scaling laws come from with toy models
- Scaling laws arise in a **toy associative memory model**
- What did we **learn** from the toy model?

### Part III: let's train our own model using scaling laws
- Can we use what we learned to train the best possible model?

---

<!-- layout: section-break -->
# Part I: scaling laws in practice

---

## The Core Idea

We start from a simple objective:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \ell(f_\theta(x), y) \right]
$$

The <span class="highlight-red">first term</span> captures the data-fitting loss,
while the <span class="highlight-blue">second term</span> (added below) is a
regularizer:

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \lambda \, \|\theta\|_2^2
$$

Increasing <span class="highlight-green">$\lambda$</span> trades off fit against
simplicity.

---

<!-- layout: section-break -->
# Part II: understanding where scaling laws come from with toy models

---

## Where we stand

In Part I, we have seen how scaling laws are **useful in practice** and that they **appear everywhere**.

<!-- step -->

<div style="margin-top: 1.5em"></div>

There are many **open questions** though:
- **why power laws** (and not, e.g., exponential laws)?
- where do the **exponents come from**? how do they depend on the data?

<!-- step -->

<div style="margin-top: 1.5em"></div>

**Goals** of this part:
- introduce a **toy model** of natural language and LLMs
- get some theoretical **intuition** on how it behaves and how it yields scaling laws
- use it to **fit scaling laws** ourselves


---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

Next-token prediction requires learning a **mapping from contexts to next-token probability distributions**

<div class="tok-example">
<div class="tok-ctx"><span class="tok-ctx-group"><span class="tok-ctx-text">the cat sat on the </span><span class="tok-ctx-label">context</span></span></div>
<div class="tok-cols">
<div class="tok-tokens"><span class="tok-col-head">&nbsp;</span>mat<br>couch<br>floor<br>roof</div>
<div class="tok-model fragment" data-colloquium-fragment="1"><span class="tok-col-head">model</span>0.31<br>0.24<br>0.19<br>0.02</div>
<div class="tok-data fragment" data-colloquium-fragment="1"><span class="tok-col-head">data</span>0.5<br>0.18<br>0.32<br>0</div>
</div>
</div>

<!-- step -->

<div style="margin-top: 1.5em"></div>

The <strong class="tok-model">model</strong> explicitly represents **next-token probability distributions**, while the <strong class="tok-data">data</strong> only provides samples from an **implicit target distribution**.

<!-- step -->

<div style="margin-top: 1em"></div>

We further assume that **no structure shortcuts the learning**: nothing generalises from one context to another, so the model has **no choice but to memorise** the mapping, context by context.

---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and classification**

<!-- figure: embed-fig -->

We assume that the embeddings are iid distributed **uniformly on the unit sphere**.

---

<!-- rows: 3/9 -->
## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and classification**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Contexts follow a **power law distribution**

===

<!-- row-columns: 1/1 -->

Power laws appear in many places [@newman2005power]. **Zipf's law** for word frequencies is the famous example for language: **frequency is approximately inverse to rank** [@zipf1949human].

We assume that contexts follow a **power law**, that is

$$
p(i) \propto i^{-\alpha}.
$$

with $i$ the index of the context.

|||

<!-- figure: zipf-fig -->

---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and classification**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 4.** There is **only one** likely next token in the data

**Let's keep our life simple**, at least for now :)

---

## The toy model

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and classification**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 4.** There is **only one** likely next token in the data

<div style="margin-top: 1.5em"></div>

More formally, we have
- **Infinitely many different contexts**, with context $i$ appearing with probability $p(i) \propto i^{-\alpha}$.
- Each context $i$ has a **random embedding** $e_i$ drawn from the unit sphere of dimension $h$.
- The model predicts the next-token distribution through
$$p(\cdot | i) = \mathrm{softmax}(We_i).$$
- There is **only one** out of the $d$ tokens exists in the data (we have some multi-class classification problem).

---

## Some theoretical intuition

**Model.** $p(\cdot | i) = \mathrm{softmax}(We_i)$, with $p(i) \propto i^{-\alpha}$. Embedding dimension $h$, $d$ possible next tokens.

<div style="margin-top: 1.5em"></div>

Can we guess?
- How many **contexts can the model store** (100% accuracy)?
- How does the **optimal loss** scale as a function of the **model size**?


---

## Some theoretical intuition

**Model.** $p(\cdot | i) = \mathrm{softmax}(We_i)$, with $p(i) \propto i^{-\alpha}$. Embedding dimension $h$, $d$ possible next tokens.

<!-- step -->

<div style="margin-top: 1.5em"></div>

We can get some intuition of the storage capacity with the following **Hebbian model**
$$W = \sum_i z_i e_i^\top \quad \text{with } z_{ij} = 1 \text{ if } j = y^*_i \text{ and } 0 \text{ otherwise}.$$

<!-- step -->

Corresponds to taking **one gradient descent step**, starting from $W=0$, on the objective $$\frac{1}{2}\lVert W e_i - z_i \rVert^2.$$ 

<!-- step -->

<div style="margin-top: 1.5em"></div>

Contexts with embeddings close to $e_i$ will be **pushed towards outputting** $z_i$ (and thus to predict the correct next-token).

---

## Some theoretical intuition

**Model.** $p(\cdot | i) = \mathrm{softmax}(We_i)$, with $p(i) \propto i^{-\alpha}$. Embedding dimension $h$, $d$ possible next tokens.

<div style="margin-top: 1.5em"></div>

We can get some intuition of the storage capacity with the following **Hebbian model**
$$W = \sum_i z_i e_i^\top \quad \text{with } z_{ij} = 1 \text{ if } j = y^*_i \text{ and } 0 \text{ otherwise}.$$

Predicting the next-token for context $i$:
$$ We_i = \underset{\textbf{signal}}{\underline{z_i e_i^\top e_i \vphantom{\sum_{j\neq i}}}} + \underset{\textbf{noise}}{\underline{\sum_{j\neq i} z_j e_j^\top e_i}}$$

<!-- step -->
The prediction for context $i$ is perturbed by the **other embeddings close to** $e_i$.

---

## Some theoretical intuition

**Hebbian model.** $W = \sum_i z_i e_i^\top$, so the row of $W$ for token $y$ is $\sum_{i \,:\, y^*_i = y} e_i^\top$: **one sum per token**.

<!-- figure: w-build-fig -->

<!-- step -->

<div style="margin-top: 0.4em"></div>

**Querying.** $(W e_i)_y = \sum_{j \,:\, y^*_j = y} e_j^\top e_i$ -- compare $e_i$ against every stored embedding and see **which colour dominates**.

---

<!-- class: carry-line -->
## Some theoretical intuition

**Interference.** The prediction for context $i$ is perturbed by the **other embeddings close to** $e_i$.

<div style="margin-top: -1em"></div>

<!-- figure: sphere-fig -->

<!-- step -->

<div style="margin-top: 0.4em"></div>

**As a result**, capacity increases with $d$ and $h$.

---

<!-- columns: 1/1 -->
<!-- class: ct-slide -->
## Capacity, in theory and in practice

<!-- figure: capacity-chart -->

|||

In theory, capacity is **proportional to the number of parameters**, up to logarithmic factors. We fix $d=256$ and vary $h$.

<!-- step -->

At large $h$ the trained model is **exactly parallel to $N$** (slope $1.01$) and stores <span class="highlight-navy">0.16 memories per parameter</span>.

<div class="cap-cue" data-cap-series="trained"></div>

<!-- step -->

Hebbian gets <span class="highlight-red">0.02</span>, an **order of magnitude** less.

<div class="cap-cue" data-cap-series="hebbian"></div>

<!-- step -->

**To learn more.** [@cabannes2024scaling] for the full analysis, [@nichani2025factual] for other architectures, [@zucchet2026ambiguity] for several possible next tokens.

---

<!-- rows: 2/11/1 -->
<!-- class: pc-slide -->
## Capacity, in theory and in practice

In practice, capacity is also **proportional to the number of parameters** — the *total* count in a mixture of experts, so an MoE stores more per **active** parameter.

===

<!-- row-columns: 1/1 -->

**Facts.** GPT-2 on $N$ synthetic biographies, one dot per model.

<!-- figure: pc-facts -->

|||

**Random bitstrings.** GPT-2 trained to saturation, one curve per model size.

<!-- figure: pc-bitstrings -->

===

<div class="inline-footnote pc-cite">

Redrawn from Figure 1(b) of [@allenzhu2024capacity] and Figure 1 of [@morris2025memorization]; values read off the published plots.

</div>

---

<!-- rows: 4 -->
<!-- class: eq-rows -->
## From capacity to scaling laws

Now that we have some idea of how the capacity scales as a function of the model size, **how do we get scaling laws?**

<!-- step -->

**For the model size:** assume infinite data, and that the first contexts below capacity get 0 loss and some value $l$ afterwards.

===

<!-- row-columns: 3/2 -->

<!-- step -->

As a result:

$$L(N) = \sum_i p(i) \, L(i)$$

|||

<div class="text-sm fragment" data-fragment-index="2">

Average the per-context loss over **how often each context appears**.

</div>

===

<!-- row-columns: 3/2 -->

<!-- step -->

$$\phantom{L(N)} \propto \sum_{i \leq \mathrm{capacity}(N)} i^{-\alpha} \times 0 \; + \sum_{i > \mathrm{capacity}(N)} i^{-\alpha} \times l$$

|||

<div class="text-sm fragment" data-fragment-index="3">

Contexts **below capacity are free**; each one above costs $l$.

</div>

===

<!-- row-columns: 3/2 -->

<!-- step -->

$$\phantom{L(N)} \propto \mathrm{capacity}(N)^{1 - \alpha} \propto N^{1-\alpha}.$$

|||

<div class="text-sm fragment" data-fragment-index="4">

The tail sum is set by **where it starts**, and capacity grows like $N$.

</div>

---

## From capacity to scaling laws

**For the model size:** the loss is a step function of the context index, and the step moves right as $N$ grows.

<!-- figure: loss-step-fig -->

<!-- The figure builds in three steps (shade the area, carry it across to one
     point, then sweep N into the law); the caveat belongs with the law, so it
     waits for the last of them. -->

<div class="inline-footnote fragment" data-fragment-index="3">

Note that in this case there is no residual entropy and the loss will converge to $0$ under infinite compute and data.

</div>

---

<!-- rows: 3 -->
<!-- class: eq-rows -->
## From capacity to scaling laws

**For the dataset size:** assume instead that the model gets right every context it has **seen at least once**.

Context $i$ appears in the first $D$ tokens, in expectation, as soon as

===

<!-- row-columns: 3/2 -->

$$p(i) \, D \geq 1 \qquad \Longleftrightarrow \qquad i \leq D^{1/\alpha}.$$

|||

<div class="text-sm">

Only the **first $D^{1/\alpha}$ contexts** are ever seen.

</div>

===

<!-- row-columns: 3/2 -->

<!-- step -->

Using the same argument as before,

$$L(D) = \sum_{i > D^{1/\alpha}} p(i) \, l \; \propto \; \left(D^{1/\alpha}\right)^{1-\alpha} = D^{\frac{1}{\alpha} - 1}.$$

|||

<div class="text-sm fragment" data-fragment-index="1">

Same tail sum, cut at $D^{1/\alpha}$ **instead of the capacity**.

</div>


---

<!-- columns: 3/2 -->
## From capacity to scaling laws

We now have intuition for how the loss scales as a function of model size $N$ and data $D$.

<!-- figure: scaling-twin-fig -->

|||

Power laws in the **data** become power laws in the **loss**, in both model size $N$ and tokens $D$.

<!-- step -->

Whatever the value of $\alpha$, the **bottleneck is the data** $D$: we need more tokens than parameters.

<!-- step -->

As $\alpha$ increases and the tail becomes rarer, the data becomes **even more of a bottleneck** -- it takes longer to see the relevant data points.

---

## Are the toy model predictions good enough?

If we further **assume Chinchilla** scaling, we get:
$$L(N, D) = \frac{A}{N^{\alpha - 1}} + \frac{B}{D^{\frac{1}{\alpha}-1}}.$$

From the calculations we did in Slide [HERE] (and some minor expression massaging), we get
$$\begin{align*}
N^*(C) &\propto C^{\frac{1}{1+\alpha}}\\
D^*(C) &\propto C^{\frac{\alpha}{1+\alpha}}\\
L(C) &\propto C^{\frac{\alpha - 1}{1+\alpha}}
\end{align*}$$

Does this match what's happening **in practice**?

---

## Are the toy model predictions good enough?

At fixed **compute budgets**, we sweep the **model size** and fit a parabola in $\log N$.

<!-- figure: isoflop-figure -->

<div class="text-sm">

Six IsoFLOP profiles. Learning rates are tuned for each run.

</div>

<!-- BEGIN isoflop-exponents (generated by scripts/isoflop_slide.py) -->

<div style="margin-top: 1.2em"></div>

The loss decreases as $C^{-0.091}$ (predicted $C^{-0.091}$), the model size grows as $C^{0.449}$ (vs. $C^{0.455}$).

<!-- END isoflop-exponents -->

---

<!-- rows: 2/2/11 -->
<!-- class: chin-slide -->
## Are the toy model predictions good enough?

We assumed the **Chinchilla scaling** and recovered similar compute exponents as in theory, but **how good is the full law?**

===

<!-- row-columns: 1/1 -->

$$L-L_\infty=A N^{-a}+B D^{-b}.$$

|||

<div class="chin-independence">

$N$ and data $D$ treated as **independent**.

</div>

===

<!-- row-columns: 1/1 -->

<!-- BEGIN results-fit (generated by scripts/grid_slide.py) -->
Fit on **75 low-compute runs**, then test on 4 held-out runs
at $8$--$16\times$ the fitting budget.

<div class="fit-matrix">
<div></div><div class="fm-head">fit region</div><div class="fm-head">extrapolation</div>
<div class="fm-label fm-red">Chinchilla</div><div class="fm-value fm-red">3.2%</div><div class="fm-value fm-red">4.8%</div>
<div class="fm-label fm-navy">Skaling</div><div class="fm-value fm-navy">0.4%</div><div class="fm-value fm-navy">0.6%</div>
</div>

<div class="inline-footnote">

Relative rms error on $L-L_\infty$; $L_\infty$ is known.

</div>

<!-- END results-fit -->

|||

<div class="chin-story">

**Additive scaling laws do not extrapolate as well.**

Idea: **couple model size and data**.

$$L-L_\infty=\left(\frac{A}{N^a}+\frac{B}{D^b}\right)^k$$

**To learn more.** [@videau2026skaling] on why this functional form makes sense and how it better extrapolates.

</div>

---

<!-- columns: 1/1 -->
## Do the exponents follow the theory?

How good are the **scaling exponents** compared to the theory as we change the
data distribution, in particular $\alpha$?

<!-- step -->

To estimate those coefficients, we train **models of a given size until the loss
plateaus** (model coefficient), and train the **largest reasonable model varying the
dataset size** (data coefficient).

<!-- step -->

Our super super simple theory is **not too bad**!


|||

<!-- figure: results-alpha -->


---

<!-- columns: 1/1 -->
## What if data was finite?

Same experiment as before, but only keeping **first 10,000 contexts** (instead of infinitely many).

<!-- step -->

The loss **starts** decreasing as a **power law** and finishes as an **exponential** one.

<div class="cap-cue" data-cap-series="first 10k"></div>

<!-- step -->

This is the regime **deep learning used to be in**: with a bounded amount to learn, returns die faster than any power law.

|||

<!-- figure: finite-chart -->


---

## Conclusion and takeaways (Part II)

We modeled language modeling with an **associative memory** exposed to **power law data** distribution.

With **simple theory**, we were able to **accurately predict** scaling laws exponents.

```box
title: Takeway
tone: accent
content: Scaling power laws **naturally arise** when the data itself is produced in power laws.
```

**Warning.** With the evidence that we have, we cannot say anything on whether such a mechanism is the bottleneck in LLMs that yields specific scaling behavior. Our toy model makes predictions on how changes in the data change scaling laws, 

We have just touched upon (toy) models and theories of scaling, some cool papers here.


---

<!-- layout: section-break -->

# Part III: let's train our own model using scaling laws

--- 

## Training a frontier toy model

**Problem.** We have access to new important data and want to train the best possible (toy) model on it.

**Constraints.** You don't know what the model is and can control the number of parameters $N$, the number of tokens $D$, as well as the learning rate, and what are the tuning experiments you do. The total amount of compute for both hyperparameter optimization and the final run. 

**Your job.** Come up with the **best model**!


---

## Conclusion

The toy model already produces **power laws out of a power-law data distribution**: the exponents come from the **tail of the data**, not from the architecture.

<div style="margin-top: 0.8em"></div>

<!-- step -->

**Open questions, theoretically**
- Real language is **not a bag of independent facts**: what happens once contexts **share structure**?
- The toy model only **memorises** -- what is the analogue of capacity for **generalisation**?
- Trained models beat Hebbian by an **order of magnitude**: what sets that constant?
- **Emergence**: how do smooth power laws produce **sharp jumps** downstream?

---

## Conclusion

**Open questions, empirically**
- Do **data curation, synthetic data and repeated epochs** move the exponents, or only the constants?
- Do the same laws hold for **new architectures** (MoE, state-space models), and for **post-training and RL**?
- Can we predict **benchmark performance** rather than loss? And where does **inference-time compute** fit?

<div style="margin-top: 1.2em"></div>

<!-- step -->

The honest summary: we can **measure** scaling laws far better than we can **explain** them.
