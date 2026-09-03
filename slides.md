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

MLSS Tübingen Tutorial -- September 8th 2026

<!-- The deck's stylesheet, for every slide: colloquium has no hook for an
     external sheet, and a <link> in the body is honoured all the same.  The
     script beside it tidies the punctuation of the reference list colloquium
     generates at the end of the deck. -->

<link rel="stylesheet" href="assets/slides.css">
<script src="assets/references.js"></script>

---

## Why scaling laws?

The early successes of large language models relied on **two main observations**:
1. As we scale models and data, the next-token prediction improves in a **predictable way**.
2. As loss improves, the ability of the model to **solve tasks** we ultimately care about increase.

<!-- step -->
<div style="margin-top: 1.5em"></div>

Scaling laws are the science behind **1.** and are what justified the huge investments in compute.

<!-- step -->
<div style="margin-top: 1.5em"></div>

They interesting from an **engineering perspective**, (useful tools to design models), and from a **scientific standpoint** (suggest some universal principles underlying learning to be discovered!).

<!-- step -->
<div style="margin-top: 1.5em"></div>

```box
title: Goal of the tutorial
tone: accent
content: Build **intuition** on why scaling law are **useful** and **why they hold**.
```

---

## Agenda

### (Very) quick crash course on language modeling

<!-- step -->
<div style="margin-top: 1.5em"></div>

### Part I: scaling laws in practice
- **What scaling laws are** and what do they tell us
- How scaling laws guide **model and learning recipe** development

<!-- step -->
<div style="margin-top: 1.5em"></div>

### Part II: understanding where scaling laws come from with toy models
- Scaling laws arise in a **toy associative memory model**
- What did we **learn** from the toy model?

<!-- step -->
<div style="margin-top: 1.5em"></div>

### Part III: train our own model using scaling laws

---

<!-- layout: section-break -->
# (Very) quick crash course on language modeling

---

## Language model pretraining

In modern large language models, there are many training stages; we focus on **pretraining**.

**Context** in, **next-token distribution** out.

<!-- figure: llm-arch-fig -->

<!-- step -->

At inference: **sample**, append, repeat.

---

<!-- layout: section-break -->
# Part I: scaling laws in practice

---

## The observation

Train a family of language models, plot the **best loss** each one reaches against the **compute** it cost, and the points fall on a **straight line on log-log axes** -- over eight orders of magnitude.

<!-- figure: envelope-fig -->

<div class="inline-footnote">

The compute-efficient frontier of the runs in [@kaplan2020scaling], Figure 2; the next two slides are where such a curve comes from.

</div>

<!-- step -->
<div style="margin-top: 1.5em"></div>

Nothing about deep learning says this **has** to happen. That it does is what lets us **predict** a model before paying to train it -- but to state it precisely we need the **loss**, the **model size**, the **data** and the **compute**.

--- 

<!-- animate: bullets -->
## Loss, model size, data and compute

The **variables** we care about are:
- **Loss $L$.** Next-token prediction cross-entropy, which is the **objective of learning** (in the pretraining phase).
- **Model size $N$.** Number of the **parameters** the neural network has.
- **Number of tokens $D$.** Total **number of tokens trained on**. Each token usually serves as target of the next token objective once, but can appear in the context much more often.
- **Compute $C$.** Total number of **elementary operations** (e.g., addition or multiplication in a certain numerical precision like FP16) performed to train the entire model. Proxy for how much it **cost** to train a model.

<!-- step -->

<div style="margin-top: 1.5em"></div>

$N$, $D$ and $C$ are actually **coupled**: we will now show that $C = 6ND$. 

---

## How much does one matrix multiplication cost?

A linear layer does $y = Wx$, where
$W\in\mathbb{R}^{m\times n}$ has $N=mn$ parameters.

<!-- figure: matmul-dot-fig -->

---

## From one multiplication to training compute

**Learning matrix weights** for **each new token** in the context requires the **same number of flops**.

<!-- figure: training-flops-fig -->

<!-- step -->

<div style="margin-top: 1.5em"></div>

For **Transformers**, matrix multiplications **dominate** the FLOPs cost (but not memory!), so $C = 6ND$ approximately holds; see [@kaplan2020scaling] for details.

<!-- step -->

<div style="margin-top: 1.5em"></div>

**Side note.** This holds for dense Transformers; only the number of **activated** parameters matter in mixture of experts, they **increase the number of parameters** while keeping **FLOPs constant**.

---

## Scaling laws

<!-- rows: 8 -->

How does the loss $L$ evolve as a function of $D$, $N$, and $C$?

===

<!-- row-columns: 3/2 -->
<!-- figure: kaplan-tokens-fig -->

<div class="inline-footnote">
Training curves from [@kaplan2020scaling].
</div>

|||

A **larger model** reaches any given loss after **fewer tokens**.

<div style="margin-top: 1.5em"></div>

Performance improves as we scale models up, but this does not look as smooth as what we were promised...



---

## Scaling laws

<!-- rows: 8 -->

How does the loss $L$ evolve as a function of $D$, $N$, and $C$?

===

<!-- row-columns: 3/2 -->

<!-- figure: kaplan-compute-fig -->

<div class="inline-footnote">
Training curves from [@kaplan2020scaling].
</div>

|||

The same runs against compute. The **envelope** (best loss for a given budget) **decreases smoothly** with compute.

<div style="margin-top: 1.5em"></div>

<div class="fragment" data-fragment-index="1">

A compute-optimal model is **undertrained**: optimal training stops with loss still ~$8\%$ above convergence level.

</div>

---

## Scaling laws to train the best large language model

In traditional deep learning
1. **tune hyperparameters** of the model (learning rate, structure of the network, architecture choices...)
2. see how it affects **validation loss**
3. **train the final model** using the best hyperparameters

<!-- step -->
<div style="margin-top: 1.5em"></div>

**Problem:** we can only train our large language model **once**; how should we pick the hyperparameters?

<!-- step -->
<div style="margin-top: 1.5em"></div>

We can leverage **scaling laws** to do that!

---

## Compute-optimal model and dataset sizes

We now know what the **compute-optimal loss** looks like, but not how to achieve it (i.e. how to scale $N$ and $D$).

<!-- step -->
<div style="margin-top: 1.5em"></div>

Different strategies exist:
- **Pareto front**
- **IsoFLOP curves**
- **Parametric fit**

<!-- step -->
<div style="margin-top: 1.5em"></div>

All should end up with the **same result**, but they manipulate the data in different ways; it is great to build **intuition** and will be **useful** for Part III, so we review them next following [@hoffmann2022training].



---

## Pareto front

For each **model size**, compare loss against compute (exactly as we did before, but this time with other data).

<!-- figure: chinchilla-pareto -->

<div class="inline-footnote">

All $43$ model sizes of [@hoffmann2022training] present in the reconstruction of its Figure 4 by [@besiroglu2024chinchilla]; the legend names six of them.

</div>



--- 

## IsoFLOP curves

Same runs, sliced the other way: at **fixed compute budgets**, sweep the **model size**.

<!-- figure: chinchilla-isoflop -->

<div class="inline-footnote">

One $\log L$ parabola in $\log N$ fitted per compute budget, over all nine budgets of [@hoffmann2022training].

</div>

<div class="fragment" data-fragment-index="3">


</div>

--- 

<!-- columns: 1/1 -->
<!-- class: chin-law -->
## Parametric fit

One law for **all** the runs:

$$L=E + \frac{A}{N^{a}}+\frac{B}{D^{b}}$$

<div class="chin-consts">

$E=1.81$, $A=476$, $a=0.35$, $B=1799$, $b=0.36$

</div>

**Note.** Only this construction models the **irreducible loss** $E$, the entropy of the text itself.

<div class="fragment" data-fragment-index="3">

Optimizing the loss under the constraint $C=6ND$ gives $N^*\propto C^{0.51}$ and $D^*\propto C^{0.49}$.

</div>


|||

<!-- figure: chinchilla-parametric -->

<div class="inline-footnote">

Data from [@hoffmann2022training], fitting from [@besiroglu2024chinchilla] (all data points except ones that are too overparametrized).

</div>

---

<!-- rows: 4 -->
<!-- class: eq-rows -->
## Chinchilla rule of thumb


Look at the Chinchilla law more closely:
$$L = E + A \, N^{-a} + B \, D^{-b}.$$
The two exponents, $a=0.35$ and $b=0.36$, are **almost the same**.


<!-- step -->
<div style="margin-top: 1.5em"></div>

As a result, the ratio between the compute-optimal model and dataset size remains roughly **constant**.


<!-- step -->
<div style="margin-top: 1.5em"></div>

Rule of thumb: **$20$ tokens per parameter**. 
Way less than what [@kaplan2020scaling] advocated for before [@besiroglu2024chinchilla], and the current trend is towards a larger number for other reasons.

<!-- step -->
<div style="margin-top: 1.5em"></div>

**Exercise.** Derive the **optimal model size** and **dataset size** from the parametric fit and justify the claim that we have made here.

---

## One layer deeper

So far, we have only discussed **two parameters** $N$ and $D$ that we can play with.

<!-- step -->

<div style="margin-top: 1.5em"></div>

What about more specific **architecture details**? e.g., number of layers, width of the model, the number of attention heads...

And **optimization details**? e.g., batch size, sequence length...

<!-- step -->
<div style="margin-top: 1.5em"></div>

Too expensive to **retune everything everytime**, so there are some choices that everyone is making (e.g., what expansion ratio should an MLP have, ratio width / depth of the network, what learning rates should be etc...)

<!-- step -->
<div style="margin-top: 1.5em"></div>

As a result, scaling laws are **recipe-specific** (on top of being data specific)!

<!-- step -->
<div style="margin-top: 1.5em"></div>

That said, there are some principles that one could use: compare recipe-specific **scaling laws**.

--- 

## What about other hyperparameters

How to develop a **recipe** (model / optimizer / data...) when we only have limited compute? What matters is how good it will be at **large scale**, but we can only afford **small runs**.

One size cannot tell us: compare **scaling laws**.

<!-- figure: recipe-trends-fig -->

<!-- step -->
<div style="margin-top: 1.5em"></div>

**PSA:** Only fully trust methods that use baselines with **properly tuned learning rates**!!

--- 

## Conclusion and takeaways (Part I)

In next-token prediction language modeling, performance (and other observables) **increases smoothly as a function of compute**.

<!-- step -->
<div style="margin-top: 1.5em"></div>

Scaling laws provide us with a **framework** to design the **best large model** possible when we cannot afford to train it multiple times.

<!-- step -->
<div style="margin-top: 1.5em"></div>

This paradigm applies elsewhere, as long as
1. **model is powerful enough** (Transformers usually are!)
2. **data is rich enough** and the long tail captures **behavior we care about**


<!-- step -->
<div style="margin-top: 1.5em"></div>

Such conditions hold in **other domains**, like vision [@zhai2022scaling] or time series [@edwards2024scaling].

--- 

<!-- animate: bullets -->
## To learn more

We have only **scratched the surface** of scaling laws...

As large language models are becoming more and more complex and used in different practical scenarios, many questions arise:
- data is **diverse**, e.g. many languages, how should we balance them? [@longpre2026atlas]
- what happens when we **run out** of internet data? [@muennighoff2023scaling]
- serving models also costs compute: how to split compute between **training and inference**? [@sardana2024beyond; @jones2021scaling]
- architectures are getting more complex, e.g. **mixture of experts** [@clark2022unified] or **different sequence layers** [@poli2024mechanistic]: do the laws carry over?
- training now has **many stages**: how does that change the tradeoffs? [@khatri2026art]

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

<div style="margin-top: 1.5em"></div>

**Assumption.** No structure shortcuts the learning; nothing generalizes from one context to another, so the model has **no choice but to memorize** the mapping, context by context.

---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and prediction**

<!-- figure: embed-fig -->

<!-- The sentence belongs to the figure's embedding beat, so it carries that
     beat's index by hand instead of opening a step of its own: the figure
     already opens both of the slide's steps, and step 1 is the one that draws
     the unit circle. -->

<div class="fragment" data-fragment-index="1">

We assume that the embeddings are iid distributed **uniformly on the unit sphere**.

</div>

---

<!-- rows: 3/9 -->
## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and prediction**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Contexts follow a **power law distribution**

===

<!-- row-columns: 1/1 -->

Power laws appear in many places [@newman2005power]. **Zipf's law** for word frequencies is the famous example for language: **frequency is approximately inverse to rank** [@zipf1949human].


<!-- This step is the slide's only marker-generated one, and it is what the
     Zipf plot in the right-hand column pins itself to: the two containers are
     separate, so the figure repeats this beat's index by hand.  Keep this
     marker ahead of the figure -- colloquium numbers markers in document
     order, so moving it after the column split would renumber the beat and
     silently strand the plot. -->
<!-- step -->
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

**Assumption 2.** Sequence models as **random embedding generation and prediction**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 4.** There is **only one** likely next token in the data

**Let's keep our life simple**, at least for now :)

---

## The toy model

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Sequence models as **random embedding generation and prediction**

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
- There is **only one** out of the $d$ tokens exists in the data (we have some multi-class prediction problem).

---

## Some theoretical intuition

**Model.** $p(\cdot | i) = \mathrm{softmax}(We_i)$, with $p(i) \propto i^{-\alpha}$. Embedding dimension $h$, $d$ possible next tokens.

<div style="margin-top: 1.5em"></div>

Can we guess?
- How many **contexts can the model store** (100% accuracy)?
- How does the **optimal loss** scale as a function of the **model size**?
- How does the **optimal loss** scale as a function of the **dataset size**?

<div style="margin-top: 1.5em"></div>

In the following:
1. **Derivation of the capacity** of the model and scaling as a function of **model size $N$**.
2. **Heuristic derivation** for the scaling as a function of **dataset size $D$**.

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

**Hebbian model.** $W = \sum_i z_i e_i^\top$.

<!-- figure: w-build-fig -->

<!-- step -->

<div style="margin-top: 1.5em"></div>

**Querying.** $(W e_i)_y = \sum_{j \,:\, y^*_j = y} e_j^\top e_i$: compare the query $e_i$ with all the other embeddings and see **which color dominates**.

---

<!-- class: carry-line -->
## Some theoretical intuition

**Interference.** The prediction for context $i$ is perturbed by the **other embeddings close to** $e_i$.

<!-- figure: sphere-fig -->

<!-- step -->

<div style="margin-top: 1.5em"></div>

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

<!-- rows: 1/4 -->
<!-- class: pc-slide -->
## Capacity, in theory and in practice

In practice, capacity is also **proportional to the number of parameters**. For Mixture of Experts, the **total** number of parameters matter, so they store **more knowledge per active parameter**.

===

<!-- row-columns: 1/1 -->

<!-- figure: pc-facts -->

<div class="inline-footnote pc-caption">

**Stored synthetic facts**, 1000 exposures each [@allenzhu2024capacity]

</div>

|||

<!-- figure: pc-bitstrings -->

<div class="inline-footnote pc-caption">

**Random bitstrings memorized** [@morris2025memorization]

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

<!-- step -->

Context $i$ appears in the first $D$ tokens, in expectation, as soon as

===

<!-- row-columns: 3/2 -->

<!-- step -->

$$p(i) \, D \geq 1 \qquad \Longleftrightarrow \qquad i \leq D^{1/\alpha}.$$

|||

<div class="text-sm fragment" data-fragment-index="2">

Only the **first $D^{1/\alpha}$ contexts** are ever seen.

</div>

===

<!-- row-columns: 3/2 -->

<!-- step -->

Using the same argument as before,

$$L(D) = \sum_{i > D^{1/\alpha}} p(i) \, l \; \propto \; \left(D^{1/\alpha}\right)^{1-\alpha} = D^{\frac{1}{\alpha} - 1}.$$

|||

<div class="text-sm fragment" data-fragment-index="3">

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

As $\alpha$ increases and the tail becomes rarer, the data becomes **even more of a bottleneck**; it takes longer to see the relevant data points.

<!-- step -->

**Note.** Faster loss decay is not necessarily a good thing here, as the long tail takes **longer to learn**.

---

## Are the toy model predictions good enough?

If we further **assume Chinchilla** scaling, we get:
$$L(N, D) = \frac{A}{N^{\alpha - 1}} + \frac{B}{D^{\frac{1}{\alpha}-1}}.$$

<!-- step -->

Optimizing under the constraint $C=6ND$ gives
$$\begin{align*}
N^*(C) &\propto C^{\frac{1}{1+\alpha}}\\
D^*(C) &\propto C^{\frac{\alpha}{1+\alpha}}\\
L(C) &\propto C^{\frac{1 - \alpha }{1+\alpha}}
\end{align*}$$

<!-- step -->

Does this match what's happening **in simulations**?

---

## Are the toy model predictions good enough?

At fixed **compute budgets**, we sweep the **model size** and fit a parabola in $\log N$.

<!-- figure: isoflop-figure -->

<div class="text-sm">

Six IsoFLOP profiles. Learning rates are tuned for each run.

</div>

<!-- The figure builds over three clicks (the second panel, the envelope, then
     N*(C)); the fitted numbers are the punchline, so they wait for all of them. -->

<!-- step -->

<!-- BEGIN isoflop-exponents (generated by scripts/isoflop_slide.py) -->

<div style="margin-top: 1.5em"></div>

The loss decreases as $C^{-0.091}$ (predicted $C^{-0.091}$), the model size grows as $C^{0.449}$ (vs. $C^{0.455}$).

<!-- END isoflop-exponents -->

---

<!-- rows: 2/2/11 -->
<!-- class: chin-slide -->
## Are the toy model predictions good enough?

We assumed the **Chinchilla scaling** and recovered similar compute exponents as in theory, but **how good is the full law?**

===

<!-- row-columns: 1/1 -->

$$L= E + A N^{-a}+B D^{-b}.$$

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
<div class="fm-label fm-red fragment" data-colloquium-fragment="1">Chinchilla</div><div class="fm-value fm-red fragment" data-fragment-index="1">3.2%</div><div class="fm-value fm-red fragment" data-fragment-index="1">4.8%</div>
<div class="fm-label fm-navy fragment" data-colloquium-fragment="1">Skaling</div><div class="fm-value fm-navy fragment" data-fragment-index="2">0.4%</div><div class="fm-value fm-navy fragment" data-fragment-index="2">0.6%</div>
</div>

<div class="inline-footnote">

Relative rms error on $L-E; $E$ is known here (0).

</div>

<!-- END results-fit -->

|||

<div class="chin-story fragment" data-fragment-index="2">

**Additive scaling laws do not extrapolate well.**

Idea: **couple model size and data**.

$$L = E + \left(\frac{A}{N^a}+\frac{B}{D^b}\right)^k$$

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

<div class="fragment" data-fragment-index="1">

<!-- figure: results-alpha -->

</div>


---

<!-- columns: 1/1 -->
## What if data was finite?

Same experiment as before, but only keeping **first 10,000 contexts** (instead of infinitely many).

<!-- step -->

The loss **starts** decreasing as a **power law** and finishes as an **exponential** one.

<!-- The cue names the solid series in full.  `first 10k` on its own also
     matches `first 10k, tested on all`, and assets/plot.js resolves a cue by
     longest match, so the specific name is what keeps the two blue curves on
     separate beats. -->

<div class="cap-cue" data-cap-series="first 10k contexts"></div>

<!-- step -->

This is the regime **deep learning used to be in**: with a bounded amount to learn, returns die faster than any power law.

<!-- step -->

But score those **same models** on the full distribution and the gain largely **vanishes**: the held-out loss flattens, and stays **worse** than training on everything.

<div class="cap-cue" data-cap-series="tested on all"></div>

|||

<!-- figure: finite-chart -->


---

<!-- columns: 1/1 -->
## Digression: emergent behavior under the hood

The same nine models, asked a **threshold question** instead of an average one: for a band of context ranks, does the model predict the **most likely next token**?

<!-- step -->

Each band **switches on** at its own model size, one after the other -- and each switch is a **sigmoid over about a decade** of $N$, not a step.

<!-- step -->

Nothing jumps in the average, though: over the same models the loss slides smoothly from $1.55$ to $0.55$ nats. **Emergence can hide inside a power law.**

|||

<!-- figure: emergence-chart -->

<div class="inline-footnote">

Top-1 accuracy against the mode of $p(\cdot|i)$, counted uniformly inside each band; chance is $1/512$. The head bands saturate near $0.95$ because these conditionals are broad.

</div>

---

## Conclusion and takeaways (Part II)

We modeled language modeling with an **associative memory** exposed to **power law data** distribution.

With **simple theory**, we were able to **accurately predict** scaling laws exponents.

```box
title: Takeaway
tone: accent
content: Scaling power laws **naturally arise** when the data itself is produced in power laws.
```

<div style="margin-top: 1.5em"></div>

**Warning.** We cannot say anything on whether **this mechanism is the bottleneck** that yields specific scaling behavior, we can just say **how changing the data affects scaling** (which would be much harder in practice!).

We have **just touched upon** (toy) models and theories of scaling, some **cool papers** here.


---

<!-- layout: section-break -->

# Part III: let's train our own model using scaling laws

--- 

## Training a frontier toy model

**Problem.** We have access to new important data and want to train the **best possible (toy) model** on it.

<!-- step -->
<div style="margin-top: 1.5em"></div>

**Constraints.** You don't know what the model is and can control the **number of parameters** $N$, the **number of tokens** $D$, as well as the **learning rate**, and what are the tuning experiments you do. The total amount of compute for both hyperparameter optimization and the final run is fixed. 

<!-- step -->
<div style="margin-top: 1.5em"></div>

**Your job.** Come up with the **best model**!

