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
---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Contexts follow a **power law distribution**

<!-- don't use specific classes, does colloquium has a column-based layout starting from the middle of the page? this would be much cleaner? reduce size of the plot, remove small text. Make sure to check the other slides so that this does not appear again.
For the plot, instead of putting log scale, put the actual probability and context rank, e.g. 1 10 100 1k 10k 100k and the actual frequencies in base 10. Use alpha = 1.2 in the exemple. Remove the red dots. Move many rare contexts a slightly to the left.
-->
<div class="zipf-layout">
<div class="zipf-copy">

Power laws appear in many places [@newman2005power]. **Zipf's law** for word frequencies is the famous example for language: **frequency is approximately inverse to rank** [@zipf1949human].

We assume that contexts follow a **power law**, that is
$$
p(i) \propto i^{-\alpha}.
$$
with $i$ the index of the context.


<!>

</div>
<svg class="zipf-fig" viewBox="0 0 580 330" role="img" aria-label="A log-log rank-frequency plot following Zipf's law">
  <line class="zf-axis" x1="72" y1="25" x2="72" y2="278"/>
  <line class="zf-axis" x1="72" y1="278" x2="548" y2="278"/>
  <path class="zf-guide" d="M88 48 L520 251"/>
  <circle class="zf-dot" cx="88" cy="48" r="6"/>
  <circle class="zf-dot" cx="137" cy="72" r="5.5"/>
  <circle class="zf-dot" cx="180" cy="94" r="5"/>
  <circle class="zf-dot" cx="222" cy="114" r="4.8"/>
  <circle class="zf-dot" cx="267" cy="135" r="4.5"/>
  <circle class="zf-dot" cx="310" cy="157" r="4.2"/>
  <circle class="zf-dot" cx="357" cy="177" r="4"/>
  <circle class="zf-dot" cx="401" cy="198" r="3.8"/>
  <circle class="zf-dot" cx="447" cy="220" r="3.6"/>
  <circle class="zf-dot" cx="494" cy="242" r="3.4"/>
  <text class="zf-label" x="34" y="154" text-anchor="middle" transform="rotate(-90 34 154)">frequency p(i)</text>
  <text class="zf-label" x="310" y="319" text-anchor="middle">context rank i</text>
  <text class="zf-note" x="111" y="37">few frequent contexts</text>
  <text class="zf-note" x="358" y="267">many rare contexts</text>
  <text class="zf-law" x="359" y="101">slope = −α</text>
  <text class="zf-log" x="82" y="298">log scale</text>
  <text class="zf-log" x="485" y="298">log scale</text>
</svg>
</div>

---

## Simplifying language modeling

<!-- let's swap the order of  2 and 3 (slides and also the list). Here and everywhere relevant. -->

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Sequence models as **random embedding generation and classification**

<!-- comments, reduce the size of the contexts, should be small text. Model embeddings as vector as in one of the next figure, on the unit circle
only three of them, move the figure, but not titles a bit lower -->

<svg class="embed-fig" viewBox="0 0 1160 320" role="img" aria-label="Contexts mapped to random embeddings and then classified into next-token probabilities">
  <defs>
    <marker id="ef-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
  </defs>
  <text class="ef-head" x="160" y="35" text-anchor="middle">contexts</text>
  <text class="ef-head" x="575" y="35" text-anchor="middle">embedding space</text>
  <text class="ef-head" x="994" y="35" text-anchor="middle">next token</text>

  <rect class="ef-context" x="22" y="68" width="278" height="56" rx="8"/>
  <rect class="ef-context" x="22" y="145" width="278" height="56" rx="8"/>
  <rect class="ef-context" x="22" y="222" width="278" height="56" rx="8"/>
  <text class="ef-code" x="42" y="102">the cat sat on the …</text>
  <text class="ef-code" x="42" y="179">Paris is the capital of …</text>
  <text class="ef-code" x="42" y="256">water freezes at …</text>

  <line class="ef-arrow" x1="318" y1="173" x2="398" y2="173"/>
  <text class="ef-muted" x="358" y="151" text-anchor="middle">encode</text>

  <circle class="ef-space" cx="575" cy="173" r="132"/>
  <line class="ef-grid" x1="443" y1="173" x2="707" y2="173"/>
  <line class="ef-grid" x1="575" y1="41" x2="575" y2="305"/>
  <circle class="ef-point ef-blue" cx="511" cy="108" r="8"/>
  <circle class="ef-point ef-red" cx="640" cy="121" r="8"/>
  <circle class="ef-point ef-green" cx="544" cy="239" r="8"/>
  <circle class="ef-point ef-orange" cx="658" cy="224" r="6"/>
  <circle class="ef-point ef-navy" cx="482" cy="198" r="6"/>
  <text class="ef-vector" x="493" y="91">e₁</text>
  <text class="ef-vector" x="650" y="109">e₂</text>
  <text class="ef-vector" x="524" y="264">e₃</text>

  <line class="ef-arrow" x1="727" y1="173" x2="807" y2="173"/>
  <g>
  <text class="ef-token" x="835" y="93">mat</text><rect class="ef-bar ef-blue" x="900" y="76" width="183" height="22" rx="3"/><text class="ef-prob" x="1096" y="94">.52</text>
  <text class="ef-token" x="835" y="137">couch</text><rect class="ef-bar ef-blue-light" x="900" y="120" width="81" height="22" rx="3"/><text class="ef-prob" x="994" y="138">.23</text>
  <text class="ef-token" x="835" y="181">floor</text><rect class="ef-bar ef-blue-light" x="900" y="164" width="53" height="22" rx="3"/><text class="ef-prob" x="966" y="182">.15</text>
  <text class="ef-token" x="835" y="225">roof</text><rect class="ef-bar ef-blue-light" x="900" y="208" width="25" height="22" rx="3"/><text class="ef-prob" x="938" y="226">.07</text>
  </g>
</svg>

We assume that the embeddings are iid distributed **uniformly on the unit sphere**.

---

## Simplifying language modeling

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Sequence models as **random embedding generation and classification**

<div style="margin-top: -0.5em"></div>

**Assumption 4.** There is **only one** likely next token in the data

**Let's keep our life simple**, at least for now :)

---

## The toy model

**Assumption 1.** Language modeling as **context to next-token memories**

<div style="margin-top: -0.5em"></div>

**Assumption 2.** Contexts follow a **power law distribution**

<div style="margin-top: -0.5em"></div>

**Assumption 3.** Sequence models as **random embedding generation and classification**

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
**Interference.** The prediction for context $i$ is perturbed by the **other embeddings close to** $e_i$.

---

<!-- class: carry-line -->
## Some theoretical intuition

**Interference.** The prediction for context $i$ is perturbed by the **other embeddings close to** $e_i$.

<!-- on the left, I want to have some visual reminder of what the weights are W = sum_ tok(in color) \arrow{emb} -->

<div style="margin-top: -1em"></div>

<!-- step -->

<svg class="sphere-fig" viewBox="0 0 1180 420" role="img">
<defs>
<marker id="sf-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="sf-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="sf-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
<marker id="sf-head-orange" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#e67e22"/></marker>
<marker id="sf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<line class="sf-vec sf-legend sf-navy" x1="16" y1="150" x2="38" y2="150" marker-end="url(#sf-head-navy)"/>
<text x="56" y="157">embedding</text>
<text x="16" y="196"><tspan class="sf-t1">cor</tspan><tspan class="sf-t2">rect</tspan><tspan class="sf-t3">&#160;tok</tspan><tspan class="sf-t4">en</tspan></text>
</g>
<g>
<circle class="sf-sphere" cx="675" cy="175" r="105"/>
<line class="sf-vec sf-navy" x1="675" y1="175" x2="706.5" y2="88.5" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="675" y1="175" x2="752.2" y2="124.9" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="675" y1="175" x2="745.5" y2="234.1" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="675" y1="175" x2="767.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<text class="sf-muted sf-small" x="675" y="308" text-anchor="middle">unit sphere</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="675" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-bad">incorrect</tspan></text>
<text class="sf-small" x="746" y="200" text-anchor="middle">query</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="965" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="965" cy="175" r="105"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="987.3" y2="85.7" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1021.6" y2="102.5" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1011.0" y2="254.7" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="1057.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="706" y1="346" x2="958" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="835" y="382" text-anchor="middle">increasing <tspan class="sf-var">h</tspan> makes</text>
<text class="sf-small" x="835" y="408" text-anchor="middle">embeddings more orthogonal</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="385" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="385" cy="175" r="105"/>
<line class="sf-vec sf-green" x1="385" y1="175" x2="410.4" y2="86.6" marker-end="url(#sf-head-green)"/>
<line class="sf-vec sf-red" x1="385" y1="175" x2="463.0" y2="126.2" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-orange" x1="385" y1="175" x2="453.4" y2="236.6" marker-end="url(#sf-head-orange)"/>
<line class="sf-vec sf-navy" x1="385" y1="175" x2="477.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="644" y1="346" x2="392" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="515" y="382" text-anchor="middle">increasing <tspan class="sf-var">d</tspan> spreads</text>
<text class="sf-small" x="515" y="408" text-anchor="middle">noise over more tokens</text>
</g>
</svg>

<!-- step -->

<div style="margin-top: 1.5em"></div>

**As a result**, capacity increases with $d$ and $h$.

---

<!-- columns: 1/1 -->
## Capacity, in theory and in practice

<!-- Move all figures to the figures part, each of them in a separate HTML file. Center this figure vertically so that it looks nicer -->

<!-- BEGIN capacity-chart (generated by scripts/capacity_sweep.py) -->

<div class="cap-legend">
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="6 5"/></svg>one memory per parameter</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>trained</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#c0392b"/></svg>Hebbian</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "trained"
      color: "#0f3460"
      data:
        - {x: 8192, y: 1071.3}
        - {x: 16384, y: 2336.0}
        - {x: 32768, y: 5002.0}
        - {x: 65536, y: 10284.0}
        - {x: 131072, y: 20901.0}
        - {x: 262144, y: 41839.7}
        - {x: 524288, y: 84473.0}
    - label: "Hebbian"
      color: "#c0392b"
      data:
        - {x: 8192, y: 109.3}
        - {x: 16384, y: 134.0}
        - {x: 32768, y: 636.0}
        - {x: 65536, y: 1154.7}
        - {x: 131072, y: 2281.7}
        - {x: 262144, y: 5117.7}
        - {x: 524288, y: 10664.0}
    - label: "one memory per parameter"
      color: "#9ca3af"
      data:
        - {x: 8192, y: 8192}
        - {x: 16384, y: 16384}
        - {x: 32768, y: 32768}
        - {x: 65536, y: 65536}
        - {x: 131072, y: 131072}
        - {x: 262144, y: 262144}
        - {x: 524288, y: 524288}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "number of parameters N"}
      min: 5650
      max: 760218
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
    y:
      type: logarithmic
      title: {display: true, text: "capacity"}
      min: 75
      max: 760218
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/capacity-chart.js"></script>

<!-- END capacity-chart -->

|||

In theory, capacity is **proportional to the number of parameters**, up to logarithmic factors. We fix $d=256$ and vary $h$.

<!-- step -->

At large $h$ the trained model is **exactly parallel to $N$** (slope $1.01$) and stores<span class="highlight-navy">0.16 memories per parameter</span>.

<div class="cap-cue" data-cap-series="trained"></div>

<!-- step -->

Hebbian gets <span class="highlight-red">0.02</span>, an **order of magnitude** less.

<div class="cap-cue" data-cap-series="hebbian"></div>

<!-- step -->

**To learn more.** [@cabannes2024scaling] for the full analysis, [@nichani2025factual] for other architectures, [@zucchet2026ambiguity] for several possible next tokens.

---

## Capacity, in theory and in practice

In practice, capacity is also proportional to the number of parameters (total parameters in mixture of experts; they store more knowledge for the same amount of active parameters!).

<!-- Two columns, one with the Physics of LLMs 3.3. Fig 1 b), plot, take the data from the paper and stylize it in our way (y axis is "memory stored"). Right column is Morris et al. 2026 memorization https://arxiv.org/pdf/2505.24832 Figure 1, same thing, data in our style. Citation of teh two papers below, centered -->

---

## From capacity to scaling laws

Now that we have some idea of how the capacity scales as a function of the model size, **how do we get scaling laws?**

<!-- step -->

**For the model size:** assume infinite data, and that the first contexts below capacity get 0 loss and some value $l$ afterwards. 

<!-- there is more space for the subtext on the right, increase it. And again if colloquium has some built in way of dealing with such columns we should use it. Might be some feature to ask to add if not available -->

<!-- step -->

As a result:

<div class="eq-step">
<div>

$$L(N) = \sum_i p(i) \, L(i)$$

</div>
<div class="text-sm">

Average the per-context loss over **how often each context appears**.

</div>
</div>

<!-- step -->

<div class="eq-step">
<div>

$$\phantom{L(N)} \propto \sum_{i \leq \mathrm{capacity}(N)} i^{-\alpha} \times 0 \; + \sum_{i > \mathrm{capacity}(N)} i^{-\alpha} \times l$$

</div>
<div class="text-sm">

Contexts **below capacity are free**; each one above costs $l$.

</div>
</div>

<!-- step -->

<div class="eq-step">
<div>

$$\phantom{L(N)} \propto \mathrm{capacity}(N)^{1 - \alpha} \propto N^{1-\alpha}.$$

</div>
<div class="text-sm">

The tail sum is set by **where it starts**, and capacity grows like $N$.

</div>
</div>

---

## From capacity to scaling laws

**For the model size:** the loss is a step function of the context index, and the step moves right as $N$ grows.

<svg class="plot-fig" viewBox="0 0 1180 400" role="img">
<defs>
<marker id="pf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<text class="pf-muted pf-small" x="110" y="28" text-anchor="middle"><tspan class="pf-var">L</tspan>(<tspan class="pf-var">i</tspan>)</text>
<line class="pf-axis" x1="110" y1="300" x2="110" y2="48" marker-end="url(#pf-head-axis)"/>
<line class="pf-axis" x1="110" y1="300" x2="530" y2="300" marker-end="url(#pf-head-axis)"/>
<line class="pf-guide" x1="110" y1="120" x2="300" y2="120"/>
<text class="pf-var" x="92" y="127" text-anchor="end">l</text>
<text class="pf-muted pf-small" x="92" y="307" text-anchor="end">0</text>
<line class="pf-sep" x1="300" y1="300" x2="300" y2="60"/>
<path class="pf-curve" d="M 118 300 L 300 300 L 300 120 L 512 120"/>
<line class="pf-tick" x1="300" y1="300" x2="300" y2="312"/>
<text class="pf-red pf-small" x="300" y="338" text-anchor="middle">capacity(<tspan class="pf-var">N</tspan>)</text>
<text class="pf-muted pf-small" x="320" y="380" text-anchor="middle">context index <tspan class="pf-var">i</tspan></text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text class="pf-muted pf-small" x="700" y="28" text-anchor="middle"><tspan class="pf-var">L</tspan>(<tspan class="pf-var">N</tspan>)</text>
<line class="pf-axis" x1="700" y1="300" x2="700" y2="48" marker-end="url(#pf-head-axis)"/>
<line class="pf-axis" x1="700" y1="300" x2="1120" y2="300" marker-end="url(#pf-head-axis)"/>
<line class="pf-axis" x1="694" y1="90" x2="700" y2="90"/>
<line class="pf-axis" x1="694" y1="180" x2="700" y2="180"/>
<line class="pf-axis" x1="694" y1="270" x2="700" y2="270"/>
<text class="pf-muted pf-small" x="682" y="97" text-anchor="end">1</text>
<text class="pf-muted pf-small" x="682" y="187" text-anchor="end">0.1</text>
<text class="pf-muted pf-small" x="682" y="277" text-anchor="end">0.01</text>
<line class="pf-axis" x1="730" y1="300" x2="730" y2="307"/>
<line class="pf-axis" x1="850" y1="300" x2="850" y2="307"/>
<line class="pf-axis" x1="970" y1="300" x2="970" y2="307"/>
<line class="pf-axis" x1="1090" y1="300" x2="1090" y2="307"/>
<text class="pf-muted pf-small" x="730" y="338" text-anchor="middle">1k</text>
<text class="pf-muted pf-small" x="850" y="338" text-anchor="middle">10k</text>
<text class="pf-muted pf-small" x="970" y="338" text-anchor="middle">100k</text>
<text class="pf-muted pf-small" x="1090" y="338" text-anchor="middle">1m</text>
<path class="pf-curve" d="M 730 90 L 1090 270"/>
<text class="pf-navy pf-small" x="960" y="170" text-anchor="middle"><tspan class="pf-var">N</tspan><tspan dy="-9" font-size="0.72em">1&#8722;<tspan class="pf-var">&#945;</tspan></tspan></text>
<text class="pf-muted pf-small" x="910" y="380" text-anchor="middle">number of parameters <tspan class="pf-var">N</tspan></text>
</g>
</svg>

<div class="inline-footnote fragment" data-fragment-index="1">

Note that in this case there is no residual entropy and the loss will converge to $0$ under infinite compute and data.

</div>

---

## From capacity to scaling laws

**For the dataset size:** assume instead that the model gets right every context it has **seen at least once**.

<div style="margin-top: 0.6em"></div>

Context $i$ appears in the first $D$ tokens, in expectation, as soon as

<!-- there is more space for the subtext on the right, increase it. And again if colloquium has some built in way of dealing with such columns we should use it. Might be some feature to ask to add if not available -->

<div class="eq-step">
<div>

$$p(i) \, D \geq 1 \qquad \Longleftrightarrow \qquad i \leq D^{1/\alpha}.$$

</div>
<div class="text-sm">

Only the **first $D^{1/\alpha}$ contexts** are ever seen.

</div>
</div>

<!-- step -->

Using the same argument as before,

<div class="eq-step">
<div>

$$L(D) = \sum_{i > D^{1/\alpha}} p(i) \, l \; \propto \; \left(D^{1/\alpha}\right)^{1-\alpha} = D^{\frac{1}{\alpha} - 1}.$$

</div>
<div class="text-sm">

Same tail sum, cut at $D^{1/\alpha}$ **instead of the capacity**.

</div>
</div>


---

<!-- columns: 3/2 -->
## From capacity to scaling laws

We now have intuition for how the loss scales as a function of model size $N$ and data $D$.

<svg class="plot-fig" id="sl-twin-fig" viewBox="0 0 1040 520" role="img">
<defs>
<marker id="sl-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
<clipPath id="sl-box"><rect x="120" y="60" width="840" height="340"/></clipPath>
</defs>
<text class="pf-muted pf-small" x="120" y="42" text-anchor="middle"><tspan class="pf-var">L</tspan></text>
<line class="pf-axis" x1="120" y1="400" x2="120" y2="60" marker-end="url(#sl-head-axis)"/>
<line class="pf-axis" x1="114" y1="100" x2="120" y2="100"/>
<line class="pf-axis" x1="114" y1="220" x2="120" y2="220"/>
<line class="pf-axis" x1="114" y1="340" x2="120" y2="340"/>
<text class="pf-muted pf-small" x="104" y="107" text-anchor="end">1</text>
<text class="pf-muted pf-small" x="104" y="227" text-anchor="end">0.1</text>
<text class="pf-muted pf-small" x="104" y="347" text-anchor="end">0.01</text>
<line class="pf-axis" x1="120" y1="400" x2="960" y2="400" marker-end="url(#sl-head-axis)"/>
<line class="pf-axis" x1="170" y1="400" x2="170" y2="408"/>
<line class="pf-axis" x1="410" y1="400" x2="410" y2="408"/>
<line class="pf-axis" x1="650" y1="400" x2="650" y2="408"/>
<line class="pf-axis" x1="890" y1="400" x2="890" y2="408"/>
<text class="pf-muted pf-small" x="170" y="434" text-anchor="middle">1k</text>
<text class="pf-muted pf-small" x="410" y="434" text-anchor="middle">10k</text>
<text class="pf-muted pf-small" x="650" y="434" text-anchor="middle">100k</text>
<text class="pf-muted pf-small" x="890" y="434" text-anchor="middle">1M</text>
<g clip-path="url(#sl-box)">
<path class="pf-curve" id="sl-line-n" d="M 170 100 L 890 244"/>
<path class="pf-curve pf-curve-d" id="sl-line-d" d="M 170 100 L 890 203"/>
</g>
<text class="pf-navy pf-small" id="sl-label-n" x="902" y="238"><tspan class="pf-var">N</tspan><tspan dy="-9" font-size="0.72em">1&#8722;<tspan class="pf-var">&#945;</tspan></tspan></text>
<text class="pf-red pf-strong pf-small" id="sl-label-d" x="902" y="197"><tspan class="pf-var">D</tspan><tspan dy="-9" font-size="0.72em">1/<tspan class="pf-var">&#945;</tspan>&#8722;1</tspan></text>
<line class="pf-curve" x1="314" y1="478" x2="358" y2="478"/>
<text class="pf-navy pf-small" x="368" y="485">model size <tspan class="pf-var">N</tspan></text>
<line class="pf-curve pf-curve-d" x1="568" y1="478" x2="612" y2="478"/>
<text class="pf-red pf-strong pf-small" x="622" y="485">dataset size <tspan class="pf-var">D</tspan></text>
</svg>

<div class="alpha-control">
<span class="alpha-readout" id="sl-readout"></span>
<input id="sl-alpha" type="range" min="1.05" max="2" step="0.05" value="1.4">
</div>

<script src="assets/scaling-slider.js"></script>

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

<!-- The legend on the left is not centered. Also I would like to remove the L^* proto part of things, and move the C... on top of the line. In the plots make sure to add more space with the exponent, right now the - is overlapping the C-->

At fixed **compute budgets**, we sweep the **model size** and fit a parabola in $\log N$.

<!-- BEGIN isoflop-figure (generated by scripts/isoflop_slide.py) -->

<div class="if-row">
<div class="cap-legend">
<span class="num"><em>C</em> (&#215;10<sup>9</sup>)</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#86b6ef" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#86b6ef"/></svg>10</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#6da7ec" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#6da7ec"/></svg>40</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#5598e7" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#5598e7"/></svg>160</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#256abf" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#256abf"/></svg>640</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#184f95" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#184f95"/></svg>2.6k</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0d366b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0d366b"/></svg>10k</span>
</div>
<svg class="plot-fig" viewBox="0 0 1180 400" role="img" aria-label="Left: loss against model size for six compute budgets, each a U-shaped IsoFLOP profile with a fitted parabola and its optimum marked; a dashed line through the optima has slope minus alpha minus one. Right: the compute-optimal model size against compute, measured and fitted.">
<defs>
<clipPath id="if-a"><rect x="152" y="8" width="410" height="300"/></clipPath>
<clipPath id="if-b"><rect x="764" y="8" width="388" height="300"/></clipPath>
</defs>
<line class="pf-axis" x1="152" y1="308" x2="562" y2="308"/>
<line class="pf-axis" x1="152" y1="308" x2="152" y2="18"/>
<line class="pf-axis" x1="152" y1="308" x2="152" y2="316"/>
<text class="pf-muted pf-small" x="152" y="342" text-anchor="middle">10k</text>
<line class="pf-axis" x1="300" y1="308" x2="300" y2="316"/>
<text class="pf-muted pf-small" x="300" y="342" text-anchor="middle">100k</text>
<line class="pf-axis" x1="447" y1="308" x2="447" y2="316"/>
<text class="pf-muted pf-small" x="447" y="342" text-anchor="middle">1M</text>
<line class="pf-axis" x1="144" y1="273" x2="152" y2="273"/>
<text class="pf-muted pf-small" x="138" y="280" text-anchor="end">0.8</text>
<line class="pf-axis" x1="144" y1="199" x2="152" y2="199"/>
<text class="pf-muted pf-small" x="138" y="206" text-anchor="end">1</text>
<line class="pf-axis" x1="144" y1="138" x2="152" y2="138"/>
<text class="pf-muted pf-small" x="138" y="145" text-anchor="end">1.2</text>
<line class="pf-axis" x1="144" y1="64" x2="152" y2="64"/>
<text class="pf-muted pf-small" x="138" y="71" text-anchor="end">1.5</text>
<text class="pf-muted pf-small" x="357" y="380" text-anchor="middle">model size <tspan class="pf-var">N</tspan></text>
<text class="pf-muted pf-small" transform="rotate(-90 78 163)" x="78" y="163" text-anchor="middle">loss</text>
<circle cx="197.9" cy="48.3" r="3.6" fill="#86b6ef"/>
<circle cx="228.1" cy="63.5" r="3.6" fill="#86b6ef"/>
<circle cx="256.7" cy="69.4" r="3.6" fill="#86b6ef"/>
<circle cx="272.5" cy="69.3" r="3.6" fill="#86b6ef"/>
<circle cx="301.1" cy="63.1" r="3.6" fill="#86b6ef"/>
<circle cx="331.2" cy="49.3" r="3.6" fill="#86b6ef"/>
<circle cx="242.4" cy="95.0" r="3.6" fill="#6da7ec"/>
<circle cx="256.7" cy="102.4" r="3.6" fill="#6da7ec"/>
<circle cx="286.8" cy="111.6" r="3.6" fill="#6da7ec"/>
<circle cx="316.9" cy="112.7" r="3.6" fill="#6da7ec"/>
<circle cx="345.5" cy="107.1" r="3.6" fill="#6da7ec"/>
<circle cx="375.7" cy="95.1" r="3.6" fill="#6da7ec"/>
<circle cx="272.5" cy="131.9" r="3.6" fill="#5598e7"/>
<circle cx="301.1" cy="146.5" r="3.6" fill="#5598e7"/>
<circle cx="331.2" cy="154.2" r="3.6" fill="#5598e7"/>
<circle cx="361.3" cy="154.8" r="3.6" fill="#5598e7"/>
<circle cx="390.0" cy="148.6" r="3.6" fill="#5598e7"/>
<circle cx="420.1" cy="135.1" r="3.6" fill="#5598e7"/>
<circle cx="316.9" cy="176.8" r="3.6" fill="#256abf"/>
<circle cx="345.5" cy="190.2" r="3.6" fill="#256abf"/>
<circle cx="375.7" cy="196.8" r="3.6" fill="#256abf"/>
<circle cx="405.8" cy="195.9" r="3.6" fill="#256abf"/>
<circle cx="434.4" cy="189.2" r="3.6" fill="#256abf"/>
<circle cx="464.5" cy="175.4" r="3.6" fill="#256abf"/>
<circle cx="361.3" cy="221.6" r="3.6" fill="#184f95"/>
<circle cx="390.0" cy="233.4" r="3.6" fill="#184f95"/>
<circle cx="420.1" cy="239.5" r="3.6" fill="#184f95"/>
<circle cx="450.2" cy="236.8" r="3.6" fill="#184f95"/>
<circle cx="464.5" cy="233.2" r="3.6" fill="#184f95"/>
<circle cx="494.6" cy="221.6" r="3.6" fill="#184f95"/>
<circle cx="405.8" cy="267.1" r="3.6" fill="#0d366b"/>
<circle cx="434.4" cy="277.6" r="3.6" fill="#0d366b"/>
<circle cx="450.2" cy="280.6" r="3.6" fill="#0d366b"/>
<circle cx="478.8" cy="279.2" r="3.6" fill="#0d366b"/>
<circle cx="508.9" cy="272.9" r="3.6" fill="#0d366b"/>
<circle cx="539.1" cy="260.4" r="3.6" fill="#0d366b"/>
<g class="fragment" data-colloquium-fragment="1">
<g clip-path="url(#if-a)">
<path d="M 183.2 38.7 L 190.0 43.4 L 196.8 47.8 L 203.5 51.8 L 210.3 55.5 L 217.1 58.7 L 223.9 61.6 L 230.7 64.0 L 237.5 66.0 L 244.2 67.6 L 251.0 68.7 L 257.8 69.4 L 264.6 69.7 L 271.4 69.5 L 278.2 68.8 L 284.9 67.8 L 291.7 66.2 L 298.5 64.3 L 305.3 61.9 L 312.1 59.1 L 318.8 56.0 L 325.6 52.4 L 332.4 48.4 L 339.2 44.1 L 346.0 39.4" fill="none" stroke="#86b6ef" stroke-width="2.4" stroke-linecap="round"/>
<path d="M 227.6 86.9 L 234.4 91.0 L 241.2 94.7 L 248.0 98.2 L 254.7 101.3 L 261.5 104.0 L 268.3 106.5 L 275.1 108.5 L 281.9 110.2 L 288.7 111.5 L 295.4 112.4 L 302.2 113.0 L 309.0 113.1 L 315.8 112.9 L 322.6 112.3 L 329.4 111.3 L 336.1 110.0 L 342.9 108.3 L 349.7 106.2 L 356.5 103.7 L 363.3 100.9 L 370.1 97.7 L 376.8 94.2 L 383.6 90.4 L 390.4 86.3" fill="none" stroke="#6da7ec" stroke-width="2.4" stroke-linecap="round"/>
<path d="M 257.7 122.6 L 265.1 127.5 L 272.5 132.1 L 279.9 136.3 L 287.3 140.1 L 294.6 143.5 L 302.0 146.5 L 309.4 149.1 L 316.8 151.3 L 324.1 153.0 L 331.5 154.3 L 338.9 155.1 L 346.3 155.5 L 353.7 155.4 L 361.0 154.9 L 368.4 153.9 L 375.8 152.5 L 383.2 150.7 L 390.6 148.4 L 397.9 145.6 L 405.3 142.5 L 412.7 138.9 L 420.1 135.0 L 427.5 130.7 L 434.8 126.0" fill="none" stroke="#5598e7" stroke-width="2.4" stroke-linecap="round"/>
<path d="M 302.2 168.3 L 309.5 172.8 L 316.9 177.1 L 324.3 180.9 L 331.7 184.4 L 339.1 187.5 L 346.4 190.1 L 353.8 192.4 L 361.2 194.3 L 368.6 195.7 L 376.0 196.7 L 383.3 197.3 L 390.7 197.4 L 398.1 197.1 L 405.5 196.3 L 412.8 195.1 L 420.2 193.5 L 427.6 191.5 L 435.0 189.0 L 442.4 186.2 L 449.7 182.9 L 457.1 179.3 L 464.5 175.3 L 471.9 170.9 L 479.3 166.2" fill="none" stroke="#256abf" stroke-width="2.4" stroke-linecap="round"/>
<path d="M 346.6 213.4 L 353.4 217.4 L 360.2 221.1 L 366.9 224.5 L 373.7 227.5 L 380.5 230.2 L 387.3 232.6 L 394.1 234.6 L 400.9 236.2 L 407.6 237.5 L 414.4 238.4 L 421.2 239.0 L 428.0 239.2 L 434.8 238.9 L 441.6 238.4 L 448.3 237.4 L 455.1 236.1 L 461.9 234.4 L 468.7 232.3 L 475.5 229.9 L 482.3 227.2 L 489.0 224.1 L 495.8 220.7 L 502.6 217.0 L 509.4 212.9" fill="none" stroke="#184f95" stroke-width="2.4" stroke-linecap="round"/>
<path d="M 391.0 260.3 L 397.8 263.8 L 404.6 267.0 L 411.4 269.8 L 418.1 272.4 L 424.9 274.6 L 431.7 276.5 L 438.5 278.0 L 445.3 279.2 L 452.1 280.1 L 458.8 280.6 L 465.6 280.7 L 472.4 280.5 L 479.2 279.9 L 486.0 279.0 L 492.8 277.7 L 499.5 276.1 L 506.3 274.1 L 513.1 271.8 L 519.9 269.2 L 526.7 266.3 L 533.5 263.0 L 540.2 259.5 L 547.0 255.6 L 553.8 251.5" fill="none" stroke="#0d366b" stroke-width="2.4" stroke-linecap="round"/>
</g>
<path d="M 265.1 63.7 L 271.1 69.7 L 265.1 75.7 L 259.1 69.7 Z" fill="#86b6ef" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 308.5 107.1 L 314.5 113.1 L 308.5 119.1 L 302.5 113.1 Z" fill="#6da7ec" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 348.8 149.5 L 354.8 155.5 L 348.8 161.5 L 342.8 155.5 Z" fill="#5598e7" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 389.1 191.4 L 395.1 197.4 L 389.1 203.4 L 383.1 197.4 Z" fill="#256abf" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 427.6 233.2 L 433.6 239.2 L 427.6 245.2 L 421.6 239.2 Z" fill="#184f95" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 464.9 274.7 L 470.9 280.7 L 464.9 286.7 L 458.9 280.7 Z" fill="#0d366b" stroke="#fcfcfb" stroke-width="1.6"/>
</g>
<g class="fragment" data-colloquium-fragment="1">
<g clip-path="url(#if-a)">
<path class="pf-guide" d="M 244.0 45.7 L 367.4 176.0 L 490.9 306.3"/>
</g>
<text class="pf-muted pf-small" x="558" y="48" text-anchor="end"><tspan class="pf-var">L</tspan>* &#8733; <tspan class="pf-var">C</tspan><tspan dy="-9" font-size="0.72em">&#8722;0.091</tspan></text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<line class="pf-axis" x1="764" y1="308" x2="1152" y2="308"/>
<line class="pf-axis" x1="764" y1="308" x2="764" y2="18"/>
<line class="pf-axis" x1="796" y1="308" x2="796" y2="316"/>
<text class="pf-muted pf-small" x="796" y="342" text-anchor="middle">10</text>
<line class="pf-axis" x1="903" y1="308" x2="903" y2="316"/>
<text class="pf-muted pf-small" x="903" y="342" text-anchor="middle">100</text>
<line class="pf-axis" x1="1009" y1="308" x2="1009" y2="316"/>
<text class="pf-muted pf-small" x="1009" y="342" text-anchor="middle">1k</text>
<line class="pf-axis" x1="1116" y1="308" x2="1116" y2="316"/>
<text class="pf-muted pf-small" x="1116" y="342" text-anchor="middle">10k</text>
<line class="pf-axis" x1="756" y1="230" x2="764" y2="230"/>
<text class="pf-muted pf-small" x="750" y="237" text-anchor="end">100k</text>
<line class="pf-axis" x1="756" y1="80" x2="764" y2="80"/>
<text class="pf-muted pf-small" x="750" y="87" text-anchor="end">1M</text>
<text class="pf-muted pf-small" x="958" y="380" text-anchor="middle">compute <tspan class="pf-var">C</tspan> (&#215;10<tspan dy="-9" font-size="0.72em">9</tspan>)</text>
<text class="pf-muted pf-small" transform="rotate(-90 690 163)" x="690" y="163" text-anchor="middle">model size <tspan class="pf-var">N</tspan>*</text>
<g clip-path="url(#if-b)">
<path class="pf-guide" d="M 764.0 282.6 L 958.0 160.2 L 1152.0 37.8"/>
</g>
<path d="M 796.4 258.7 L 802.4 264.7 L 796.4 270.7 L 790.4 264.7 Z" fill="#86b6ef" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 860.5 214.7 L 866.5 220.7 L 860.5 226.7 L 854.5 220.7 Z" fill="#6da7ec" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 924.6 173.8 L 930.6 179.8 L 924.6 185.8 L 918.6 179.8 Z" fill="#5598e7" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 988.7 133.0 L 994.7 139.0 L 988.7 145.0 L 982.7 139.0 Z" fill="#256abf" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 1052.8 93.9 L 1058.8 99.9 L 1052.8 105.9 L 1046.8 99.9 Z" fill="#184f95" stroke="#fcfcfb" stroke-width="1.6"/>
<path d="M 1116.9 56.1 L 1122.9 62.1 L 1116.9 68.1 L 1110.9 62.1 Z" fill="#0d366b" stroke="#fcfcfb" stroke-width="1.6"/>
<text class="pf-muted pf-small" x="1037" y="90" text-anchor="end"><tspan class="pf-var">C</tspan><tspan dy="-9" font-size="0.72em">0.449</tspan></text>
</g>
</svg>
</div>

<!-- END isoflop-figure -->

<div class="text-sm">

Six IsoFLOP profiles. Learning rates are tuned for each run.

</div>

<!-- BEGIN isoflop-exponents (generated by scripts/isoflop_slide.py) -->

<div style="margin-top: 1.2em"></div>

The loss decreases as $C^{-0.091}$ (predicted $C^{-0.091}$), the model size grows as $C^{0.449}$ (vs. $C^{0.455}$).

<!-- END isoflop-exponents -->

---

## Are the toy model predictions good enough?

<div class="chin-premise">

We assumed the **Chinchilla scaling** and recovered similar compute exponents as in theory, but **how good is the full law?**

</div>

<div class="chin-equation-row">
<div>

$$L-L_\infty=A N^{-a}+B D^{-b}.$$

</div>
<div class="chin-independence">

$N$ and data $D$ treated as **independent**.

</div>
</div>

<div style="margin-top: 2.2em"></div>

<div class="chin-lower">
<div>

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

</div>
<div class="chin-story">

**Additive scaling laws do not extrapolate as well.**

Idea: **couple model size and data**.

$$L-L_\infty=\left(\frac{A}{N^a}+\frac{B}{D^b}\right)^k$$

**To learn more.** [@videau2026skaling] on why this functional form makes sense and how it better extrapolates.

</div>
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

<!-- BEGIN results-alpha (generated by scripts/grid_slide.py) -->
<div class="cap-legend">
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><circle cx="15" cy="5" r="3.2" fill="#0f3460"/></svg>model limited</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><circle cx="15" cy="5" r="3.2" fill="#c0392b"/></svg>data limited</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="6 5"/></svg>theory</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "measured, model axis"
      color: "#0f3460"
      data:
        - {x: 1.1, y: 0.1124}
        - {x: 1.2, y: 0.2116}
        - {x: 1.3, y: 0.2944}
        - {x: 1.5, y: 0.5077}
        - {x: 1.8, y: 0.8113}
    - label: "theory, model axis"
      color: "#0f3460"
      data:
        - {x: 1.05, y: 0.0500}
        - {x: 1.085, y: 0.0848}
        - {x: 1.12, y: 0.1196}
        - {x: 1.154, y: 0.1543}
        - {x: 1.189, y: 0.1891}
        - {x: 1.224, y: 0.2239}
        - {x: 1.259, y: 0.2587}
        - {x: 1.293, y: 0.2935}
        - {x: 1.328, y: 0.3283}
        - {x: 1.363, y: 0.3630}
        - {x: 1.398, y: 0.3978}
        - {x: 1.433, y: 0.4326}
        - {x: 1.467, y: 0.4674}
        - {x: 1.502, y: 0.5022}
        - {x: 1.537, y: 0.5370}
        - {x: 1.572, y: 0.5717}
        - {x: 1.607, y: 0.6065}
        - {x: 1.641, y: 0.6413}
        - {x: 1.676, y: 0.6761}
        - {x: 1.711, y: 0.7109}
        - {x: 1.746, y: 0.7457}
        - {x: 1.78, y: 0.7804}
        - {x: 1.815, y: 0.8152}
        - {x: 1.85, y: 0.8500}
    - label: "measured, data axis"
      color: "#c0392b"
      data:
        - {x: 1.1, y: 0.0873}
        - {x: 1.2, y: 0.1637}
        - {x: 1.3, y: 0.2284}
        - {x: 1.5, y: 0.3483}
        - {x: 1.8, y: 0.4907}
    - label: "theory, data axis"
      color: "#c0392b"
      data:
        - {x: 1.05, y: 0.0476}
        - {x: 1.085, y: 0.0782}
        - {x: 1.12, y: 0.1068}
        - {x: 1.154, y: 0.1337}
        - {x: 1.189, y: 0.1590}
        - {x: 1.224, y: 0.1829}
        - {x: 1.259, y: 0.2055}
        - {x: 1.293, y: 0.2269}
        - {x: 1.328, y: 0.2471}
        - {x: 1.363, y: 0.2663}
        - {x: 1.398, y: 0.2846}
        - {x: 1.433, y: 0.3020}
        - {x: 1.467, y: 0.3185}
        - {x: 1.502, y: 0.3343}
        - {x: 1.537, y: 0.3494}
        - {x: 1.572, y: 0.3638}
        - {x: 1.607, y: 0.3775}
        - {x: 1.641, y: 0.3907}
        - {x: 1.676, y: 0.4034}
        - {x: 1.711, y: 0.4155}
        - {x: 1.746, y: 0.4271}
        - {x: 1.78, y: 0.4383}
        - {x: 1.815, y: 0.4491}
        - {x: 1.85, y: 0.4595}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: linear
      title: {display: true, text: "tail exponent alpha"}
      min: 1.02
      max: 1.88
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
    y:
      type: linear
      title: {display: true, text: "loss exponent"}
      min: 0
      max: 0.92
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/results-chart.js"></script>

<!-- END results-alpha -->


---

<!-- columns: 1/1 -->
## What if data was finite?

Same experiment as before, but only keeping **first 10,000 contexts** (instead of infinitely many).

<!-- step -->

<!-- Can we animate the blue line in the plot at the same time as this step? -->

The loss **starts** decreasing as a **power law** and finishes as an **exponential** one.

<!-- step -->

This is the regime in which **deep learning used to be**! Comes with a bunch of overfitting problems which are much less common under infinite data. <!-- pls double check that statement -->

<!-- For the first 10k contexts, add 2 data points at larger scale. Remove the text below. Make sure the dots in the plot are full and not empty as currently is -->

|||

<div class="cap-legend">
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>first 10k contexts</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#c0392b"/></svg>infinite context pool</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "first 10k contexts"
      color: "#0f3460"
      data:
        - {x: 4096, y: 1.694101}
        - {x: 8192, y: 1.325302}
        - {x: 16384, y: 1.033968}
        - {x: 32768, y: 0.786548}
        - {x: 65536, y: 0.573029}
        - {x: 131072, y: 0.394475}
        - {x: 262144, y: 0.253310}
    - label: "infinite context pool"
      color: "#c0392b"
      data:
        - {x: 16384, y: 1.553531}
        - {x: 32768, y: 1.342995}
        - {x: 65536, y: 1.167684}
        - {x: 131072, y: 1.019898}
        - {x: 262144, y: 0.896523}
        - {x: 524288, y: 0.793940}
        - {x: 1048576, y: 0.715241}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "number of parameters N"}
      min: 3500
      max: 1200000
      grid: {drawOnChartArea: false}
      ticks: {padding: 8, maxTicksLimit: 5}
    y:
      type: logarithmic
      title: {display: true, text: "loss"}
      min: 0.2
      max: 1.8
      grid: {drawOnChartArea: false}
      ticks: {padding: 8, maxTicksLimit: 5}
```

<div class="inline-footnote">

$\alpha=1.2$; 6.55M online draws; exact frequency-weighted loss; best learning rate at every point.

</div>


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
