---
title: "Scaling law tutorial"
author: "Nicolas Zucchet"
date: "2026-08-18"
theme: default
aspect_ratio: "16:9"

fonts:
  heading: "Playfair Display"

footer:
  left: "Nicolas Zucchet"
  center: "Scaling law tutorial"
  right: "auto"

custom_css: |
  /* Local body font (not on Google Fonts): prefer the installed copy,
     fall back to the files in ./fonts so exports stay portable. */
  @font-face {
    font-family: "GothamNicolas";
    src: local("GothamNicolas"),
         url("fonts/GothamNicolas-Regular.otf") format("opentype");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
  }
  @font-face {
    font-family: "GothamNicolas";
    src: local("GothamNicolas Bold"), local("GothamNicolas-Bold"),
         url("fonts/GothamNicolas-Bold.otf") format("opentype");
    font-weight: 700;
    font-style: normal;
    font-display: swap;
  }
  :root {
    --colloquium-font-body: "GothamNicolas", "Helvetica Neue", Arial, sans-serif;
  }

  .highlight-red { color: #c0392b; font-weight: 600; }
  .highlight-blue { color: #2980b9; font-weight: 600; }
  .highlight-green { color: #27ae60; font-weight: 600; }

  /* Bold text picks up the theme's dark blue title. */
  .slide strong,
  .slide b {
    color: var(--colloquium-title);
  }

  /* "Simplifying language modeling": the model's explicit next-token
     distribution vs. the data's implicit one, each in its own accent.
     Extra specificity so bold inside these spans keeps the role colour
     instead of falling back to the navy `.slide strong` rule. */
  .slide .tok-model, .slide .tok-model strong, .slide strong.tok-model {
    color: #2980b9;
  }
  .slide .tok-data, .slide .tok-data strong, .slide strong.tok-data {
    color: #c0392b;
  }

  /* Centered, code-like example block: candidate next tokens in a single
     black column, one probability column per source (model, then data). */
  .tok-example {
    display: flex;
    align-items: flex-start;
    gap: 1.5em;
    width: fit-content;
    max-width: 100%;
    margin: 0.6em auto;
    padding: 0.8em 1.6em;
    background: var(--colloquium-code-bg);
    border-radius: 8px;
    font-family: var(--colloquium-font-mono);
    font-size: 0.78em;
    line-height: 1.55;
  }

  /* The context, underlined and labelled underneath. */
  /* Drop the context down by exactly one column-header line (0.8em text at
     1.55 line-height + its 0.45em bottom margin) so it sits on the first
     token row rather than in the middle of the stack. */
  .tok-ctx {
    text-align: center;
    margin-top: calc(0.8em * 1.55 + 0.8 * 0.45em);
  }
  .tok-ctx-group { display: inline-block; }
  .tok-ctx-text {
    display: block;
    padding-bottom: 0.2em;
    border-bottom: 2px solid var(--colloquium-muted);
  }
  .tok-ctx-label {
    display: block;
    margin-top: 0.2em;
    font-size: 0.8em;
    color: var(--colloquium-muted);
  }

  /* Token / probability columns. */
  .tok-cols { display: flex; justify-content: center; gap: 2.4em; }
  .tok-col-head {
    display: block;
    margin-bottom: 0.45em;
    font-size: 0.8em;
    letter-spacing: 0.04em;
  }
  .tok-tokens { text-align: left; }
  .tok-model, .tok-data { text-align: right; min-width: 3.6em; }

  /* Extra half line of air under the title-slide headline
     (theme default is 0.3em at font-size 2.8em). */
  .slide--title h1 {
    margin-bottom: calc(0.6em + 18px);
  }

  /* Remove space between paragraph and bullet points */
  .slide p + ul {
    margin-top: -0.5em;   /* 0.8em gap → ~0.4em */
  }

  /* Increase space after slide titles */
  .slide h2 {
    margin-bottom: 1em;
  }

  /* "Interference" figure: three unit spheres of random embeddings.  One
     arrow = one context embedding, coloured by its correct next token, so
     "several nearby arrows sharing a colour" is exactly the failure mode. */
  .sphere-fig {
    display: block;
    width: 100%;
    height: auto;
    margin: 0.4em auto 0;
    font-family: var(--colloquium-font-body);
  }
  .sphere-fig .sf-sphere {
    fill: none;
    stroke: var(--colloquium-muted);
    stroke-width: 1.5;
  }
  .sphere-fig .sf-vec {
    stroke-width: 2.6;
    stroke-linecap: round;
  }
  .sphere-fig .sf-axis {
    stroke: var(--colloquium-muted);
    stroke-width: 1.6;
  }
  .sphere-fig .sf-navy   { stroke: #0f3460; }
  .sphere-fig .sf-red    { stroke: #c0392b; }
  .sphere-fig .sf-green  { stroke: #27ae60; }
  .sphere-fig .sf-orange { stroke: #e67e22; }
  .sphere-fig text {
    fill: var(--colloquium-text);
    font-size: 18px;
  }
  .sphere-fig .sf-muted { fill: var(--colloquium-muted); }
  .sphere-fig .sf-strong { font-weight: 700; }
  .sphere-fig .sf-small { font-size: 16px; }
  /* Legend: the word itself carries the colour code, one hue per character. */
  .sphere-fig .sf-t1 { fill: #0f3460; }
  .sphere-fig .sf-t2 { fill: #c0392b; }
  .sphere-fig .sf-t3 { fill: #27ae60; }
  .sphere-fig .sf-t4 { fill: #e67e22; }
  /* Math-ish italics for the scalars d and h inside the captions. */
  .sphere-fig .sf-var {
    font-family: "Latin Modern Math", "Cambria Math", Georgia, serif;
    font-style: italic;
    font-size: 1.05em;
  }

---

# Scaling laws in large language models and toy models

Nicolas Zucchet -- [nzucchet@stanford.edu](mailto:nzucchet@stanford.edu)

September 8th 2026

---

## Agenda

### Part I: scaling laws in practice
- Some **history** and why scaling laws have been (and are) **central to the development of LLMs**
- **What scaling laws are** and what do they tell us
- How scaling laws guide **model and learning recipe** development

### Part II: understanding where scaling laws come from with toy models
- Scaling laws arise in a **toy associative memory model**
- Training our **own model** using scaling laws
- What did we **learn** from the toy model?

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

<div style="margin-top: 1em"></div>

There are many **open questions** though:
- **why power laws** (and not, e.g., exponential laws)?
- where do the **exponents come from**? how do they depend on the data?

<div style="margin-top: 1em"></div>

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

**Assumption 2.** Sequence model as random embedding generation

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

<div style="margin-top: 1.5em"></div>

We can get some intuition of the storage capacity with the following **Hebbian model**
$$W = \sum_i z_i e_i^\top \quad \text{with } z_{ij} = 1 \text{ if } j = y^*_i \text{ and } 0 \text{ otherwise}.$$

Corresponds to taking **one gradient descent step**, starting from $W=0$, on the objective $$\frac{1}{2}\lVert W e_i - z_i \rVert^2.$$ 

Contexts with embeddings close to $e_i$ will be **pushed towards outputting** $z_i$ (and thus to predict the correct next-token).

---

## Some theoretical intuition

**Interference.** The prediction for context $i$ is driven by the **other embeddings close to** $e_i$.

<svg class="sphere-fig" viewBox="0 0 1180 418" role="img" aria-label="Three unit spheres of random context embeddings, showing how increasing the number of tokens d or the embedding dimension h removes the interference that causes a wrong prediction.">
<defs>
<marker id="sf-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="sf-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="sf-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
<marker id="sf-head-orange" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#e67e22"/></marker>
<marker id="sf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<line class="sf-vec sf-navy" x1="16" y1="150" x2="58" y2="150" marker-end="url(#sf-head-navy)"/>
<text x="76" y="156">embedding</text>
<text x="16" y="196"><tspan class="sf-t1">c</tspan><tspan class="sf-t2">o</tspan><tspan class="sf-t3">r</tspan><tspan class="sf-t4">r</tspan><tspan class="sf-t1">e</tspan><tspan class="sf-t2">c</tspan><tspan class="sf-t3">t</tspan><tspan class="sf-t4">&#160;</tspan><tspan class="sf-t2">t</tspan><tspan class="sf-t3">o</tspan><tspan class="sf-t4">k</tspan><tspan class="sf-t1">e</tspan><tspan class="sf-t2">n</tspan></text>
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
<text x="675" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-strong">incorrect</tspan></text>
<text x="748" y="199" text-anchor="middle">query</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="965" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-strong">correct</tspan></text>
<circle class="sf-sphere" cx="965" cy="175" r="105"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="987.3" y2="85.7" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1021.6" y2="102.5" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1011.0" y2="254.7" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="1057.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="700" y1="346" x2="948" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="824" y="378" text-anchor="middle">increasing <tspan class="sf-var">h</tspan> makes</text>
<text class="sf-small" x="824" y="400" text-anchor="middle">embeddings more orthogonal</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="385" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-strong">correct</tspan></text>
<circle class="sf-sphere" cx="385" cy="175" r="105"/>
<line class="sf-vec sf-green" x1="385" y1="175" x2="410.4" y2="86.6" marker-end="url(#sf-head-green)"/>
<line class="sf-vec sf-red" x1="385" y1="175" x2="463.0" y2="126.2" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-orange" x1="385" y1="175" x2="453.4" y2="236.6" marker-end="url(#sf-head-orange)"/>
<line class="sf-vec sf-navy" x1="385" y1="175" x2="477.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="650" y1="346" x2="402" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="526" y="378" text-anchor="middle">increasing <tspan class="sf-var">d</tspan> spreads the noise</text>
<text class="sf-small" x="526" y="400" text-anchor="middle">over more tokens</text>
</g>
</svg>

---

% > Don't forget to cite Bietti and curse of ambiguity

---

## Results


---

## Test


---

## Conclusion

% > what are the open questions in the space? both empirically and theoretically



