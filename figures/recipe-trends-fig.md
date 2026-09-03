<!-- Why one model size cannot decide between two recipes.  Three panels, loss
     against number of parameters, log-log, revealed left to right.

     SCHEMATIC.  Nothing here is measured: the losses, the exponents and the
     crossing point are all invented, chosen only to make the three readings
     as different as possible.  It is drawn deliberately as a cartoon -- the
     loss axis carries no numbers at all -- so it cannot be mistaken for the
     digitized figures elsewhere in the deck (kaplan-*-fig, chinchilla-*),
     which are real data and say where they came from.

     The three panels are the same experiment read three ways:
       left   -- the only thing a small budget buys: two runs at one size,
                 500M params, method B below method A.
       middle -- one possible continuation: B's curve is shallower, the two
                 laws meet at ~40B and A is ahead beyond that, so the
                 small-scale win does not survive.
       right  -- the other: B's curve is steeper, the gap widens with scale,
                 and B is the one worth investing in.
     The middle and right curves both pass exactly through the two left-hand
     points, which is the argument of the slide: identical evidence at one size,
     opposite conclusions at scale.  That is carried by the dots alone, redrawn
     at the same place on all three panels; there is no guide line and no size
     printed on it, because the panel titles already say what each reading is
     and a dashed rule through every panel only competed with the curves.

     Geometry, computed once and baked in (the axes are log, so a curve
     sketched by eye would not sit right next to the plotted figures in this
     deck; the mapping is the same construction as scripts/chinchilla_svg.py's
     `Log`/`Panel`):
       x: log N from 5e7 to 2e11 over 300 units; panels start at x = 96, 460,
          824, so the three share one scale and one tick set (100M, 1B, 10B,
          100B).  Curves are inset to 1e8 .. 1.5e11 so they stop short of the
          frame instead of dying in its corner.
       y: log L from 1.5 to 3.6 over 190 units, bottom y = 250, top y = 60.
          A third of a decade: narrow enough that the exponents below read as
          slopes rather than as a vertical smear.  The viewBox stops at 332,
          just under the x-axis titles: the legend used to sit below them and
          the PSA line below that, which pushed the PSA into the footer.  In an
          `.if-row` the legend costs width instead of height, and the width it
          costs shrinks the whole figure -- which is what buys the room back.
       A pure power law L = L0 (N/500M)^-a is a straight line on these axes,
          so each curve is two endpoints.  L0 = 3.00 (A) and 2.65 (B);
          exponents a = 0.070 / 0.042 (middle, they cross at 4.2e10) and
          0.052 / 0.088 (right, they never cross in range).

     Reveals: one step per panel, marker on the group so colloquium counts it
     (see scripts/_steps.py).  Step 1 also brings the shared y label and, via a
     hand-written index, the legend column, since the left panel's two dots need
     naming.  The slide's PSA line is a fourth step of its own, in slides.md. -->
<div class="if-row rt-row">
<!-- The key runs down the LEFT of the panel row, not under it: `.if-row` is the
     deck's own legend-beside-the-plot layout (figures/chinchilla-pareto.md,
     figures/isoflop-figure.md use it too), so this figure reads like the other
     multi-series plots in the talk.  It is HTML rather than SVG because the
     column has to be laid out against the plot, which only the flex row can do.
     It carries panel 1's step by hand (`data-fragment-index`, not a marker of
     its own -- a marker would count as an extra step and separate the key from
     the dots it names). -->
<div class="cap-legend fragment" data-fragment-index="1">
<span class="rt-key-a"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>method A</span>
<span class="rt-key-b"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#c0392b"/></svg>method B</span>
</div>
<svg class="plot-fig" viewBox="0 0 1180 332" role="img" aria-label="Schematic, three panels of loss against number of parameters on log-log axes, comparing two recipes called method A and method B. Left: a single model size, with one point per method and method B lower. Middle: full scaling laws through those same two points, method B shallower, so the two curves meet around 40 billion parameters and method A is better beyond. Right: full scaling laws through the same two points again, but method B is steeper, so its advantage grows with scale. Invented, illustrative numbers, not measurements.">

<!-- Panel 1: the two runs a small budget buys, at one size. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="rt-title" x="246" y="28" text-anchor="middle">one size: B looks better</text>
<line class="pf-axis" x1="96" y1="250" x2="396" y2="250"/>
<line class="pf-axis" x1="96" y1="250" x2="96" y2="60"/>
<line class="pf-axis" x1="121.1" y1="250" x2="121.1" y2="258"/>
<line class="pf-axis" x1="204.4" y1="250" x2="204.4" y2="258"/>
<line class="pf-axis" x1="287.6" y1="250" x2="287.6" y2="258"/>
<line class="pf-axis" x1="370.9" y1="250" x2="370.9" y2="258"/>
<text class="pf-muted pf-small" x="121.1" y="284" text-anchor="middle">100M</text>
<text class="pf-muted pf-small" x="204.4" y="284" text-anchor="middle">1B</text>
<text class="pf-muted pf-small" x="287.6" y="284" text-anchor="middle">10B</text>
<text class="pf-muted pf-small" x="370.9" y="284" text-anchor="middle">100B</text>
<text class="pf-muted pf-small" x="246" y="316" text-anchor="middle">model size <tspan class="pf-var">N</tspan></text>
<text class="pf-muted pf-small" transform="rotate(-90 60 155)" x="60" y="155" text-anchor="middle">loss (lower is better)</text>
<circle class="rt-dot rt-fill-a" cx="179.3" cy="99.6" r="5.5"/>
<circle class="rt-dot rt-fill-b" cx="179.3" cy="126.5" r="5.5"/>
</g>

<!-- Panel 2: continuation where B is shallower, so the laws meet at ~40B. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="rt-title" x="610" y="28" text-anchor="middle">&#8230; but the trends meet</text>
<line class="pf-axis" x1="460" y1="250" x2="760" y2="250"/>
<line class="pf-axis" x1="460" y1="250" x2="460" y2="60"/>
<line class="pf-axis" x1="485.1" y1="250" x2="485.1" y2="258"/>
<line class="pf-axis" x1="568.4" y1="250" x2="568.4" y2="258"/>
<line class="pf-axis" x1="651.6" y1="250" x2="651.6" y2="258"/>
<line class="pf-axis" x1="734.9" y1="250" x2="734.9" y2="258"/>
<text class="pf-muted pf-small" x="485.1" y="284" text-anchor="middle">100M</text>
<text class="pf-muted pf-small" x="568.4" y="284" text-anchor="middle">1B</text>
<text class="pf-muted pf-small" x="651.6" y="284" text-anchor="middle">10B</text>
<text class="pf-muted pf-small" x="734.9" y="284" text-anchor="middle">100B</text>
<text class="pf-muted pf-small" x="610" y="316" text-anchor="middle">model size <tspan class="pf-var">N</tspan></text>
<path class="rt-line rt-a" d="M 485.1 75.1 L 749.6 186.2"/>
<path class="rt-line rt-b" d="M 485.1 111.8 L 749.6 178.5"/>
<circle class="rt-dot rt-fill-a" cx="543.3" cy="99.6" r="5.5"/>
<circle class="rt-dot rt-fill-b" cx="543.3" cy="126.5" r="5.5"/>
</g>

<!-- Panel 3: continuation where B is steeper, so the gap keeps opening. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="rt-title" x="974" y="28" text-anchor="middle">&#8230; or the gap widens</text>
<line class="pf-axis" x1="824" y1="250" x2="1124" y2="250"/>
<line class="pf-axis" x1="824" y1="250" x2="824" y2="60"/>
<line class="pf-axis" x1="849.1" y1="250" x2="849.1" y2="258"/>
<line class="pf-axis" x1="932.4" y1="250" x2="932.4" y2="258"/>
<line class="pf-axis" x1="1015.6" y1="250" x2="1015.6" y2="258"/>
<line class="pf-axis" x1="1098.9" y1="250" x2="1098.9" y2="258"/>
<text class="pf-muted pf-small" x="849.1" y="284" text-anchor="middle">100M</text>
<text class="pf-muted pf-small" x="932.4" y="284" text-anchor="middle">1B</text>
<text class="pf-muted pf-small" x="1015.6" y="284" text-anchor="middle">10B</text>
<text class="pf-muted pf-small" x="1098.9" y="284" text-anchor="middle">100B</text>
<text class="pf-muted pf-small" x="974" y="316" text-anchor="middle">model size <tspan class="pf-var">N</tspan></text>
<path class="rt-line rt-a" d="M 849.1 81.4 L 1113.6 163.9"/>
<path class="rt-line rt-b" d="M 849.1 95.8 L 1113.6 235.4"/>
<circle class="rt-dot rt-fill-a" cx="907.3" cy="99.6" r="5.5"/>
<circle class="rt-dot rt-fill-b" cx="907.3" cy="126.5" r="5.5"/>
</g>
</svg>
</div>
