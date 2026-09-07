<svg class="plot-fig hist-fig" viewBox="0 0 1180 215" role="img" aria-label="A timeline of scaling-law research in four stations. Before 2019, the loss was already known to sometimes fall as a power law in data and model size, at small scale. In 2020 the same laws are shown to hold at language-model scale, over seven decades of compute. In 2022 they become a rule for spending a compute budget, scaling parameters and data together: the Chinchilla rule. Since 2022 they have been extended in many directions -- mixture of experts, data limits, inference, recipes, training stages -- and are now how every frontier run is planned. Only the 2020 and 2022 stations are points on the axis; the outer two are periods.">
<defs>
<marker id="ht-head" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>

<!-- This figure's type scale lives here rather than in assets/slides.css, which
     is where every other figure's does.  The reason is not preference: the
     sizes below are what stop four columns of text from overlapping, and the
     stylesheet is a separately-cached file, so a browser holding an older copy
     of it renders this figure with `.plot-fig text`'s 22px and the columns
     collide -- which is exactly what happened twice while this slide was being
     drawn.  Shipping them inside the SVG means the geometry and the type that
     has to fit it travel together, in one file, in slides.html.  The colours
     still come from the deck's tokens. -->
<style>
.hist-fig .ht-year { font-size: 25px; font-weight: 700; }
.hist-fig .ht-desc { font-size: 17px; }
.hist-fig .ht-coin { font-weight: 700; }
.hist-fig .ht-focus { fill: var(--deck-navy); }
.hist-fig .ht-dot-focus { fill: var(--deck-navy); }
</style>

<!-- The axis is the one thing on screen before the first click: the reveals
     land stations on a line that is already there.

     Only 2020 and 2022 carry a dot.  "before 2019" and "since 2022" are
     periods rather than events, so there is no point on the axis to put one
     on.  Every station is otherwise coloured identically -- navy year, body
     in text colour, navy bold for a line that names a thing -- since the
     prose under the figure, not the ink, says which two Part I works through.

     Every station is three lines on the same three baselines, 122 / 146 / 170,
     which is what keeps four columns of text from ever meeting.  "since 2022"
     takes a fourth, at 196, for the second claim it carries: fitting its list
     onto three lines left only ~40 units between it and "2022", and the one
     thing this figure has taught repeatedly is not to bet on a font metric.
     Nothing sits to its right, so the extra line costs no alignment.  The class
     goes on each `<text>` rather than on a wrapping `<g>`: `.plot-fig text`
     declares font-size and fill on the elements themselves, so a value set on
     a parent is inherited by nothing, and a grouped station silently rendered
     at 22px in full text colour -- which is what the overlapping columns of
     the previous draft were.

     Widths are hand-checked at ~7.2 canvas units per character at 17px.  A
     station's half-width is the distance to its nearest neighbour's longest
     line: 26 characters at the three narrow stations, and 37 at "since 2022".
     Shorten a line before moving a station.

     The citations are deliberately NOT in here.  Colloquium only turns
     `[@key]` into a citation in markdown, so SVG text can only ever be a
     plain-text lookalike that never reaches the reference list; they live in
     the `.hist-cites` grid under the figure, whose column tracks are sized to
     put each group under its own station, and each column is hand-indexed to
     its station's own reveal so a paper arrives with the line it belongs
     to. -->
<line class="pf-axis" x1="40" y1="80" x2="1150" y2="80" marker-end="url(#ht-head)"/>

<!-- Reveal 1: the prehistory.  "sometimes" is load-bearing: before 2019 the
     power law was a finding in particular settings, not a rule anyone was
     claiming held generally. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="ht-year ht-focus" x="135" y="42" text-anchor="middle">before 2019</text>
<text class="ht-desc" x="135" y="122" text-anchor="middle">loss sometimes falls as</text>
<text class="ht-desc" x="135" y="146" text-anchor="middle">a power law in data</text>
<text class="ht-desc" x="135" y="170" text-anchor="middle">and model size</text>
</g>

<!-- Reveal 2: Kaplan.  Navy year, navy dot, text-coloured description. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="ht-year ht-focus" x="390" y="42" text-anchor="middle">2020</text>
<circle class="ht-dot-focus" cx="390" cy="80" r="6"/>
<text class="ht-desc" x="390" y="122" text-anchor="middle">the same laws hold</text>
<text class="ht-desc" x="390" y="146" text-anchor="middle">at LLM scale, over</text>
<text class="ht-desc" x="390" y="170" text-anchor="middle">seven decades of compute</text>
</g>

<!-- Reveal 3: Hoffmann.  The rule gets the third line, bold, so it does not
     read as another clause of the sentence above it. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="ht-year ht-focus" x="620" y="42" text-anchor="middle">2022</text>
<circle class="ht-dot-focus" cx="620" cy="80" r="6"/>
<text class="ht-desc" x="620" y="122" text-anchor="middle">how to spend a budget:</text>
<text class="ht-desc" x="620" y="146" text-anchor="middle">scale <tspan class="pf-var">N</tspan> and <tspan class="pf-var">D</tspan> together</text>
<text class="ht-desc ht-coin ht-focus" x="620" y="170" text-anchor="middle">the Chinchilla rule</text>
</g>

<!-- Reveal 4: the two right-hand stations merged, because neither is one
     result: "it kept going, and it is now how models get planned" is a single
     beat.  The widest station, since it is the only one carrying two claims. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="ht-year ht-focus" x="960" y="42" text-anchor="middle">since 2022</text>
<text class="ht-desc" x="960" y="122" text-anchor="middle">extended in many directions:</text>
<text class="ht-desc" x="960" y="146" text-anchor="middle">mixture of experts, data limits,</text>
<text class="ht-desc" x="960" y="170" text-anchor="middle">inference, recipes, stages</text>
<text class="ht-desc ht-coin ht-focus" x="960" y="196" text-anchor="middle">now how every frontier run is planned</text>
</g>
</svg>
