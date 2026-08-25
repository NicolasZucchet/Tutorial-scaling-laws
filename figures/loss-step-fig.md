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
<!-- Step 1: the area under the step curve.  Summing L(i) over the contexts the
     model has not stored *is* the loss of this one model, so shading it names
     the quantity the right-hand panel plots.  The rectangle is inset by ~1.5
     units on every side: it is painted after the curve, and a flush edge would
     lay a grey wash over half of the navy stroke. -->
<g class="fragment" data-colloquium-fragment="1">
<path class="pf-area" d="M 301.5 298.5 L 301.5 121.5 L 511 121.5 L 511 298.5 Z"/>
<text class="pf-navy pf-small" x="406" y="252" text-anchor="middle"><tspan class="pf-var">L</tspan>(<tspan class="pf-var">N</tspan>)</text>
</g>
<!-- Step 2: the same number, carried across as a single point.  The leader
     leaves the shaded block at its mid-height (y = 210, halfway between the l
     plateau at 120 and the axis at 300) and runs dead level to the dot, which
     sits at exactly that height on the L(N) line -- x = 970 is where
     y = 90 + (x - 730)/2 passes through 210 -- so the eye reads one value
     moving from left panel to right, not two unrelated marks. -->
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
<line class="pf-guide" x1="514" y1="210" x2="944" y2="210" marker-end="url(#pf-head-axis)"/>
<text class="pf-muted pf-small" x="910" y="380" text-anchor="middle">number of parameters <tspan class="pf-var">N</tspan></text>
</g>
<!-- Step 3: sweep N and the single point becomes the law. -->
<g class="fragment" data-colloquium-fragment="1">
<path class="pf-curve" d="M 730 90 L 1090 270"/>
<text class="pf-navy pf-small" x="880" y="140" text-anchor="middle"><tspan class="pf-var">N</tspan><tspan dy="-9" font-size="0.72em">1&#8722;<tspan class="pf-var">&#945;</tspan></tspan></text>
</g>
<!-- The dot belongs to step 2 with the leader, but SVG has no z-index: paint
     order is document order, so a dot written up there would be crossed out by
     the step-3 curve and lose its halo.  Written last it stays on top, and an
     explicit `data-fragment-index` (the same escape hatch the slide's footnote
     uses) puts it back in the reveal order it belongs to -- only the generated
     `data-colloquium-fragment` markers get numbered, so this does not shift the
     three steps above. -->
<circle class="pf-point fragment" data-fragment-index="2" cx="970" cy="210" r="7"/>
</svg>
