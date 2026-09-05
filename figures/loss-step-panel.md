<!-- The step-loss panel: L(i) against the context index, with the step at
     capacity(N).  Shared, not copied: `loss-step-fig` draws it and then builds
     the shaded block and the carry-over to L(N) on top of it, and
     `loss-step-left-fig` draws it alone, for the slide that only states the
     assumption.  Both include it with a `figure:` comment, which the build
     expands recursively (scripts/build_slides.py), so this geometry has one
     home and the two slides show the same plot at the same scale because it is
     literally the same markup.
     NOT ONE BLANK LINE IN THIS COMMENT, and none anywhere else in this file.
     The including figures splice it inside their `<svg>` element, and an `<svg>`
     is a markdown raw-HTML block that ends at the first blank line -- after
     which these indented lines become an indented code block, and a <pre> lands
     in the middle of the figure with the panel gone.  That is invisible in the
     source and in the step and div guards; it shows up only on the slide.  The
     comments in `loss-step-fig` are spliced the same way and follow the same
     rule.
     The <marker> travels with it, which means both including figures carry a
     `pf-head-axis` def and the id appears twice in the built deck.  That is the
     one place the deck's one-id-per-figure rule is relaxed: the two defs are
     byte-identical, SVG resolves `url(#pf-head-axis)` to the first of them, and
     the arrowheads are the same arrowhead either way.  Giving the wrappers
     separate ids instead would mean the axis lines -- which live here -- could
     not name their own marker. -->
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
