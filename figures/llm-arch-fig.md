<!-- Pretraining in one line, read left to right: the context, a sequence model
     that turns it into an embedding, a prediction head that turns the embedding
     into a distribution over next tokens, and -- at the far right -- the token
     the data actually says came next.  The double-headed arrow between those
     last two is the whole training signal, so it carries the name of the loss:
     everything else on the slide exists to make that comparison possible.

     The chain is built in three beats, so the slide can be walked rather than
     read: the context alone, then the sequence model and the embedding, then the
     prediction head and the distribution, then the loss and the observed token.
     Each beat is a fragment group carrying colloquium's step marker, which it
     replaces with a sequential index, so the order is document order and the
     written value is never the step number.  (The marker attribute is not spelt
     out anywhere in this comment: colloquium rewrites it wherever it appears,
     comments included, and a mention inside one counts as a step of its own --
     an invisible extra click at the head of the slide.)  Everything, the four
     captions included, lives inside the <svg>, so a reveal cannot reflow the
     slide.

     The visual idiom is the `embed-fig` figure on the "Simplifying language
     modeling" slide, simplified: one context instead of three, one embedding
     vector on the circle instead of three, and token/bar rows with no
     probabilities on them.  The numbers are deliberately gone -- an architecture
     this schematic has no particular distribution to report, and printed values
     would only invite the audience to read them.  The phrase and the four
     candidates are the deck's running example, kept purely so the boxes have
     something legible in them.

     Colours are the roles the rest of the deck already assigns: navy for the
     model's internals, blue for what the model predicts, red for anything that
     comes from the data.

     Geometry: viewBox 1160 x 234.  1160 is the width `embed-fig` uses, and at a
     full-width column the deck lays an SVG out at roughly 1 unit = 1 px, so the
     font sizes in slides.css are close to real pixels.  Do not widen it to gain
     room: every type size here is in viewBox units and would shrink against the
     rest of the deck.

     Everything is centred on the chain axis y = 112.  The three arrow labels are
     set on *two* lines, baselines y = 44 and y = 68, which is what buys the
     horizontal air: "sequence model" on one line needed ~138 units, "sequence"
     over "model" needs ~78, and the same halving applies to the other two.  All
     four node captions share the baseline y = 216, a clear 32 units below the
     lowest thing above them (the last bar ends at y = 184), so they read as one
     row of labels with room around it rather than as text crowding the figure.

     The x layout is then one rule applied five times: 24 units of air between
     every node's edge and the arrow beside it, and every arrow 92 units long
     (they used to be 144, which is what made the row feel cramped).  Left to
     right: the context's rule ends at 250; arrow 274 -> 366; the circle spans
     390 -> 482 (r = 46); arrow 507 -> 599; the token column runs 623 -> 680 with
     its bars from 692; the widest bar stops at 842; the loss arrow runs
     866 -> 1016; and the observed token's box is 1040 -> 1132, centred at 1086 so
     its caption's right edge lands inside the viewBox.

     Arrowheads.  Every marker here is in stroke-width units, so its tip sits a
     fixed distance *past* the line's endpoint: 14.04 units for the navy
     embedding vector at stroke-width 2.6.  The vector is therefore drawn only
     46 - 14.04 = 31.96 units long, so that its tip -- not its endpoint -- lands
     on the circle.  That is the same construction `w-build-fig` and `sphere-fig`
     use, and it is why those arrows look right and this one used to overshoot. -->
<svg class="llmarch-fig" viewBox="0 0 1160 234" role="img" aria-label="Pretraining as a left-to-right chain. The underlined context 'the cat sat on the' goes through an arrow labelled sequence model into an embedding, drawn as one vector on a circle; an arrow labelled prediction head turns that embedding into a distribution over next tokens, drawn as four candidate tokens with bars of decreasing length: mat, couch, floor, roof. A red double-headed arrow labelled cross-entropy loss connects that distribution to the token actually observed in the data, mat.">
<defs>
<marker id="lp-head-grey" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0.6 L10,5 L0,9.4 z"/></marker>
<marker id="lp-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="lp-head-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0.6 L10,5 L0,9.4 z" fill="#c0392b"/></marker>
</defs>
<!-- 1. The context: a phrase, the rule under it, and the word it is called. -->
<text class="lp-ctx" x="135" y="120" text-anchor="middle">the cat sat on the</text>
<line class="lp-rule" x1="20" y1="137" x2="250" y2="137"/>
<text class="lp-cap" x="135" y="216" text-anchor="middle">context</text>
<!-- 2. The sequence model, and the embedding it produces: one vector on the
     circle, as in `embed-fig`, with its tip on the circle. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="lp-arrow" x1="274" y1="112" x2="366" y2="112" marker-end="url(#lp-head-grey)"/>
<text class="lp-step" x="320" y="44" text-anchor="middle">sequence</text>
<text class="lp-step" x="320" y="68" text-anchor="middle">model</text>
<circle class="lp-space" cx="436" cy="112" r="46"/>
<line class="lp-vec" x1="436" y1="112" x2="458.6" y2="89.4" marker-end="url(#lp-head-navy)"/>
<text class="lp-cap" x="436" y="216" text-anchor="middle">embedding</text>
</g>
<!-- 3. The prediction head, and the distribution it produces: four candidates,
     bars, no numbers. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="lp-arrow" x1="507" y1="112" x2="599" y2="112" marker-end="url(#lp-head-grey)"/>
<text class="lp-step" x="553" y="44" text-anchor="middle">prediction</text>
<text class="lp-step" x="553" y="68" text-anchor="middle">head</text>
<g class="lp-tok" text-anchor="end"><text x="680" y="59">mat</text><text x="680" y="99">couch</text><text x="680" y="139">floor</text><text x="680" y="179">roof</text></g>
<g class="lp-bar"><rect class="lp-bar-top" x="692" y="40" width="150" height="24" rx="3"/><rect x="692" y="80" width="108" height="24" rx="3"/><rect x="692" y="120" width="78" height="24" rx="3"/><rect x="692" y="160" width="34" height="24" rx="3"/></g>
<text class="lp-cap" x="732" y="216" text-anchor="middle">next-token distribution</text>
</g>
<!-- 4. The loss: what the data says came next, against what the model said. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="lp-xarrow" x1="866" y1="112" x2="1016" y2="112" marker-start="url(#lp-head-red)" marker-end="url(#lp-head-red)"/>
<text class="lp-xlab" x="941" y="44" text-anchor="middle">cross-entropy</text>
<text class="lp-xlab" x="941" y="68" text-anchor="middle">loss</text>
<rect class="lp-obs-box" x="1040" y="88" width="92" height="48" rx="9"/>
<text class="lp-obs" x="1086" y="121" text-anchor="middle">mat</text>
<text class="lp-cap" x="1086" y="216" text-anchor="middle">observed token</text>
</g>
</svg>
