<!-- Building W, in three beats: the contexts, the random embedding each one gets,
     the next token each embedding is paired with.

     Geometry note.  The viewBox is 1180 wide *because* `sphere-fig` on the next
     slide is: both are full-width children of `.slide-content`, so one viewBox
     unit renders to the same number of pixels in both and x = 590 lands on the
     same screen pixel as that figure's middle sphere (cx = 590).  The arrows
     have to start there -- clicking forward must leave them where they are --
     which pins the embedding panel to the middle of the width and leaves only
     x < 486 for the contexts.  Hence the layout: contexts, embeddings, next
     token read left to right around that fixed centre, exactly as in
     `embed-fig`, rather than crowding all three into the left half.  Do not
     widen the viewBox to gain room: every type size here is in viewBox units
     and would shrink against the rest of the deck.
     The viewBox starts at y = -9, i.e. nine units of top padding, which is the
     vertical half of the same registration: it drops the arrow origin onto the
     screen row of that middle sphere (measured, not guessed).  It lives here
     rather than in the stylesheet because a positive margin-top on the <svg>
     collapses with the paragraph above it and does nothing. -->
<svg class="wbuild-fig" viewBox="0 -9 1180 314" role="img" aria-label="Building the Hebbian weights: four contexts, each encoded as a random unit vector on a circle, and each paired with its next token -- mat for the first two, floor for the last two -- the arrow keeping the colour of the token it predicts.">
<defs>
<marker id="wb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="wb-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="wb-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
</defs>
<!-- Four contexts, coloured by their next token: two mat, two floor. -->
<text class="wb-head" x="178" y="24" text-anchor="middle">contexts</text>
<rect class="wb-tag wb-fill-navy" x="14" y="40" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-navy" x="14" y="109" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-red" x="14" y="178" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-red" x="14" y="247" width="6" height="46" rx="3"/>
<rect class="wb-context" x="26" y="40" width="305" height="46" rx="8"/>
<rect class="wb-context" x="26" y="109" width="305" height="46" rx="8"/>
<rect class="wb-context" x="26" y="178" width="305" height="46" rx="8"/>
<rect class="wb-context" x="26" y="247" width="305" height="46" rx="8"/>
<text class="wb-code" x="46" y="68">the cat sat on the &#8230;</text>
<text class="wb-code" x="46" y="137">wipe your shoes on the &#8230;</text>
<text class="wb-code" x="46" y="206">the coin rolled on the &#8230;</text>
<text class="wb-code" x="46" y="275">she mopped the kitchen &#8230;</text>
<!-- One random unit vector per context, keeping its token's colour.  These are
     the *same four vectors*, at the same four angles, as the middle sphere of
     `sphere-fig` on the next slide -- copied from it, offset only by the 19 units
     between the two circles' centres (cy 148 there, 167 here).  That is the whole
     point of the pair: the audience sees one set of embeddings built here and then
     queried there, so any difference in the arrows would read as a different
     example.  If you move one, move the other. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="wb-map" x1="372" y1="167" x2="452" y2="167"/>
<text class="wb-muted" x="412" y="149" text-anchor="middle">encode</text>
<text class="wb-head" x="590" y="24" text-anchor="middle">embeddings</text>
<circle class="wb-circle" cx="590" cy="167" r="104"/>
<line class="wb-vec wb-navy" x1="590" y1="167" x2="621.1" y2="81.5" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-red" x1="590" y1="167" x2="666.3" y2="117.5" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-red" x1="590" y1="167" x2="659.7" y2="225.5" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-navy" x1="590" y1="167" x2="681.0" y2="167.0" marker-end="url(#wb-head-navy)"/>
</g>
<!-- The pairing (z_i, e_i) the Hebbian sum is made of: one arrow per row, so the
     correspondence is read off the row rather than from the colour alone. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="wb-head" x="850" y="24" text-anchor="middle">next token</text>
<line class="wb-map" x1="736" y1="63" x2="796" y2="63"/>
<line class="wb-map" x1="736" y1="132" x2="796" y2="132"/>
<line class="wb-map" x1="736" y1="201" x2="796" y2="201"/>
<line class="wb-map" x1="736" y1="270" x2="796" y2="270"/>
<text class="wb-tok wb-fill-navy" x="812" y="71">mat</text>
<text class="wb-tok wb-fill-navy" x="812" y="140">mat</text>
<text class="wb-tok wb-fill-red" x="812" y="209">floor</text>
<text class="wb-tok wb-fill-red" x="812" y="278">floor</text>
</g>
</svg>
