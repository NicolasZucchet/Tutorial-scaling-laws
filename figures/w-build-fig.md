<!-- Building W, in three beats, read left to right: the contexts, the next
     token each one is paired with, and the encoding -- arrow, circle and the
     random unit vector each context lands on, all on one click.  That order is the order of the Hebbian sum
     W = sum_i z_i e_i^T -- the pair (context, next token) is the memory, the
     embedding is only how the context is addressed -- so the next-token column
     now sits *before* the circle rather than after it.

     Geometry note.  The circle is the last panel, at cx = 880, which is where a
     104-radius circle lands once the contexts, the four row arrows and the
     encode arrow have taken the left two thirds.  It deliberately does not line
     up with any sphere in `sphere-fig` on the next slide: the middle sphere
     there is at cx = 590 and the right-hand one at cx = 975, and 975 is the
     "large h" case with its own, more orthogonal, arrows -- landing on it would
     read as the same picture when it is a different one.  What still carries
     across the two slides is the *angles*: these four vectors are the same four,
     at the same four angles, as that middle sphere, offset by the difference
     between the circles' centres (cx 590 / cy 148 there, cx 880 / cy 167 here).
     So the audience sees one set of embeddings built here and queried there --
     if you move an arrow here, move the matching one there.  Do not widen the
     viewBox to gain room: every type size here is in viewBox units and would
     shrink against the rest of the deck.
     The viewBox starts at y = -9, i.e. nine units of top padding, so the
     headings clear the paragraph above.  It lives here rather than in the
     stylesheet because a positive margin-top on the <svg> collapses with that
     paragraph and does nothing.

     Never leave a blank line between the `<svg` tag and `</svg>`, comments
     included.  A blank line ends the HTML block markdown opened at that tag, so
     everything after it is re-parsed -- these five-space-indented comment lines
     become an indented code block, the tag itself is swallowed into it, and the
     surviving elements are orphaned outside any <svg>.  The figure then renders
     as nothing and the next slide shows through it.  `--check` does not catch
     this; the lone `.` above is what keeps that paragraph break non-blank.

     The two revealed groups each carry colloquium's step marker, which it
     replaces with a sequential index, so the beat order is document order and
     no index is ever written by hand.  (The attribute is not spelt out anywhere
     in this comment, digits and all, because colloquium rewrites it wherever it
     appears, comments included, and `--check` would read a spelt-out one as a
     hand-written reveal.) -->
<svg class="wbuild-fig" viewBox="0 -9 1180 314" role="img" aria-label="Building the Hebbian weights: four contexts, each paired with its next token -- mat for the first two, floor for the last two -- and then encoded as a random unit vector on a circle, each vector keeping the colour of the token it predicts, with one vector labelled to show that the arrow is the embedding and the colour is the output.">
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
<!-- The pairing (context, z_i) the Hebbian sum is made of: one arrow per row, so
     the correspondence is read off the row rather than from the colour alone. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="wb-head" x="472" y="24" text-anchor="middle">next token</text>
<line class="wb-map" x1="360" y1="63" x2="430" y2="63"/>
<line class="wb-map" x1="360" y1="132" x2="430" y2="132"/>
<line class="wb-map" x1="360" y1="201" x2="430" y2="201"/>
<line class="wb-map" x1="360" y1="270" x2="430" y2="270"/>
<text class="wb-tok wb-fill-navy" x="444" y="71">mat</text>
<text class="wb-tok wb-fill-navy" x="444" y="140">mat</text>
<text class="wb-tok wb-fill-red" x="444" y="209">floor</text>
<text class="wb-tok wb-fill-red" x="444" y="278">floor</text>
</g>
<!-- The encoding, in one beat: the arrow, the circle and the four vectors all
     arrive together, so the click reads as a single act -- encode these
     contexts -- rather than as scenery assembling itself.
     The whole left-hand table is what gets encoded, so the arrow sits on the
     vertical middle of the four rows (y 167, the mean of the four row centres,
     which is also the circle's cy).  Horizontally it is centred in the gap it
     spans -- 83 units of air either side, between the widest token, `floor`,
     and the circle -- so the void does not pile up on one side of it.
     .
     The two labels on the horizontal navy vector are the figure's key, and they
     are colour-coded rather than spelt out as a legend box: grey -- the colour
     of every piece of annotation in this deck -- for the arrow itself, navy --
     the colour of the token it predicts -- for the output.  Both hang off the
     same vector on purpose: one arrow carries both facts.  They stand clear of
     that vector's arrowhead (x 1010 against a tip at 971) so the head reads as
     an arrow rather than as a bullet in front of the words. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="wb-map" x1="583" y1="167" x2="693" y2="167"/>
<text class="wb-muted" x="638" y="149" text-anchor="middle">encode</text>
<text class="wb-head" x="880" y="24" text-anchor="middle">embeddings</text>
<circle class="wb-circle" cx="880" cy="167" r="104"/>
<line class="wb-vec wb-navy" x1="880" y1="167" x2="911.1" y2="81.5" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-red" x1="880" y1="167" x2="956.3" y2="117.5" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-red" x1="880" y1="167" x2="949.7" y2="225.5" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-navy" x1="880" y1="167" x2="971.0" y2="167.0" marker-end="url(#wb-head-navy)"/>
<text class="wb-muted" x="1010" y="161">embedding</text>
<text class="wb-muted wb-fill-navy" x="1010" y="187">output</text>
</g>
</svg>
