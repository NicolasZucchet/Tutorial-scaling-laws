<svg class="wbuild-fig" viewBox="0 0 1180 305" role="img" aria-label="Building the Hebbian weights: four contexts, two whose next token is mat and two whose next token is floor, encoded as four random unit vectors on a circle, each arrow keeping the colour of the token it predicts.">
<defs>
<marker id="wb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="wb-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="wb-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
</defs>
<!-- Four contexts, coloured by their next token: two mat, two floor. -->
<text class="wb-head" x="171" y="24" text-anchor="middle">contexts</text>
<text class="wb-head" x="393" y="24" text-anchor="middle">next token</text>
<rect class="wb-tag wb-fill-navy" x="14" y="40" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-navy" x="14" y="109" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-red" x="14" y="178" width="6" height="46" rx="3"/>
<rect class="wb-tag wb-fill-red" x="14" y="247" width="6" height="46" rx="3"/>
<rect class="wb-context" x="26" y="40" width="290" height="46" rx="8"/>
<rect class="wb-context" x="26" y="109" width="290" height="46" rx="8"/>
<rect class="wb-context" x="26" y="178" width="290" height="46" rx="8"/>
<rect class="wb-context" x="26" y="247" width="290" height="46" rx="8"/>
<text class="wb-code" x="44" y="68">the cat sat on the &#8230;</text>
<text class="wb-code" x="44" y="137">wipe your shoes on the &#8230;</text>
<text class="wb-code" x="44" y="206">the coin rolled on the &#8230;</text>
<text class="wb-code" x="44" y="275">she mopped the kitchen &#8230;</text>
<line class="wb-map" x1="326" y1="63" x2="352" y2="63"/>
<line class="wb-map" x1="326" y1="132" x2="352" y2="132"/>
<line class="wb-map" x1="326" y1="201" x2="352" y2="201"/>
<line class="wb-map" x1="326" y1="270" x2="352" y2="270"/>
<text class="wb-tok wb-fill-navy" x="362" y="71">mat</text>
<text class="wb-tok wb-fill-navy" x="362" y="140">mat</text>
<text class="wb-tok wb-fill-red" x="362" y="209">floor</text>
<text class="wb-tok wb-fill-red" x="362" y="278">floor</text>
<!-- One random unit vector per context, keeping its token's colour. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="wb-map" x1="420" y1="167" x2="470" y2="167"/>
<text class="wb-muted" x="445" y="152" text-anchor="middle">encode</text>
<text class="wb-head" x="590" y="24" text-anchor="middle">embeddings</text>
<circle class="wb-circle" cx="590" cy="167" r="104"/>
<line class="wb-vec wb-navy" x1="590" y1="167" x2="593.2" y2="76.1" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-navy" x1="590" y1="167" x2="680.1" y2="154.3" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-red" x1="590" y1="167" x2="654.3" y2="102.7" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-red" x1="590" y1="167" x2="632.7" y2="247.3" marker-end="url(#wb-head-red)"/>
</g>
</svg>
