<svg class="wbuild-fig" viewBox="0 0 1180 362" role="img" aria-label="Building the Hebbian weights: four contexts, two whose next token is mat and two whose next token is floor, encoded as four unit vectors, then added up per next token so that each row of W is the sum of the embeddings of the contexts that predict its token.">
<defs>
<marker id="wb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="wb-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="wb-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
</defs>
<!-- Four contexts, coloured by their next token: two mat, two floor. -->
<text class="wb-head" x="142" y="30" text-anchor="middle">contexts</text>
<text class="wb-head" x="336" y="30" text-anchor="middle">next token</text>
<rect class="wb-tag wb-fill-navy" x="14" y="86" width="5" height="44" rx="2.5"/>
<rect class="wb-tag wb-fill-navy" x="14" y="156" width="5" height="44" rx="2.5"/>
<rect class="wb-tag wb-fill-red" x="14" y="226" width="5" height="44" rx="2.5"/>
<rect class="wb-tag wb-fill-red" x="14" y="296" width="5" height="44" rx="2.5"/>
<rect class="wb-context" x="26" y="86" width="232" height="44" rx="8"/>
<rect class="wb-context" x="26" y="156" width="232" height="44" rx="8"/>
<rect class="wb-context" x="26" y="226" width="232" height="44" rx="8"/>
<rect class="wb-context" x="26" y="296" width="232" height="44" rx="8"/>
<text class="wb-code" x="40" y="113">the cat sat on the &#8230;</text>
<text class="wb-code" x="40" y="183">wipe your shoes on the &#8230;</text>
<text class="wb-code" x="40" y="253">the coin rolled on the &#8230;</text>
<text class="wb-code" x="40" y="323">she mopped the kitchen &#8230;</text>
<line class="wb-map" x1="268" y1="108" x2="296" y2="108"/>
<line class="wb-map" x1="268" y1="178" x2="296" y2="178"/>
<line class="wb-map" x1="268" y1="248" x2="296" y2="248"/>
<line class="wb-map" x1="268" y1="318" x2="296" y2="318"/>
<text class="wb-tok wb-fill-navy" x="306" y="116">mat</text>
<text class="wb-tok wb-fill-navy" x="306" y="186">mat</text>
<text class="wb-tok wb-fill-red" x="306" y="256">floor</text>
<text class="wb-tok wb-fill-red" x="306" y="326">floor</text>
<!-- One random unit vector per context, keeping its token's colour. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="wb-map" x1="372" y1="213" x2="426" y2="213"/>
<text class="wb-muted" x="399" y="191" text-anchor="middle">encode</text>
<text class="wb-head" x="546" y="30" text-anchor="middle">embeddings</text>
<circle class="wb-circle" cx="546" cy="205" r="104"/>
<line class="wb-vec wb-navy" x1="546" y1="205" x2="549.2" y2="114.1" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-navy" x1="546" y1="205" x2="636.1" y2="192.3" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-red" x1="546" y1="205" x2="610.3" y2="140.7" marker-end="url(#wb-head-red)"/>
<line class="wb-vec wb-red" x1="546" y1="205" x2="588.7" y2="285.3" marker-end="url(#wb-head-red)"/>
<text class="wb-vector" x="550" y="71" text-anchor="middle">e₁</text>
<text class="wb-vector" x="684" y="193" text-anchor="middle">e₂</text>
<text class="wb-vector" x="645" y="109" text-anchor="middle">e₃</text>
<text class="wb-vector" x="608" y="329" text-anchor="middle">e₄</text>
<text class="wb-muted" x="546" y="343" text-anchor="middle">unit circle</text>
</g>
<!-- Add them up, one sum per next token: that sum is a row of W. -->
<g class="fragment" data-colloquium-fragment="1">
<line class="wb-map" x1="712" y1="213" x2="766" y2="213"/>
<text class="wb-muted" x="739" y="191" text-anchor="middle">sum</text>
<text class="wb-head" x="910" y="30" text-anchor="middle">rows of <tspan class="wb-var">W</tspan></text>
<circle class="wb-circle wb-circle-dash" cx="910" cy="225" r="104"/>
<polyline class="wb-ghost wb-navy" points="910,225 913.6,121.1 1016.6,106.6"/>
<polyline class="wb-ghost wb-red" points="910,225 983.5,151.5 1032.4,243.3"/>
<line class="wb-vec wb-navy" x1="910" y1="225" x2="1007.9" y2="116.3" marker-end="url(#wb-head-navy)"/>
<line class="wb-vec wb-red" x1="910" y1="225" x2="1019.5" y2="241.4" marker-end="url(#wb-head-red)"/>
<text class="wb-math wb-fill-navy" x="1020" y="106">W<tspan class="wb-sub" dy="8">mat</tspan></text>
<text class="wb-math wb-fill-red" x="1032" y="252">W<tspan class="wb-sub" dy="8">floor</tspan></text>
<text class="wb-muted" x="910" y="343" text-anchor="middle"><tspan class="wb-fill-navy">e₁ + e₂</tspan> and <tspan class="wb-fill-red">e₃ + e₄</tspan>, head to tail</text>
</g>
</svg>
