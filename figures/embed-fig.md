<svg class="embed-fig" viewBox="0 0 1160 380" role="img" aria-label="Three contexts encoded as three unit-norm embedding vectors, then classified into a next-token distribution">
<defs>
<marker id="ef-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="ef-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="ef-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="ef-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
</defs>
<text class="ef-head" x="160" y="35" text-anchor="middle">contexts</text>
<text class="ef-head" x="575" y="35" text-anchor="middle">embedding space</text>
<text class="ef-head" x="994" y="35" text-anchor="middle">next token</text>
<rect class="ef-tag ef-fill-navy" x="22" y="113" width="5" height="44" rx="2.5"/>
<rect class="ef-tag ef-fill-red" x="22" y="183" width="5" height="44" rx="2.5"/>
<rect class="ef-tag ef-fill-green" x="22" y="253" width="5" height="44" rx="2.5"/>
<rect class="ef-context" x="36" y="113" width="252" height="44" rx="8"/>
<rect class="ef-context" x="36" y="183" width="252" height="44" rx="8"/>
<rect class="ef-context" x="36" y="253" width="252" height="44" rx="8"/>
<text class="ef-code" x="52" y="140">the cat sat on the …</text>
<text class="ef-code" x="52" y="210">Paris is the capital of …</text>
<text class="ef-code" x="52" y="280">water freezes at …</text>
<line class="ef-arrow" x1="306" y1="205" x2="386" y2="205"/>
<text class="ef-muted" x="346" y="183" text-anchor="middle">encode</text>
<circle class="ef-space" cx="575" cy="205" r="118"/>
<line class="ef-vec ef-navy" x1="575" y1="205" x2="542.6" y2="105.1" marker-end="url(#ef-head-navy)"/>
<line class="ef-vec ef-red" x1="575" y1="205" x2="670.2" y2="160.6" marker-end="url(#ef-head-red)"/>
<line class="ef-vec ef-green" x1="575" y1="205" x2="630.6" y2="294.0" marker-end="url(#ef-head-green)"/>
<text class="ef-vector" x="520" y="80" text-anchor="middle">e₁</text>
<text class="ef-vector" x="706" y="148" text-anchor="middle">e₂</text>
<text class="ef-vector" x="662" y="328" text-anchor="middle">e₃</text>
<text class="ef-muted" x="575" y="353" text-anchor="middle">unit circle</text>
<line class="ef-arrow" x1="711" y1="205" x2="791" y2="205"/>
<text class="ef-muted" x="751" y="183" text-anchor="middle">classify</text>
<g>
<text class="ef-token" x="835" y="139">mat</text><rect class="ef-bar ef-fill-navy" x="900" y="122" width="183" height="22" rx="3"/><text class="ef-prob" x="1096" y="140">.52</text>
<text class="ef-token" x="835" y="183">couch</text><rect class="ef-bar ef-navy-light" x="900" y="166" width="81" height="22" rx="3"/><text class="ef-prob" x="994" y="184">.23</text>
<text class="ef-token" x="835" y="227">floor</text><rect class="ef-bar ef-navy-light" x="900" y="210" width="53" height="22" rx="3"/><text class="ef-prob" x="966" y="228">.15</text>
<text class="ef-token" x="835" y="271">roof</text><rect class="ef-bar ef-navy-light" x="900" y="254" width="25" height="22" rx="3"/><text class="ef-prob" x="938" y="272">.07</text>
</g>
</svg>
