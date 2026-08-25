<svg class="sphere-fig" viewBox="0 0 1180 420" role="img">
<defs>
<marker id="sf-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="sf-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="sf-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
<marker id="sf-head-orange" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#e67e22"/></marker>
<marker id="sf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<line class="sf-vec sf-legend sf-navy" x1="16" y1="150" x2="38" y2="150" marker-end="url(#sf-head-navy)"/>
<text x="56" y="157">embedding</text>
<text x="16" y="196"><tspan class="sf-t1">cor</tspan><tspan class="sf-t2">rect</tspan><tspan class="sf-t3">&#160;tok</tspan><tspan class="sf-t4">en</tspan></text>
</g>
<g>
<circle class="sf-sphere" cx="675" cy="175" r="105"/>
<line class="sf-vec sf-navy" x1="675" y1="175" x2="706.5" y2="88.5" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="675" y1="175" x2="752.2" y2="124.9" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="675" y1="175" x2="745.5" y2="234.1" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="675" y1="175" x2="767.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<text class="sf-muted sf-small" x="675" y="308" text-anchor="middle">unit sphere</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="675" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-bad">incorrect</tspan></text>
<text class="sf-small" x="746" y="200" text-anchor="middle">query</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="965" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="965" cy="175" r="105"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="987.3" y2="85.7" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1021.6" y2="102.5" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="965" y1="175" x2="1011.0" y2="254.7" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="965" y1="175" x2="1057.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="706" y1="346" x2="958" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="835" y="382" text-anchor="middle">increasing <tspan class="sf-var">h</tspan> makes</text>
<text class="sf-small" x="835" y="408" text-anchor="middle">embeddings more orthogonal</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="385" y="42" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="385" cy="175" r="105"/>
<line class="sf-vec sf-green" x1="385" y1="175" x2="410.4" y2="86.6" marker-end="url(#sf-head-green)"/>
<line class="sf-vec sf-red" x1="385" y1="175" x2="463.0" y2="126.2" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-orange" x1="385" y1="175" x2="453.4" y2="236.6" marker-end="url(#sf-head-orange)"/>
<line class="sf-vec sf-navy" x1="385" y1="175" x2="477.0" y2="175.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="644" y1="346" x2="392" y2="346" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="515" y="382" text-anchor="middle">increasing <tspan class="sf-var">d</tspan> spreads</text>
<text class="sf-small" x="515" y="408" text-anchor="middle">noise over more tokens</text>
</g>
</svg>
