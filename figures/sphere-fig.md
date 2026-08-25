<svg class="sphere-fig" viewBox="0 0 1180 446" role="img" aria-label="Three unit spheres of random context embeddings, one arrow per context coloured by its correct next token: in the first the query's own colour dominates, in the second two same-coloured neighbours outvote it, in the third the embeddings are close to orthogonal and the prediction is correct again.">
<defs>
<marker id="sf-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="sf-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="sf-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
<marker id="sf-head-orange" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#e67e22"/></marker>
<marker id="sf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<circle class="sf-sphere" cx="590" cy="180" r="128"/>
<line class="sf-vec sf-navy" x1="590" y1="180" x2="629.3" y2="71.9" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="590" y1="180" x2="686.4" y2="117.4" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="590" y1="180" x2="678.1" y2="253.9" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="590" y1="180" x2="705.0" y2="180.0" marker-end="url(#sf-head-navy)"/>
<text class="sf-muted sf-small" x="590" y="344" text-anchor="middle">unit sphere</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="590" y="30" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-bad">incorrect</tspan></text>
<text class="sf-small" x="684" y="208" text-anchor="middle">query</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="975" y="30" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="975" cy="180" r="128"/>
<line class="sf-vec sf-navy" x1="975" y1="180" x2="1002.8" y2="68.4" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="975" y1="180" x2="1045.8" y2="89.4" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="975" y1="180" x2="1032.5" y2="279.6" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="975" y1="180" x2="1090.0" y2="180.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="605" y1="374" x2="960" y2="374" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="782" y="406" text-anchor="middle">increasing <tspan class="sf-var">h</tspan> makes</text>
<text class="sf-small" x="782" y="432" text-anchor="middle">embeddings more orthogonal</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="205" y="30" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="205" cy="180" r="128"/>
<line class="sf-vec sf-green" x1="205" y1="180" x2="236.7" y2="69.5" marker-end="url(#sf-head-green)"/>
<line class="sf-vec sf-red" x1="205" y1="180" x2="302.5" y2="119.1" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-orange" x1="205" y1="180" x2="290.5" y2="257.0" marker-end="url(#sf-head-orange)"/>
<line class="sf-vec sf-navy" x1="205" y1="180" x2="320.0" y2="180.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="575" y1="374" x2="220" y2="374" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="397" y="406" text-anchor="middle">increasing <tspan class="sf-var">d</tspan> spreads</text>
<text class="sf-small" x="397" y="432" text-anchor="middle">noise over more tokens</text>
</g>
</svg>
