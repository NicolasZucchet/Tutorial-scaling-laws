<!-- Three unit spheres.  The middle one is the query case, and its four arrows
     are shared with `w-build-fig` on the previous slide: same angles, same
     colours, offset only by the 19 units between the two circles' centres
     (cy 148 here, 167 there).  The two slides are one example seen twice --
     embeddings built, then queried -- so if you move an arrow here, move the
     matching one there. -->
<svg class="sphere-fig" viewBox="0 0 1180 356" role="img" aria-label="Three unit spheres of random context embeddings, one arrow per context coloured by its correct next token: in the first the query's own colour dominates, in the second two same-coloured neighbours outvote it, in the third the embeddings are close to orthogonal and the prediction is correct again.">
<defs>
<marker id="sf-head-navy" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#0f3460"/></marker>
<marker id="sf-head-red" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#c0392b"/></marker>
<marker id="sf-head-green" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#27ae60"/></marker>
<marker id="sf-head-orange" viewBox="0 0 6 6" refX="0.6" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.5 L6,3 L0,5.5 z" fill="#e67e22"/></marker>
<marker id="sf-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
</defs>
<g>
<circle class="sf-sphere" cx="590" cy="148" r="104"/>
<line class="sf-vec sf-navy" x1="590" y1="148" x2="621.1" y2="62.5" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="590" y1="148" x2="666.3" y2="98.5" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="590" y1="148" x2="659.7" y2="206.5" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="590" y1="148" x2="681.0" y2="148.0" marker-end="url(#sf-head-navy)"/>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="590" y="28" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-bad">incorrect</tspan></text>
<text class="sf-small" x="656" y="175" text-anchor="middle">query</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="975" y="28" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="975" cy="148" r="104"/>
<line class="sf-vec sf-navy" x1="975" y1="148" x2="997.0" y2="59.7" marker-end="url(#sf-head-navy)"/>
<line class="sf-vec sf-red" x1="975" y1="148" x2="1031.0" y2="76.3" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-red" x1="975" y1="148" x2="1020.5" y2="226.8" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-navy" x1="975" y1="148" x2="1066.0" y2="148.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="605" y1="288" x2="960" y2="288" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="782" y="318" text-anchor="middle">increasing <tspan class="sf-var">h</tspan> makes</text>
<text class="sf-small" x="782" y="344" text-anchor="middle">embeddings more orthogonal</text>
</g>
<g class="fragment" data-colloquium-fragment="1">
<text x="205" y="28" text-anchor="middle"><tspan class="sf-muted">prediction: </tspan><tspan class="sf-good">correct</tspan></text>
<circle class="sf-sphere" cx="205" cy="148" r="104"/>
<line class="sf-vec sf-green" x1="205" y1="148" x2="230.1" y2="60.6" marker-end="url(#sf-head-green)"/>
<line class="sf-vec sf-red" x1="205" y1="148" x2="282.2" y2="99.8" marker-end="url(#sf-head-red)"/>
<line class="sf-vec sf-orange" x1="205" y1="148" x2="272.7" y2="208.9" marker-end="url(#sf-head-orange)"/>
<line class="sf-vec sf-navy" x1="205" y1="148" x2="296.0" y2="148.0" marker-end="url(#sf-head-navy)"/>
<line class="sf-axis" x1="575" y1="288" x2="220" y2="288" marker-end="url(#sf-head-axis)"/>
<text class="sf-small" x="397" y="318" text-anchor="middle">increasing <tspan class="sf-var">d</tspan> spreads</text>
<text class="sf-small" x="397" y="344" text-anchor="middle">noise over more tokens</text>
</g>
</svg>
