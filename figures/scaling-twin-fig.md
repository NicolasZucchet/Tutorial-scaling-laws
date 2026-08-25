<svg class="plot-fig" id="sl-twin-fig" viewBox="0 0 1040 520" role="img">
<defs>
<marker id="sl-head-axis" viewBox="0 0 7 7" refX="0.6" refY="3.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L7,3.5 L0,6.4 z" fill="#6b7280"/></marker>
<clipPath id="sl-box"><rect x="120" y="60" width="840" height="340"/></clipPath>
</defs>
<text class="pf-muted pf-small" x="120" y="42" text-anchor="middle"><tspan class="pf-var">L</tspan></text>
<line class="pf-axis" x1="120" y1="400" x2="120" y2="60" marker-end="url(#sl-head-axis)"/>
<line class="pf-axis" x1="114" y1="100" x2="120" y2="100"/>
<line class="pf-axis" x1="114" y1="220" x2="120" y2="220"/>
<line class="pf-axis" x1="114" y1="340" x2="120" y2="340"/>
<text class="pf-muted pf-small" x="104" y="107" text-anchor="end">1</text>
<text class="pf-muted pf-small" x="104" y="227" text-anchor="end">0.1</text>
<text class="pf-muted pf-small" x="104" y="347" text-anchor="end">0.01</text>
<line class="pf-axis" x1="120" y1="400" x2="960" y2="400" marker-end="url(#sl-head-axis)"/>
<line class="pf-axis" x1="170" y1="400" x2="170" y2="408"/>
<line class="pf-axis" x1="410" y1="400" x2="410" y2="408"/>
<line class="pf-axis" x1="650" y1="400" x2="650" y2="408"/>
<line class="pf-axis" x1="890" y1="400" x2="890" y2="408"/>
<text class="pf-muted pf-small" x="170" y="434" text-anchor="middle">1k</text>
<text class="pf-muted pf-small" x="410" y="434" text-anchor="middle">10k</text>
<text class="pf-muted pf-small" x="650" y="434" text-anchor="middle">100k</text>
<text class="pf-muted pf-small" x="890" y="434" text-anchor="middle">1M</text>
<g clip-path="url(#sl-box)">
<path class="pf-curve" id="sl-line-n" d="M 170 100 L 890 244"/>
<path class="pf-curve pf-curve-d" id="sl-line-d" d="M 170 100 L 890 203"/>
</g>
<text class="pf-navy pf-small" id="sl-label-n" x="902" y="238"><tspan class="pf-var">N</tspan><tspan dy="-9" font-size="0.72em">1&#8722;<tspan class="pf-var">&#945;</tspan></tspan></text>
<text class="pf-red pf-strong pf-small" id="sl-label-d" x="902" y="197"><tspan class="pf-var">D</tspan><tspan dy="-9" font-size="0.72em">1/<tspan class="pf-var">&#945;</tspan>&#8722;1</tspan></text>
<line class="pf-curve" x1="314" y1="478" x2="358" y2="478"/>
<text class="pf-navy pf-small" x="368" y="485">model size <tspan class="pf-var">N</tspan></text>
<line class="pf-curve pf-curve-d" x1="568" y1="478" x2="612" y2="478"/>
<text class="pf-red pf-strong pf-small" x="622" y="485">dataset size <tspan class="pf-var">D</tspan></text>
</svg>

<div class="alpha-control">
<span class="alpha-readout" id="sl-readout"></span>
<input id="sl-alpha" type="range" min="1.05" max="2" step="0.05" value="1.4">
</div>

<script src="assets/scaling-slider.js"></script>
