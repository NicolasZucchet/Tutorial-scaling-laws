<svg class="matmul-fig" viewBox="0 0 1180 440" role="img" aria-label="A four by three matrix times a three-element vector. The first output starts from zero and accumulates three products, requiring exactly three multiplications and three additions. Generalizing from one row to m rows gives exactly 2mn floating point operations.">
<!-- Concrete 4 by 3 matrix-vector product. -->
<text class="mm-label" x="226" y="22" text-anchor="middle">weights W · 4 × 3</text>
<path class="mm-bracket" d="M119 36 h-11 v190 h11 M333 36 h11 v190 h-11"/>
<g class="mm-grid">
<rect x="130" y="40" width="58" height="38" rx="6"/><rect x="197" y="40" width="58" height="38" rx="6"/><rect x="264" y="40" width="58" height="38" rx="6"/>
<rect x="130" y="87" width="58" height="38" rx="6"/><rect x="197" y="87" width="58" height="38" rx="6"/><rect x="264" y="87" width="58" height="38" rx="6"/>
<rect x="130" y="134" width="58" height="38" rx="6"/><rect x="197" y="134" width="58" height="38" rx="6"/><rect x="264" y="134" width="58" height="38" rx="6"/>
<rect x="130" y="181" width="58" height="38" rx="6"/><rect x="197" y="181" width="58" height="38" rx="6"/><rect x="264" y="181" width="58" height="38" rx="6"/>
</g>
<g class="mm-num" text-anchor="middle">
<text x="159" y="66">2</text><text x="226" y="66">1</text><text x="293" y="66">3</text>
<text x="159" y="113">0</text><text x="226" y="113">2</text><text x="293" y="113">1</text>
<text x="159" y="160">1</text><text x="226" y="160">1</text><text x="293" y="160">2</text>
<text x="159" y="207">2</text><text x="226" y="207">0</text><text x="293" y="207">1</text>
</g>
<text class="mm-op" x="371" y="138" text-anchor="middle">×</text>
<text class="mm-label" x="418" y="22" text-anchor="middle">input x · 3</text>
<path class="mm-bracket" d="M400 59 h-9 v143 h9 M436 59 h9 v143 h-9"/>
<g class="mm-grid">
<rect x="402" y="65" width="32" height="38" rx="6"/><rect x="402" y="112" width="32" height="38" rx="6"/><rect x="402" y="159" width="32" height="38" rx="6"/>
</g>
<g class="mm-num" text-anchor="middle"><text x="418" y="91">4</text><text x="418" y="138">2</text><text x="418" y="185">1</text></g>
<text class="mm-op" x="478" y="138" text-anchor="middle">=</text>
<text class="mm-label" x="530" y="22" text-anchor="middle">output y · 4</text>
<path class="mm-bracket" d="M510 36 h-9 v190 h9 M550 36 h9 v190 h-9"/>
<g class="mm-output" text-anchor="middle">
<text class="fragment" data-fragment-index="1" x="530" y="66">13</text>
<text class="fragment" data-fragment-index="3" x="530" y="113">5</text>
<text class="fragment" data-fragment-index="3" x="530" y="160">8</text>
<text class="fragment" data-fragment-index="3" x="530" y="207">9</text>
</g>
<path class="mm-divider" d="M595 25 V245"/>
<!-- Reveal 1: one row is one dot product, explicitly accumulated from zero. -->
<g class="fragment" data-colloquium-fragment="1">
<g class="mm-highlight-grid"><rect class="mm-pair-a" x="130" y="40" width="58" height="38" rx="6"/><rect class="mm-pair-b" x="197" y="40" width="58" height="38" rx="6"/><rect class="mm-pair-c" x="264" y="40" width="58" height="38" rx="6"/><rect class="mm-pair-a" x="402" y="65" width="32" height="38" rx="6"/><rect class="mm-pair-b" x="402" y="112" width="32" height="38" rx="6"/><rect class="mm-pair-c" x="402" y="159" width="32" height="38" rx="6"/></g>
<g class="mm-num" text-anchor="middle"><text x="159" y="66">2</text><text x="226" y="66">1</text><text x="293" y="66">3</text><text x="418" y="91">4</text><text x="418" y="138">2</text><text x="418" y="185">1</text></g>
<text class="mm-caption" x="885" y="33" text-anchor="middle">one row, one dot product</text>
<rect class="mm-chip mm-zero" x="626" y="62" width="44" height="44" rx="8"/>
<rect class="mm-chip mm-pair-a" x="704" y="62" width="96" height="44" rx="8"/>
<rect class="mm-chip mm-pair-b" x="834" y="62" width="96" height="44" rx="8"/>
<rect class="mm-chip mm-pair-c" x="964" y="62" width="96" height="44" rx="8"/>
<g class="mm-product" text-anchor="middle"><text x="648" y="90">0</text><text x="752" y="90">2 × 4</text><text x="882" y="90">1 × 2</text><text x="1012" y="90">3 × 1</text></g>
<g class="mm-plus" text-anchor="middle"><text x="686" y="91">+</text><text x="817" y="91">+</text><text x="947" y="91">+</text><text x="1078" y="91">=</text></g>
<text class="mm-result" x="1122" y="91" text-anchor="middle">13</text>
</g>
<!-- Reveal 2: first count the concrete operations, then state the row cost in n. -->
<g class="fragment" data-colloquium-fragment="1">
<text class="mm-concrete" x="885" y="145" text-anchor="middle">3 multiplications + 3 additions = 6 FLOPs</text>
<text class="mm-count" x="885" y="211" text-anchor="middle">N multiplications + N additions = 2N FLOPs</text>
</g>
<!-- Reveal 3: repeat the exact row cost m times. -->
<g class="fragment" data-colloquium-fragment="1">
<path class="mm-general-line" d="M90 270 H1090"/>
<text class="mm-general-note" x="590" y="301" text-anchor="middle">repeat the same dot product for every row</text>
<text class="mm-concrete-bottom" x="590" y="378" text-anchor="middle">this 4 × 3 example: 4 rows × 6 FLOPs per row = 24 FLOPs</text>
<text class="mm-law" x="590" y="430" text-anchor="middle">in general: 2N FLOPs</text>
</g>
</svg>
