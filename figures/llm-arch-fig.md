<!-- What an LLM is, drawn bottom-up: a context of tokens, an alternating stack
     of sequence layers (mix across tokens) and feedforward layers (transform
     each token on its own) repeated L times, then a prediction head that turns
     the last token's representation into a distribution over next tokens.
     The four candidate tokens and their probabilities are the deck's running
     example, reused verbatim from the `tok-example` block on the
     "Simplifying language modeling" slide (model column: 0.31 / 0.24 / 0.19 /
     0.02); the target token `mat` is that slide's data column. -->
<svg class="llmarch-fig" viewBox="0 0 700 610" role="img" aria-label="A large language model drawn from the bottom up. At the bottom, the context 'the cat sat on the' as five token boxes. Above it, a block repeated L times containing a sequence layer, in which each token reads the tokens before it, and a feedforward layer, which transforms each token on its own. The last token's representation then goes through a prediction head and a softmax to give a probability distribution over next tokens: mat 0.31, couch 0.24, floor 0.19, roof 0.02. At training time the sentence supplies exactly one target token, mat.">
<defs>
<marker id="la-h-navy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 z" fill="#0f3460"/></marker>
<marker id="la-h-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 z" fill="#27ae60"/></marker>
<marker id="la-h-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 z" fill="#2980b9"/></marker>
<marker id="la-h-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 z" fill="#c0392b"/></marker>
</defs>
<!-- Top right: what comes out of the model. -->
<text class="la-head" x="356" y="26">distribution over next tokens</text>
<text class="la-gloss" x="356" y="46">a token is a word or a word piece</text>
<g class="la-cand"><text x="356" y="74">mat</text><text x="356" y="100">couch</text><text x="356" y="126">floor</text><text x="356" y="152">roof</text></g>
<g class="la-bar"><rect x="414" y="62" width="130" height="15" rx="2"/><rect x="414" y="88" width="101" height="15" rx="2"/><rect x="414" y="114" width="80" height="15" rx="2"/><rect x="414" y="140" width="8" height="15" rx="2"/></g>
<g class="la-prob" text-anchor="end"><text x="690" y="74">0.31</text><text x="690" y="100">0.24</text><text x="690" y="126">0.19</text><text x="690" y="152">0.02</text></g>
<!-- Top left: the one target token this sentence supplies. -->
<text class="la-tgt-head" x="200" y="40" text-anchor="middle">target token</text>
<rect class="la-tgt-box" x="158" y="53" width="84" height="36" rx="8"/>
<text class="la-tgt" x="200" y="78" text-anchor="middle">mat</text>
<text class="la-note" x="200" y="114" text-anchor="middle">one per context,</text>
<text class="la-note" x="200" y="134" text-anchor="middle">given by the data</text>
<text class="la-xent" x="298" y="62" text-anchor="middle">cross-entropy</text>
<path class="la-tgt-arrow" d="M250 71 H344" marker-end="url(#la-h-red)"/>
<!-- The prediction head, over the last token only. -->
<path class="la-flow-blue" d="M490 222 V176" marker-end="url(#la-h-blue)"/>
<text class="la-note" x="502" y="200">softmax</text>
<rect class="la-headbox" x="386" y="226" width="208" height="54" rx="10"/>
<text class="la-headlabel" x="490" y="260" text-anchor="middle">prediction head</text>
<path class="la-flow-navy" d="M490 326 V286" marker-end="url(#la-h-navy)"/>
<text class="la-note" x="470" y="300" text-anchor="end">last token&#8217;s representation</text>
<!-- The repeated block: five token lanes running through two kinds of layer. -->
<g class="la-lane"><path d="M90 320 V490"/><path d="M190 320 V490"/><path d="M290 320 V490"/><path d="M390 320 V490"/><path d="M490 320 V490"/></g>
<rect class="la-block" x="24" y="314" width="532" height="180" rx="12"/>
<path class="la-bracket" d="M568 320 h-10 v168 h10"/>
<text class="la-times" x="582" y="412">&#215; L</text>
<text class="la-ff-name" x="38" y="338">feedforward layer</text>
<text class="la-sub" x="212" y="338">transforms each token on its own</text>
<rect class="la-ff-band" x="38" y="346" width="504" height="44" rx="8"/>
<g class="la-ff-arrow">
<path d="M90 394 V364" marker-end="url(#la-h-green)"/>
<path d="M190 394 V364" marker-end="url(#la-h-green)"/>
<path d="M290 394 V364" marker-end="url(#la-h-green)"/>
<path d="M390 394 V364" marker-end="url(#la-h-green)"/>
<path d="M490 394 V364" marker-end="url(#la-h-green)"/>
</g>
<text class="la-sq-name" x="38" y="420">sequence layer</text>
<text class="la-sub" x="180" y="420">each token reads the tokens before it</text>
<rect class="la-sq-band" x="38" y="428" width="504" height="56" rx="8"/>
<path class="la-sq-spine" d="M490 478 V436"/>
<g class="la-sq-dot"><circle cx="90" cy="476" r="3.4"/><circle cx="190" cy="476" r="3.4"/><circle cx="290" cy="476" r="3.4"/><circle cx="390" cy="476" r="3.4"/></g>
<g class="la-sq-arrow">
<path d="M90 476 C 200 476, 300 440, 484 438" marker-end="url(#la-h-navy)"/>
<path d="M190 476 C 280 476, 360 452, 484 450" marker-end="url(#la-h-navy)"/>
<path d="M290 476 C 360 476, 410 462, 484 460" marker-end="url(#la-h-navy)"/>
<path d="M390 476 C 430 476, 450 468, 484 470" marker-end="url(#la-h-navy)"/>
</g>
<!-- The context at the bottom. -->
<g class="la-in-arrow">
<path d="M90 524 V500" marker-end="url(#la-h-navy)"/>
<path d="M190 524 V500" marker-end="url(#la-h-navy)"/>
<path d="M290 524 V500" marker-end="url(#la-h-navy)"/>
<path d="M390 524 V500" marker-end="url(#la-h-navy)"/>
<path d="M490 524 V500" marker-end="url(#la-h-navy)"/>
</g>
<g class="la-tokbox"><rect x="46" y="528" width="88" height="42" rx="8"/><rect x="146" y="528" width="88" height="42" rx="8"/><rect x="246" y="528" width="88" height="42" rx="8"/><rect x="346" y="528" width="88" height="42" rx="8"/><rect x="446" y="528" width="88" height="42" rx="8"/></g>
<g class="la-tok" text-anchor="middle"><text x="90" y="556">the</text><text x="190" y="556">cat</text><text x="290" y="556">sat</text><text x="390" y="556">on</text><text x="490" y="556">the</text></g>
<path class="la-ctx-brace" d="M46 576 v9 H534 v-9"/>
<text class="la-ctx" x="290" y="604" text-anchor="middle">context</text>
</svg>
