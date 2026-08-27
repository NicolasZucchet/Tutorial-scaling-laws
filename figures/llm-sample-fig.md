<!-- Inference: four rounds of sampling, one per click.  Each round draws one
     token from the model's next-token distribution and appends it to the
     context, so the newly drawn token shows up in navy in the next round's
     context.  Round 1 reuses the deck's running distribution (0.31 / 0.24 /
     0.19, plus the remaining mass); the later rounds are illustrative, and
     round 3 deliberately draws the second-most-likely token to make the point
     that this is a sample, not an argmax. -->
<div class="llm-sample" role="img" aria-label="Four rounds of sampling. Starting from the context 'the cat sat on the', each round draws one token from the model's next-token distribution and appends it to the context: mat, then and, then purred, then softly, giving 'the cat sat on the mat and purred softly'.">
<div class="ls-row fragment" data-colloquium-fragment="1">
<div class="ls-ctx">the cat sat on the</div>
<div class="ls-line"><span class="ls-plab">p(next token)</span><span class="ls-bar"><span class="ls-seg ls-pick" style="width:31%">mat</span><span class="ls-seg" style="width:24%">couch</span><span class="ls-seg" style="width:19%">floor</span><span class="ls-seg ls-rest" style="width:26%"></span></span><span class="ls-draw">mat</span></div>
</div>
<div class="ls-row fragment" data-colloquium-fragment="1">
<div class="ls-ctx">the cat sat on the <b>mat</b></div>
<div class="ls-line"><span class="ls-plab"></span><span class="ls-bar"><span class="ls-seg ls-pick" style="width:27%">and</span><span class="ls-seg" style="width:22%">.</span><span class="ls-seg" style="width:14%"></span><span class="ls-seg ls-rest" style="width:37%"></span></span><span class="ls-draw">and</span></div>
</div>
<div class="ls-row fragment" data-colloquium-fragment="1">
<div class="ls-ctx">the cat sat on the mat <b>and</b></div>
<div class="ls-line"><span class="ls-plab"></span><span class="ls-bar"><span class="ls-seg" style="width:26%">fell</span><span class="ls-seg ls-pick" style="width:22%">purred</span><span class="ls-seg" style="width:13%">shut</span><span class="ls-seg ls-rest" style="width:39%"></span></span><span class="ls-draw">purred</span></div>
</div>
<div class="ls-row fragment" data-colloquium-fragment="1">
<div class="ls-ctx">the cat sat on the mat and <b>purred</b></div>
<div class="ls-line"><span class="ls-plab"></span><span class="ls-bar"><span class="ls-seg ls-pick" style="width:29%">softly</span><span class="ls-seg" style="width:17%">until</span><span class="ls-seg" style="width:11%">for</span><span class="ls-seg ls-rest" style="width:43%"></span></span><span class="ls-draw">softly</span></div>
</div>
<div class="ls-aside fragment" data-fragment-index="4">the token is <b>drawn</b>, not chosen -- the top one does not always win</div>
<div class="ls-foot fragment" data-colloquium-fragment="1">One pass predicts <b>one token</b>. Feeding it back is how a next-token model samples <b>whole sequences</b>.</div>
</div>
