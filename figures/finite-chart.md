<!-- Hand-written figure: no generator script.  All three series are at the SAME
     budget, D = 2.62e7 draws / 409,600 steps, one seed, lr* interior at every cell:
       "first 10k contexts"   results/finite_support_sweep.json, the s409600 cells
       "infinite context pool" results/grid.json, the s409600 column
     Copy the `excess_star` field (not `excess_best`).  Raising the budget from the
     original 102,400 steps moved BOTH curves: see README, "Why 409,600 steps".

     "first 10k, tested on all" is the SAME nine models as "first 10k contexts",
     re-scored against the untruncated Zipf instead of against the 10k pool they
     were trained on -- a genuine train/test split, since the model has never seen
     a context beyond rank 10,000 and the full distribution puts 13.7% of its mass
     there.  From the `excess_test` field of results/finite_support_test.json
     (scripts/finite_test_loss.py).  That file also re-scores each model on its
     training distribution and reproduces the recorded training loss to better than
     1e-6 at every cell, which is what makes it safe to draw the two blue curves as
     one pair rather than as two experiments. -->
<div class="cap-legend">
<span class="fragment" data-fragment-index="1"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-navy)" stroke-width="var(--fig-data-width)"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-navy)"/></svg>first 10k contexts</span>
<span class="fragment" data-fragment-index="1"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-navy)" stroke-width="var(--fig-data-width)" stroke-dasharray="6 5"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-navy)"/></svg>tested on all</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-red)" stroke-width="var(--fig-data-width)"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-red)"/></svg>infinite context pool</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "first 10k contexts"
      color: "var(--deck-navy)"
      data:
        - {x: 4096, y: 1.693699}
        - {x: 8192, y: 1.324286}
        - {x: 16384, y: 1.032124}
        - {x: 32768, y: 0.783258}
        - {x: 65536, y: 0.566136}
        - {x: 131072, y: 0.380699}
        - {x: 262144, y: 0.228371}
        - {x: 524288, y: 0.127639}
        - {x: 1048576, y: 0.079962}
    - label: "first 10k, tested on all"
      color: "var(--deck-navy)"
      data:
        - {x: 4096, y: 2.139823}
        - {x: 8192, y: 1.828469}
        - {x: 16384, y: 1.585414}
        - {x: 32768, y: 1.389643}
        - {x: 65536, y: 1.230682}
        - {x: 131072, y: 1.119451}
        - {x: 262144, y: 1.050476}
        - {x: 524288, y: 0.904004}
        - {x: 1048576, y: 0.771079}
    - label: "infinite context pool"
      color: "var(--deck-red)"
      data:
        - {x: 16384, y: 1.552253}
        - {x: 32768, y: 1.340519}
        - {x: 65536, y: 1.162717}
        - {x: 131072, y: 1.011072}
        - {x: 262144, y: 0.881464}
        - {x: 524288, y: 0.769158}
        - {x: 1048576, y: 0.678173}
options:
  plot:
    # Styling is declared here and read by assets/plot.js, the deck's one chart
    # layer; see the SCHEMA comment there for the vocabulary.
    #
    # `dash` rather than `guide` for the test curve: it is a measurement, not a
    # yardstick, so it keeps the data stroke weight, its markers and its place in
    # front.  The dash is what says "same models, other distribution" -- same hue
    # as its solid twin, because a second colour would read as a second
    # experiment.
    markers: filled
    dash: ["tested on all"]
    # Parameter counts in the deck's own notation, 10k / 100k / 1M, the way every
    # other model size in it is written.  Chart.js's own labels here were
    # "10,000 / 100,000 / 1,000,000", three numbers whose commas are the only
    # thing telling them apart at a glance.
    xTicks: si
    # The loss axis spans a decade and a half and is read as a number, not as an
    # exponent, so the ticks are pinned to the round values a reader would look
    # for rather than left to Chart.js's log-scale choice.
    yTicks: at:0.05,0.1,0.2,0.5,1,2
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "number of parameters N"}
      min: 3500
      max: 1200000
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
    y:
      type: logarithmic
      title: {display: true, text: "loss"}
      min: 0.05
      # The test curve starts at 2.14, so the ceiling has to clear it; 2.4 leaves
      # the top marker clear of the plot edge without adding a whole decade.
      max: 2.4
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/plot.js"></script>
