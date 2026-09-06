<!-- Hand-written figure: no generator script.  All three series are at the SAME
     budget, D = 2.62e7 draws / 409,600 steps, one seed, lr* interior at every cell:
       "first 2k contexts"    results/finite_support_sweep_k2000.json, the s409600 cells
       "infinite context pool" results/grid.json, the s409600 column
     Copy the `excess_star` field (not `excess_best`).  Raising the budget from the
     original 102,400 steps moved BOTH curves: see README, "Why 409,600 steps".

     "first 2k, tested on all" is the SAME nine models as "first 2k contexts",
     re-scored against the untruncated Zipf instead of against the 2k pool they
     were trained on -- a genuine train/test split, since the model has never seen
     a context beyond rank 2,000 and the full distribution puts 19.1% of its mass
     there.  From the `excess_test` field of results/finite_support_test_k2000.json
     (scripts/finite_test_loss.py).  That file also re-scores each model on its
     training distribution and reproduces the recorded training loss to better than
     5e-7 at every cell, which is what makes it safe to draw the two blue curves as
     one pair rather than as two experiments.

     CAVEAT, and it is not a small one: the two blue series were regenerated under
     the *new* conditional p(y|x) (the universal entropy-indexed profile), while the
     red series still comes from results/grid.json, which predates that rewrite.
     L_inf moved only 2.4598 -> 2.4609 nats, so the comparison is close to fair, but
     "the held-out curve stays worse than the infinite pool" is a blue-vs-red claim
     and will not be strictly apples-to-apples until grid.json is re-run. -->
<div class="cap-legend">
<span class="fragment" data-fragment-index="1"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-navy)" stroke-width="var(--fig-data-width)"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-navy)"/></svg>first 2k contexts</span>
<span class="fragment" data-fragment-index="1"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-navy)" stroke-width="var(--fig-data-width)" stroke-dasharray="6 5"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-navy)"/></svg>tested on all</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--deck-red)" stroke-width="var(--fig-data-width)"/><circle cx="15" cy="5" r="2.8" fill="var(--deck-red)"/></svg>infinite context pool</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "first 2k contexts"
      color: "var(--deck-navy)"
      data:
        - {x: 4096, y: 1.501228}
        - {x: 8192, y: 1.086128}
        - {x: 16384, y: 0.753500}
        - {x: 32768, y: 0.478451}
        - {x: 65536, y: 0.263259}
        - {x: 131072, y: 0.123008}
        - {x: 262144, y: 0.059392}
        - {x: 524288, y: 0.037252}
        - {x: 1048576, y: 0.029313}
    - label: "first 2k, tested on all"
      color: "var(--deck-navy)"
      data:
        - {x: 4096, y: 2.205102}
        - {x: 8192, y: 1.908226}
        - {x: 16384, y: 1.684341}
        - {x: 32768, y: 1.549851}
        - {x: 65536, y: 1.487971}
        - {x: 131072, y: 1.436818}
        - {x: 262144, y: 1.203615}
        - {x: 524288, y: 1.012545}
        - {x: 1048576, y: 0.868785}
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
    # for rather than left to Chart.js's log-scale choice.  The 2k training curve
    # reaches 0.0293, a decade below where the 10k one stopped, so the floor and
    # the tick list both go one step lower than they used to.
    yTicks: at:0.02,0.05,0.1,0.2,0.5,1,2
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
      # The training curve bottoms out at 0.0293 on the 2k pool; 0.02 keeps that
      # marker clear of the axis.
      min: 0.02
      # The test curve starts at 2.21, so the ceiling has to clear it; 2.4 leaves
      # the top marker clear of the plot edge without adding a whole decade.
      max: 2.4
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/plot.js"></script>
