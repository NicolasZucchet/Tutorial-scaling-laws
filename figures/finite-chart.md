<!-- Hand-written figure: no generator script.  Both series are at the SAME budget,
     D = 2.62e7 draws / 409,600 steps, one seed, lr* interior at every cell:
       "first 10k contexts"   results/finite_support_sweep.json, the s409600 cells
       "infinite context pool" results/grid.json, the s409600 column
     Copy the `excess_star` field (not `excess_best`).  Raising the budget from the
     original 102,400 steps moved BOTH curves: see README, "Why 409,600 steps". -->
<div class="cap-legend">
<span class="fragment" data-fragment-index="1"><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>first 10k contexts</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#c0392b"/></svg>infinite context pool</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "first 10k contexts"
      color: "#0f3460"
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
    - label: "infinite context pool"
      color: "#c0392b"
      data:
        - {x: 16384, y: 1.552253}
        - {x: 32768, y: 1.340519}
        - {x: 65536, y: 1.162717}
        - {x: 131072, y: 1.011072}
        - {x: 262144, y: 0.881464}
        - {x: 524288, y: 0.769158}
        - {x: 1048576, y: 0.678173}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "number of parameters N"}
      min: 3500
      max: 1200000
      grid: {drawOnChartArea: false}
      ticks: {padding: 8, maxTicksLimit: 5}
    y:
      type: logarithmic
      title: {display: true, text: "loss"}
      min: 0.05
      max: 1.8
      grid: {drawOnChartArea: false}
      ticks: {padding: 8, maxTicksLimit: 5}
```

<script src="assets/finite-chart.js"></script>
