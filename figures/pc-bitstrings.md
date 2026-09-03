<!-- Morris et al. 2025, random bitstrings memorized against training set size,
     one curve per model size.

     COLOURS.  The four stops are stops 1, 3, 5 and 6 of the six-stop ladder
     figures/pc-facts.md uses in the panel beside this one, so the two plots on
     the slide are one colour scheme rather than two: light-to-dark is
     small-to-large in both, and every colour here is literally one of the
     colours there.  That ladder is the deck's canonical single-hue RAMP
     (scripts/chinchilla_svg.py) sampled for even lightness steps; see the
     COLOURS note in figures/pc-facts.md for how its t values were chosen.
     Taking every other stop gives ~20 L* per step, which is what four series
     over one lightness range wants.  The dashed yardstick is var(--fig-guide) in both.

     What the colour *means* is transposed between the two panels -- here it is
     the model size and the x axis is the data, there it is the amount of data
     and the x axis is the model -- so the axis titles and the legends have to
     do that work; the shared ramp only says "more" in the same direction. -->
<div class="cap-legend">
<span><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="#86b6ef" stroke-width="var(--fig-data-width)"/><circle cx="11" cy="5" r="2.8" fill="#86b6ef"/></svg>0.17M</span>
<span><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="#3b7fd2" stroke-width="var(--fig-data-width)"/><circle cx="11" cy="5" r="2.8" fill="#3b7fd2"/></svg>0.5M</span>
<span><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="#174d91" stroke-width="var(--fig-data-width)"/><circle cx="11" cy="5" r="2.8" fill="#174d91"/></svg>2.5M</span>
<span><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="#0a2d56" stroke-width="var(--fig-data-width)"/><circle cx="11" cy="5" r="2.8" fill="#0a2d56"/></svg>7M params</span>
<span><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="var(--fig-guide)" stroke-width="var(--fig-hair-width)" stroke-dasharray="6 5"/></svg>bits in the data</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "0.17M params"
      color: "#86b6ef"
      data:
        - {x: 256, y: 0.18}
        - {x: 1024, y: 0.299}
        - {x: 2048, y: 0.314}
        - {x: 4096, y: 0.329}
        - {x: 8192, y: 0.351}
        - {x: 16384, y: 0.359}
        - {x: 32768, y: 0.355}
        - {x: 131072, y: 0.343}
        - {x: 524288, y: 0.291}
        - {x: 2097152, y: 0.203}
    - label: "0.5M params"
      color: "#3b7fd2"
      data:
        - {x: 256, y: 0.18}
        - {x: 1024, y: 0.62}
        - {x: 2048, y: 0.836}
        - {x: 4096, y: 1.014}
        - {x: 8192, y: 1.037}
        - {x: 16384, y: 1.074}
        - {x: 32768, y: 0.998}
        - {x: 131072, y: 0.972}
        - {x: 524288, y: 0.939}
        - {x: 2097152, y: 0.742}
    - label: "2.5M params"
      color: "#174d91"
      data:
        - {x: 256, y: 0.18}
        - {x: 1024, y: 0.72}
        - {x: 2048, y: 1.26}
        - {x: 4096, y: 2.1}
        - {x: 8192, y: 3.31}
        - {x: 16384, y: 3.5}
        - {x: 32768, y: 3.57}
        - {x: 131072, y: 3.35}
        - {x: 524288, y: 3.37}
        - {x: 2097152, y: 2.71}
    - label: "7M params"
      color: "#0a2d56"
      data:
        - {x: 256, y: 0.18}
        - {x: 1024, y: 0.75}
        - {x: 2048, y: 1.47}
        - {x: 4096, y: 3}
        - {x: 8192, y: 5.27}
        - {x: 16384, y: 8.35}
        - {x: 32768, y: 11.7}
        - {x: 131072, y: 12.2}
        - {x: 524288, y: 11.6}
        - {x: 2097152, y: 10.7}
    - label: "ref: bits in the data"
      color: "var(--fig-guide)"
      data:
        - {x: 256, y: 0.18}
        - {x: 2097152, y: 1476}
options:
  plot:
    # The same declaration as the panel beside it (figures/pc-facts.md), for the
    # reason given there: the pair has to read as one comparison.  Read by
    # assets/plot.js; see the SCHEMA comment there.
    markers: filled
    guide: ["ref:"]
    xTicks: si
    yTicks: si
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "training set size D (sequences)"}
      min: 190
      max: 3200000
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
    y:
      type: logarithmic
      title: {display: true, text: "memory stored (M bits)"}
      min: 0.08
      max: 320
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/plot.js"></script>
