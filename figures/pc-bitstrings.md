<div class="cap-legend">
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#7ba3cc" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#7ba3cc"/></svg>0.17M</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#4a7db0" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#4a7db0"/></svg>0.5M</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#26598f" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#26598f"/></svg>2.5M</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>7M params</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#9ca3af" stroke-width="1.6" stroke-dasharray="6 5"/></svg>bits in the data</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "0.17M params"
      color: "#7ba3cc"
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
      color: "#4a7db0"
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
      color: "#26598f"
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
      color: "#0f3460"
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
      color: "#9ca3af"
      data:
        - {x: 256, y: 0.18}
        - {x: 2097152, y: 1476}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "training set size (sequences)"}
      min: 190
      max: 3200000
      grid: {drawOnChartArea: false}
      ticks: {padding: 6}
    y:
      type: logarithmic
      title: {display: true, text: "memory stored (M bits)", font: {size: 11}}
      min: 0.08
      max: 320
      grid: {drawOnChartArea: false}
      ticks: {padding: 6}
```

<script src="assets/pc-chart.js"></script>
