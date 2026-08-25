<div class="cap-legend">
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#0f3460" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#0f3460"/></svg>first 10k contexts</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="2.2"/><circle cx="15" cy="5" r="2.8" fill="#c0392b"/></svg>infinite context pool</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "first 10k contexts"
      color: "#0f3460"
      data:
        - {x: 4096, y: 1.694101}
        - {x: 8192, y: 1.325302}
        - {x: 16384, y: 1.033968}
        - {x: 32768, y: 0.786548}
        - {x: 65536, y: 0.573029}
        - {x: 131072, y: 0.394475}
        - {x: 262144, y: 0.253310}
        - {x: 524288, y: 0.163384}
        - {x: 1048576, y: 0.120958}
    - label: "infinite context pool"
      color: "#c0392b"
      data:
        - {x: 16384, y: 1.553531}
        - {x: 32768, y: 1.342995}
        - {x: 65536, y: 1.167684}
        - {x: 131072, y: 1.019898}
        - {x: 262144, y: 0.896523}
        - {x: 524288, y: 0.793940}
        - {x: 1048576, y: 0.715241}
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
      min: 0.1
      max: 1.8
      grid: {drawOnChartArea: false}
      ticks: {padding: 8, maxTicksLimit: 5}
```

<script src="assets/finite-chart.js"></script>
