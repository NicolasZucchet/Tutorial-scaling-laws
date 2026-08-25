<div class="cap-legend">
<span><svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><circle cx="8" cy="5" r="3" fill="#7ba3cc"/></svg>N = 50K</span>
<span><svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><circle cx="8" cy="5" r="3" fill="#4a7db0"/></svg>200K</span>
<span><svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><circle cx="8" cy="5" r="3" fill="#26598f"/></svg>1M</span>
<span><svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><circle cx="8" cy="5" r="3" fill="#0f3460"/></svg>2M facts</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#9ca3af" stroke-width="1.6" stroke-dasharray="6 5"/></svg>2 bits/param</span>
<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="#c0392b" stroke-width="1.6" stroke-dasharray="2 4"/></svg>1 bit/param</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "50K facts"
      color: "#7ba3cc"
      data:
        - {x: 0.757, y: 0.915}
        - {x: 1.12, y: 1.37}
        - {x: 1.45, y: 1.83}
        - {x: 2.65, y: 2.6}
        - {x: 3.71, y: 2.8}
        - {x: 6.1, y: 2.91}
        - {x: 9.47, y: 2.91}
        - {x: 12.5, y: 2.92}
    - label: "200K facts"
      color: "#4a7db0"
      data:
        - {x: 2.24, y: 3.47}
        - {x: 3.02, y: 5.05}
        - {x: 4.45, y: 6.65}
        - {x: 6.1, y: 8.05}
        - {x: 7.59, y: 8.88}
        - {x: 11, y: 10.2}
        - {x: 14.3, y: 11.1}
        - {x: 19.1, y: 11.2}
        - {x: 28.6, y: 11.2}
        - {x: 48.3, y: 11.3}
    - label: "1M facts"
      color: "#26598f"
      data:
        - {x: 11, y: 16.5}
        - {x: 14.3, y: 22.2}
        - {x: 19.1, y: 28.4}
        - {x: 24.9, y: 34.7}
        - {x: 28.6, y: 37.9}
        - {x: 41.5, y: 43}
        - {x: 54.8, y: 46.6}
        - {x: 76.9, y: 53.7}
        - {x: 110, y: 53.9}
        - {x: 162, y: 54.1}
        - {x: 225, y: 54.1}
    - label: "2M facts"
      color: "#0f3460"
      data:
        - {x: 20.9, y: 27.7}
        - {x: 25, y: 38.5}
        - {x: 30.7, y: 39.3}
        - {x: 40.3, y: 58.3}
        - {x: 58.5, y: 73.7}
        - {x: 113, y: 105}
        - {x: 162, y: 106}
        - {x: 236, y: 106}
        - {x: 319, y: 106}
        - {x: 441, y: 106}
    - label: "ref: 2 bits/param"
      color: "#9ca3af"
      data:
        - {x: 0.6, y: 1.2}
        - {x: 600, y: 1200}
    - label: "ref: 1 bit/param"
      color: "#c0392b"
      data:
        - {x: 0.6, y: 0.6}
        - {x: 600, y: 600}
options:
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "model size (M params)"}
      min: 0.6
      max: 600
      grid: {drawOnChartArea: false}
      ticks: {padding: 6}
    y:
      type: logarithmic
      title: {display: true, text: "memory stored (M bits)", font: {size: 11}}
      min: 0.6
      max: 1300
      grid: {drawOnChartArea: false}
      ticks: {padding: 6}
```
