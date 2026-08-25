// Finite-data chart (slide "What if data was finite?"): marker styling.
//
// Colloquium's ```chart``` block gives every line dataset `backgroundColor:
// "transparent"` and `tension: 0.3`, which draws hollow markers on a curved line.
// Hollow is a *meaning* elsewhere in this deck -- results-chart.js uses it for the
// alpha-sweep points that are still data-limited and therefore only bounds -- so
// leaving it here would say something the slide does not mean.  Both series here
// are ordinary measurements, so both get filled markers and straight log-log
// segments, exactly like the capacity and emergence charts.
//
// The chart is identified by a dataset label no other chart in the deck uses, so
// this file never touches the capacity, results or emergence charts.  Loaded from
// a <script src> tag placed after the chart block in slides.md: the pass below runs
// at parse time and only sees the canvases parsed so far.
(function () {
  function isFiniteChart(cfg) {
    return (cfg.data.datasets || []).some(function (ds) {
      return (ds.label || "").indexOf("context pool") !== -1;
    });
  }

  document.querySelectorAll("canvas[data-chart-config]").forEach(function (c) {
    var cfg = JSON.parse(c.getAttribute("data-chart-config"));
    if (!isFiniteChart(cfg)) return;
    cfg.data.datasets.forEach(function (ds) {
      ds.tension = 0;          // log-log data: straight segments, no invented curvature
      ds.pointRadius = 2.8;
      ds.pointBackgroundColor = ds.borderColor;   // filled markers, not rings
      ds.pointBorderColor = ds.borderColor;
    });
    c.setAttribute("data-chart-config", JSON.stringify(cfg));
  });
})();
