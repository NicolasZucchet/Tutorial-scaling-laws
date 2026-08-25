// Practice-capacity charts (slide "Capacity, in theory and in practice", the
// second one): marker/dash styling and decade-only tick labels written 1k / 1M.
//
// The two charts on that slide share the y-axis title "memory stored (M bits)",
// which no other chart in the deck uses; that is how they are picked out here,
// the same way assets/emergence-chart.js identifies its pair.  The styling pass
// runs at parse time, so the <script src> tag has to sit after both chart blocks
// -- it lives at the end of figures/pc-bitstrings.md, the right-hand panel, and
// therefore sees the canvases of both panels.  One file for the pair rather than
// one per figure: the pass is shared, and the tick plugin must only be
// registered once (hence the __pcTicksRegistered guard below).  The plugin waits
// for Chart.js, which is deferred.
(function () {
  var Y_TITLE = "memory stored (M bits)";
  function isPC(cfg) {
    var y = (((cfg.options || {}).scales || {}).y || {}).title || {};
    return y.text === Y_TITLE;
  }
  document.querySelectorAll("canvas[data-chart-config]").forEach(function (c) {
    var cfg = JSON.parse(c.getAttribute("data-chart-config"));
    if (!isPC(cfg)) return;
    // Allen-Zhu and Li plot one dot per trained model and draw no curve through
    // them; Morris et al. plot one curve per model.  The left panel is the one
    // whose series are dataset sizes.
    var scatter = (cfg.data.datasets || []).some(function (ds) {
      return (ds.label || "").indexOf("facts") !== -1;
    });
    cfg.data.datasets.forEach(function (ds) {
      var label = ds.label || "";
      ds.tension = 0;   // log-log data: straight segments, no invented curvature
      ds.pointBackgroundColor = ds.borderColor;
      if (label.indexOf("ref:") === 0) {   // yardsticks: dashed, unmarked, behind
        ds.borderDash = label.indexOf("1 bit") !== -1 ? [2, 4] : [6, 5];
        ds.borderWidth = 1.6;
        ds.pointRadius = 0;
        ds.order = 10;
      } else if (scatter) {
        ds.showLine = false;
        ds.pointRadius = 2.4;
      } else {
        ds.borderWidth = 2.2;
        ds.pointRadius = 2.4;
      }
    });
    c.setAttribute("data-chart-config", JSON.stringify(cfg));
  });
  // Ticks on powers of ten only, and written 1k / 1M rather than "1,000,000":
  // the axes run over five decades and the long forms collide.
  function compact(v) {
    if (v >= 1e6) return (v / 1e6) + "M";
    if (v >= 1e3) return (v / 1e3) + "k";
    return String(v);
  }
  function isDecade(v) {
    var k = Math.round(Math.log(v) / Math.LN10);
    return v > 0 && Math.abs(Math.pow(10, k) - v) <= 1e-6 * v;
  }
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof Chart === "undefined" || window.__pcTicksRegistered) return;
    window.__pcTicksRegistered = true;
    Chart.register({
      id: "pcTicks",
      beforeInit: function (chart) {
        if (!isPC(chart.config)) return;
        var scales = (chart.config.options || {}).scales || {};
        ["x", "y"].forEach(function (id) {
          if (!scales[id]) return;
          scales[id].afterBuildTicks = function (axis) {
            axis.ticks = axis.ticks.filter(function (t) { return isDecade(t.value); });
          };
          scales[id].ticks = Object.assign({autoSkip: false, maxRotation: 0},
                                           scales[id].ticks, {callback: compact});
        });
      },
    });
  });
})();
