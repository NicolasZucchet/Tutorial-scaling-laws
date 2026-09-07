// The lab's live scoreboard: every student's PREDICTED hero loss against the
// loss they ACTUALLY got, coloured by the share of the budget they spent on
// that one run.  The data is whatever the Google Form has collected so far,
// read straight out of its responses sheet as CSV while the slide is on screen.
//
// WHY IT FETCHES INSTEAD OF BEING BAKED IN.  Every other chart in the deck is a
// ```chart fence whose numbers were computed at build time (see assets/plot.js);
// this one cannot be, because the numbers do not exist until the room has run
// the lab.  So it is the deck's only chart built in the browser: a plain
// <canvas> with no `data-chart-config`, which colloquium's bootstrap therefore
// leaves alone, and Chart.js -- already on the page for the six baked charts --
// driven from here.  assets/plot.js's global plugin still runs on it (that is
// how it gets the deck's type, and its canvas pinned to the container's own
// pixels), but the `plot:` styling pass is skipped: this chart has no fence.
//
// WHERE THE DATA COMES FROM, and the one thing to get right when you set it up.
// The container names one or more CSV URLs, tried in order until one parses, in
// either of two attributes:
//
//   data-csv-b64   the same comma-separated list, base64'd.  THIS IS THE ONE THE
//                  DECK SHIPS.  A responses sheet has to be readable by anyone
//                  with its link for this slide to work at all, so its URL is a
//                  password of sorts -- and slides.html is published.  Base64 is
//                  NOT security: anyone who opens the network tab sees the
//                  request.  What it buys is that the URL is not sitting in the
//                  page source, in the repo, or in a search engine's index of
//                  either, which is the difference between "someone had to go
//                  looking" and "someone scrolled past it".
//   data-csv       the readable form, for a sheet nobody minds sharing.
//
// Two kinds of URL work in both, and they are NOT equally fresh:
//
//   gviz   https://docs.google.com/spreadsheets/d/<SHEET_ID>/gviz/tq?tqx=out:csv&sheet=<TAB>
//          Served live -- a response shows up on the next poll, a few seconds
//          later.  Needs the sheet shared as "anyone with the link can view".
//          THIS IS THE ONE TO USE DURING THE LAB.
//
//   pub    https://docs.google.com/spreadsheets/d/e/<PUB_ID>/pub?gid=<GID>&single=true&output=csv
//          File > Share > Publish to web.  No link-sharing needed, but Google
//          caches it, so a new response can take ~5 minutes to appear.  Fine as
//          the second entry in `data-csv`, useless as the only one.
//
// Both send `access-control-allow-origin: *`, so the fetch works from the Pages
// deck and from a file:// copy alike.  Neither needs an API key.
//
// COLUMNS are matched by what their headers *contain*, case-insensitively (see
// COLUMNS below), so the form's questions can be worded however you like as long
// as the predicted-loss question says "predict", the actual-loss one says
// "actual" or "obtained", and the compute one says "%", "fraction" or "compute".
// Nothing depends on column order, and a "Timestamp" column is ignored.
//
// THE PLOT IS ANONYMOUS, by construction and not by convention: the form does not
// ask who you are, so the sheet has no name column and there is nothing here to
// label a dot with.  Every row is a dot, and a second submission is a second dot
// -- without a name there is no way to tell a correction from a classmate.  This
// is also why there is no in-plot label pass: an earlier version drew names
// beside the dots, and it went out with the question.
(function () {
  if (window.__heroBoard) return;   // one deck, one execution, however many tags
  window.__heroBoard = true;

  var SEL = "[data-hero-board]";

  // Poll while the slide is up; back right off when it is not.  The sheet is a
  // static CSV of a few hundred bytes, so 8 s is cheap, and the lab's whole
  // point is that the plot fills in while the room watches.
  var ACTIVE_MS = 8000;
  var IDLE_MS = 60000;

  // The colour axis is FIXED at 0-100 % of the budget rather than fitted to the
  // data: a scale that rescaled itself as responses arrived would repaint every
  // dot mid-talk, and "dark means most of the budget went into the hero run" is
  // the reading we want, not "dark means most of it *among those who answered so
  // far*".  The colourbar in figures/hero-board.md is the same two tokens.
  var SHARE_LO = 0, SHARE_HI = 100;

  // ------------------------------------------------------------------- CSV
  //
  // A hand-rolled parser rather than a split(","): a name with a comma in it,
  // or a free-text answer with a newline, is a quoted field, and splitting
  // silently shifts every later column of that row into the wrong question.
  function parseCSV(text) {
    var rows = [], row = [], field = "", quoted = false;
    text = String(text).replace(/^﻿/, "");
    for (var i = 0; i < text.length; i++) {
      var c = text.charAt(i);
      if (quoted) {
        if (c !== '"') { field += c; continue; }
        if (text.charAt(i + 1) === '"') { field += '"'; i++; }   // "" -> literal "
        else quoted = false;
      } else if (c === '"') {
        quoted = true;
      } else if (c === ",") {
        row.push(field); field = "";
      } else if (c === "\n" || c === "\r") {
        if (c === "\r" && text.charAt(i + 1) === "\n") i++;
        row.push(field); field = ""; rows.push(row); row = [];
      } else {
        field += c;
      }
    }
    row.push(field);
    rows.push(row);
    return rows.filter(function (r) {
      return r.some(function (v) { return v.trim() !== ""; });
    });
  }

  // Header -> role, by substring.  Order is priority: "predicted loss" and
  // "obtained loss" both contain "loss", so the more specific rules run first,
  // and each header can only be claimed once.
  var COLUMNS = [
    ["predicted", /predict|guess|expect/i],
    ["actual", /actual|obtain|achiev|measur|real|got/i],
    ["share", /%|percent|share|fraction|compute|flop|budget/i],
  ];

  function mapColumns(header) {
    var of = {}, taken = {};
    COLUMNS.forEach(function (pair) {
      for (var i = 0; i < header.length; i++) {
        if (taken[i] || !pair[1].test(header[i])) continue;
        of[pair[0]] = i;
        taken[i] = true;
        return;
      }
    });
    return of;
  }

  // "0.42", "42%", "1,024" -> a number; anything else -> null, which is what
  // keeps a half-filled row from plotting a dot at zero.
  function num(v) {
    var x = parseFloat(String(v == null ? "" : v).replace(/[,\s%]/g, ""));
    return isFinite(x) ? x : null;
  }

  // The compute question can be answered as a percentage or as a fraction, and
  // both readings are common enough that guessing is better than dropping the
  // row: an explicit "%" is taken at face value, and a bare number <= 1 is read
  // as a fraction.  (A hero run is never 1 % of the budget, so the ambiguous
  // case at exactly 1 does not arise in practice.)
  function share(raw) {
    var x = num(raw);
    if (x === null) return null;
    if (String(raw).indexOf("%") !== -1) return x;
    return x <= 1 ? x * 100 : x;
  }

  function entries(text) {
    var rows = parseCSV(text);
    if (rows.length < 2) return [];
    var of = mapColumns(rows[0].map(function (h) { return h.trim(); }));
    if (of.predicted == null || of.actual == null) {
      throw new Error('the sheet has no "predicted" and "actual" columns -- ' +
                      "its headers are: " + rows[0].join(" | "));
    }
    var out = [];
    rows.slice(1).forEach(function (r) {
      var p = num(r[of.predicted]), a = num(r[of.actual]);
      if (p === null || a === null) return;      // a partial row is not a point
      out.push({predicted: p, actual: a, share: share(r[of.share])});
    });
    return out;
  }

  // ---------------------------------------------------------------- colour
  function token(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function rgb(hex) {
    var h = String(hex).trim().replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
            parseInt(h.slice(4, 6), 16)];
  }

  // One hue, light to dark, because the quantity has no midpoint and no good or
  // bad end -- two hues would read as two kinds of run.  The endpoints are the
  // same tokens the colourbar's CSS gradient uses.
  function ramp(t) {
    var lo = rgb(token("--hero-share-lo", "#9ac7e2"));
    var hi = rgb(token("--hero-share-hi", "#0f3460"));
    var u = Math.max(0, Math.min(1, t));
    var mix = lo.map(function (c, i) { return Math.round(c + u * (hi[i] - c)); });
    return "rgb(" + mix.join(",") + ")";
  }

  function colorFor(e) {
    if (e.share === null) return token("--fig-guide", "#9ca3af");
    return ramp((e.share - SHARE_LO) / (SHARE_HI - SHARE_LO));
  }

  // ------------------------------------------------------------------ axes
  //
  // Both axes get the SAME range, always: the whole slide is "did the dot land
  // on the diagonal", and that question is only legible when the diagonal is at
  // 45 degrees and a horizontal distance means what a vertical one does.
  function bounds(data) {
    var vs = [];
    data.forEach(function (e) { vs.push(e.predicted, e.actual); });
    // With nothing to plot yet, the axes still have to say something.  3.20-3.40
    // is where the hero losses actually land: the three reference attempts in
    // README.md come out at 3.2609, 3.2765 and 3.2993 nats, against a zero-tuning
    // ceiling of 3.2552.  So the empty plot is the plot the first dot arrives
    // into, rather than a window that jumps the moment it does.
    if (!vs.length) return [3.20, 3.40];
    var lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs);
    var pad = Math.max(0.04, 0.12 * (hi - lo));
    return [lo - pad, hi + pad];
  }

  // At most six ticks per axis, from steps of 1, 2 or 5 x 10^k, and the densest
  // such step wins.  Two constraints force this:
  //
  //   * A label like "3.26" is ~48 deck units wide at the deck's 20px axis type,
  //     and the plot is ~700 wide, so six is comfortable and ten is not.  Chart.js
  //     autoSkip does still apply to the list installed from afterBuildTicks below
  //     -- with maxRotation 0 it drops every other x label rather than tilting
  //     them -- but leaning on that labels the two axes at different intervals,
  //     and both axes here carry the same range, so that reads as a mistake.
  //   * No 2.5 in the set: a 0.025 step over the losses this lab produces gives
  //     ticks at 3.225 and 3.275, which either print as 3.23 and 3.28 -- labels
  //     that do not name the lines they sit on -- or take a third decimal and get
  //     wider still.  Every step here is exact at the decimals `fmt` shows.
  //
  // The cost is that a wide spread can come down to three or four labels.  That is
  // the right trade for this plot: what is being read is a dot's distance from the
  // diagonal, and the axis is there to give that distance a scale, not to be
  // counted along.
  var MAX_TICKS = 6;

  function ticksAt(step, lo, hi) {
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      out.push(Math.round(v / step) * step);
    }
    return out;
  }

  function ticks(lo, hi) {
    var steps = [];
    for (var k = -4; k <= 1; k++) {
      [1, 2, 5].forEach(function (m) { steps.push(m * Math.pow(10, k)); });
    }
    for (var i = 0; i < steps.length; i++) {
      var list = ticksAt(steps[i], lo, hi);
      if (list.length <= MAX_TICKS) return list;
    }
    return ticksAt(steps[steps.length - 1], lo, hi);
  }

  // As many decimals as the step needs and no more, so a 0.02 step reads 3.26 and
  // a 0.005 one reads 3.265.
  function fmt(v, step) {
    return v.toFixed(Math.max(0, Math.ceil(-Math.log(step) / Math.LN10)));
  }

  // ----------------------------------------------------------------- chart
  function build(canvas) {
    var guide = token("--fig-guide", "#9ca3af");
    var dash = token("--fig-dash", "6 5").split(/[\s,]+/).map(Number);
    var point = parseFloat(token("--fig-point", "2.8"));
    return new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          // Dataset 0 is the y = x line: a yardstick, so hairline, dashed and
          // behind.  Dataset 1 is the room.
          {label: "perfect prediction", data: [], showLine: true, borderColor: guide,
           borderDash: dash, borderWidth: parseFloat(token("--fig-hair-width", "1.5")),
           pointRadius: 0, order: 10},
          // 2x the deck's marker token: these dots are the whole slide and the plot
          // is now ~700 units wide, so they carry more ink than a series marker on
          // a half-slide chart would.
          {label: "submissions", data: [], showLine: false, pointRadius: point * 2,
           pointHoverRadius: point * 2, pointBackgroundColor: [],
           pointBorderColor: token("--fig-surface", "#ffffff"), pointBorderWidth: 1,
           order: 1},
        ],
      },
      options: {
        maintainAspectRatio: false,
        animation: false,
        devicePixelRatio: 3,
        plugins: {legend: {display: false}, tooltip: {enabled: false}},
        scales: {
          // maxRotation 0: the tick list is fixed (see `draw`), so Chart.js cannot
          // thin it, and its fallback when labels crowd is to tilt them 45
          // degrees -- which no other axis in the deck does.
          x: {type: "linear", title: {display: true, text: "predicted loss (nats)"},
              grid: {drawOnChartArea: false}, ticks: {padding: 8, maxRotation: 0}},
          y: {type: "linear", title: {display: true, text: "actual loss (nats)"},
              grid: {drawOnChartArea: false}, ticks: {padding: 8}},
        },
      },
    });
  }

  // Chart.js sizes a responsive canvas from the canvas's *rendered* rect, and the
  // deck is a fixed 1280x720 box that a CSS transform scales to the viewport --
  // so on a 1600px-wide window every measurement came back 1.25x too big and the
  // plot ran off the right edge of the slide.  assets/plot.js hit this too and
  // documents it at length under `pinSize`; its fix runs at beforeInit, which is
  // too early here (this chart is built while its slide is still display:none, so
  // the box has no size yet).  Hence the same fix, applied on every draw and
  // therefore first succeeding on the draw that follows the slide going up.
  //
  // The canvas takes its box whole -- the slide has one wide column for it, and a
  // plot that fills it is worth more here than a square one.  What the square
  // bought was the y = x line at 45 degrees, which is the cleanest form of "how
  // far off was the prediction"; at this aspect the line lies flatter and a miss
  // reads as vertical distance from it.  That still reads, and the axes still
  // carry the same range (see `bounds`), so the line is still the identity and
  // not a fit.
  //
  // Pinning at all, rather than leaving Chart.js responsive: Chart.js sizes a
  // responsive canvas from the canvas's *rendered* rect, and the deck is a fixed
  // 1280x720 box that a CSS transform scales to the viewport -- so on a 1600px
  // window every measurement came back 1.25x too big and the plot ran off the
  // slide.  assets/plot.js hit this too and documents it at length under
  // `pinSize`; its fix runs at beforeInit, which is too early here (this chart is
  // built while its slide is still display:none, so the box has no size yet).
  // Hence the same fix, applied on every draw, first succeeding on the draw that
  // follows the slide going up.
  function pin(chart) {
    var box = chart.canvas.parentNode;
    if (!box || !box.clientWidth || !box.clientHeight) return;
    chart.options.responsive = false;
    chart.resize(box.clientWidth, box.clientHeight);
  }

  function draw(chart, data) {
    var b = bounds(data), lo = b[0], hi = b[1];
    var tv = ticks(lo, hi), step = tv.length > 1 ? tv[1] - tv[0] : 0.1;
    chart.data.datasets[0].data = [{x: lo, y: lo}, {x: hi, y: hi}];
    chart.data.datasets[1].data = data.map(function (e) {
      return {x: e.predicted, y: e.actual};
    });
    chart.data.datasets[1].pointBackgroundColor = data.map(colorFor);
    ["x", "y"].forEach(function (axis) {
      var s = chart.options.scales[axis];
      s.min = lo;
      s.max = hi;
      s.afterBuildTicks = function (scale) {
        scale.ticks = tv.map(function (v) { return {value: v}; });
      };
      s.ticks.callback = function (v) { return fmt(v, step); };
    });
    pin(chart);
    chart.update("none");
  }

  // ------------------------------------------------------------- no readout
  //
  // There is deliberately no "N runs, median error X" line under the plot.  The
  // plot is the readout: how many dots there are, and which side of the diagonal
  // they sit on, is the whole point, and a sentence restating it in numbers is
  // one more thing on screen to read out loud.  The status element it used to
  // live in is kept for FAILURES only -- it is empty (and, per the stylesheet,
  // not laid out at all) whenever the sheet is being read successfully, so a
  // sheet that stops answering mid-lab still says so instead of just freezing.

  // ----------------------------------------------------------------- fetch
  //
  // Cache-busted, because a re-fetch that the browser answers from its own cache
  // is a scoreboard that stops updating for no visible reason.
  function bust(url) {
    return url + (url.indexOf("?") === -1 ? "?" : "&") + "_=" + Date.now();
  }

  // A sheet that was never made link-readable does NOT come back as 401 here.
  // Google answers the sign-in page with a redirect to accounts.google.com, which
  // carries no CORS header, so the browser rejects the fetch itself and the
  // status is never visible to us -- it surfaces as TypeError "Failed to fetch"
  // in Chrome and "Load failed" in Safari, exactly like being offline.  Since
  // that is the one setup mistake anyone actually makes, the message names it.
  var UNREACHABLE = /failed to fetch|load failed|networkerror|network error/i;

  // Short, because this line sits one line above the deck's footer: anything that
  // wraps lands on top of it.  The full text goes in the title attribute.
  function why(err) {
    if (UNREACHABLE.test(err.message)) {
      return 'sheet not readable -- share it with "anyone with the link"';
    }
    return err.message.length > 60 ? err.message.slice(0, 57) + "..." : err.message;
  }

  function load(urls) {
    var i = 0;
    function attempt() {
      if (i >= urls.length) return Promise.reject(new Error("no CSV URL answered"));
      var url = urls[i++];
      return fetch(bust(url), {cache: "no-store", redirect: "follow"})
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.text();
        })
        .then(function (text) {
          // A sheet that is not shared answers 200 with a sign-in page, which
          // would otherwise parse as one very confusing row.
          if (/^\s*</.test(text)) throw new Error("got HTML, not CSV -- is the sheet shared?");
          return entries(text);
        })
        .catch(function (err) {
          if (i < urls.length) return attempt();
          throw err;
        });
    }
    return attempt();
  }

  // ------------------------------------------------------------------ wire
  // Both attributes, base64 first, concatenated -- so a readable fallback URL can
  // sit beside an obscured primary one.
  function sources(box) {
    var lists = [];
    var b64 = (box.getAttribute("data-csv-b64") || "").replace(/\s+/g, "");
    if (b64) {
      try {
        lists.push(atob(b64));
      } catch (e) {
        lists.push("");   // a mangled blob is a missing URL, not an exception
      }
    }
    lists.push(box.getAttribute("data-csv") || "");
    return lists.join(",").split(",")
      .map(function (s) { return s.trim(); })
      .filter(function (s) { return s && s.indexOf("PASTE") === -1; });
  }

  function mount(box) {
    var urls = sources(box);
    var canvas = box.querySelector("canvas");
    var status = box.querySelector("[data-hero-status]");
    var slide = box.closest(".slide");
    var chart = build(canvas);
    var busy = false, timer = null;

    function say(text, bad, full) {
      if (!status) return;
      status.textContent = text;
      status.title = full || text;
      status.classList.toggle("hero-status--bad", !!bad);
    }

    if (!urls.length) {
      draw(chart, []);
      say("no sheet configured -- see figures/hero-board.md", true);
      return;
    }

    function refresh() {
      if (busy) return;
      busy = true;
      load(urls).then(function (data) {
        draw(chart, data);
        say("");
      }).catch(function (err) {
        say(why(err), true, err.message + " -- click the line to retry");
      }).then(function () { busy = false; });
    }

    // Poll fast while the slide is up, slowly when it is not, and not at all in
    // a hidden tab -- a deck left open on a lectern should not sit on the sheet
    // all afternoon.
    function tick() {
      var active = !slide || slide.classList.contains("active");
      if (!document.hidden && active) refresh();
      timer = setTimeout(tick, active ? ACTIVE_MS : IDLE_MS);
    }

    // Arriving on the slide asks straight away rather than waiting out the
    // current interval; so does clicking the status line, which is the escape
    // hatch when the venue's wifi dropped a request.
    if (slide) {
      new MutationObserver(function () {
        if (slide.classList.contains("active")) {
          clearTimeout(timer);
          tick();
        }
      }).observe(slide, {attributes: true, attributeFilter: ["class"]});
    }
    if (status) {
      status.style.cursor = "pointer";
      status.addEventListener("click", refresh);
    }
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) refresh();
    });

    draw(chart, []);
    say("loading the sheet ...");
    tick();
  }

  window.addEventListener("load", function () {
    if (typeof Chart === "undefined") return;
    Array.prototype.forEach.call(document.querySelectorAll(SEL), mount);
  });
})();
