<!-- The lab scoreboard: predicted vs actual hero loss, live from the form's
     responses sheet.  Unlike every other chart in the deck, no number here is
     computed at build time -- the canvas is empty in slides.html and is filled in
     the browser by assets/hero-board.js, which is where the whole mechanism (URL
     kinds, freshness, column matching) is documented.

     IT IS WIRED AND WORKING: the form is "Tutorial scaling laws MLSS" (Predicted
     loss / Obtained loss / Fraction spent on hero run -- no name, so the plot is
     anonymous and a resubmission is a second dot), its QR is in
     figures/hero-qr.md, and data-csv-b64 below is the live-CSV URL of its
     responses sheet.  To see what that is, or to check that it still answers:

         uv run python scripts/hero_board.py --current

     WHY IT IS BASE64 AND NOT READABLE HERE.  The sheet has to be readable by
     anyone holding its link for this slide to work, so that link is a password of
     sorts -- and this file, and the slides.html built from it, are published.
     Base64 is obfuscation, not secrecy: the browser's network tab shows the
     request either way.  What it buys is that the URL is not in the page source,
     the repo, or a search index of either.  Keep it that way -- do not paste the
     decoded URL into a commit, and prefer `--current` over `--csv "<url>"` so it
     does not land in a shell history either.

     To repoint the slide, at the same sheet or another one:

         uv run python scripts/hero_board.py --sheet "<the sheet's share link>"

     which reads the tab's gid out of the link, writes the encoded URL back here
     and immediately reads it the way the slide will.  A sheet that was never
     shared shows up as "sheet not readable" in red under the plot, and NOT as a
     401 -- see assets/hero-board.js for that CORS wrinkle.

     A File > Share > Publish to web CSV also works, as a comma-separated second
     entry, but Google caches it by ~5 minutes: a fallback, not a source. -->
<div class="hero-board" data-hero-board
     data-csv-b64="aHR0cHM6Ly9kb2NzLmdvb2dsZS5jb20vc3ByZWFkc2hlZXRzL2QvMWR5RWRiWEVlcDk1UVkyV3pxZnNFUnpoZnIwcDZCRVVGR0xSTVpsRDBTZ2MvZ3Zpei90cT90cXg9b3V0JTNBY3N2JmdpZD05MzkwOTY5MDQ=">

<div class="hero-board-plot"><canvas></canvas></div>

<div class="hero-board-foot">
<div class="hero-bar"><span>0 %</span><i></i><span>100 %</span><em>of the budget spent on the hero run</em></div>
<div class="hero-status" data-hero-status>loading the sheet ...</div>
</div>

</div>

<script src="assets/hero-board.js"></script>
