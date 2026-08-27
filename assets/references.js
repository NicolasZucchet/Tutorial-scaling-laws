// Reference-list punctuation (the "References" slides colloquium generates).
//
// colloquium's `_format_reference` builds each entry as
//
//     parts = [f'{author_line}.']
//     parts.append(f' &ldquo;<em>{title}</em>.&rdquo;')
//
// i.e. it appends a period after the author list and after the title without
// looking at what is already there.  Both already end in punctuation often
// enough that the doubling is visible on almost every line of the deck's
// bibliography:
//
//     Levine, Y., ... and Shashua, A.. "Limits to Depth Efficiencies ..."
//                                  ^^   author list ends in an initial
//     Liu, Z., Zhao, C., ... et al.. "MobileLLM: ..."
//                                ^^   truncated list ends in "et al."
//     ... "How Much Do Language Models Memorize?." arXiv preprint ...
//                                              ^^ title ends in a question mark
//
// This runs over the generated entries and drops the redundant period.  It is a
// display fix, not a data fix: refs.bib is correct, and the author fields cannot
// be written any other way -- colloquium derives "A." from the first name itself.
//
// Done in JS rather than as a post-processing pass in scripts/build_slides.py so
// that `serve` shows the same thing the built deck does: colloquium renders the
// references itself in both, and only the static build passes through our script.
// Waits for DOMContentLoaded because the reference slides come last in the
// document, after the <script src> on the title slide has already been parsed.
(function () {
  function tidy() {
    document.querySelectorAll(".colloquium-reference").forEach(function (ref) {
      var html = ref.innerHTML;
      // Author list already ended in "." (an initial, or "et al.").
      html = html.replace(/\.\.(\s*)(“|&ldquo;)/, ".$1$2");
      // Title already ended in its own punctuation, inside the quotes.
      html = html.replace(/([.?!])<\/em>\.(\s*)(”|&rdquo;)/, "$1</em>$2$3");
      ref.innerHTML = html;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tidy);
  } else {
    tidy();
  }
})();
