// Force the pydata-sphinx-theme theme switcher to cycle light ↔ dark only,
// skipping the "auto" (system-selected) state — applied site-wide.
//
// pydata-sphinx-theme stores the preference under `localStorage["mode"]` and
// reflects it on `<html data-theme="...">`. The built-in cycle is:
//   light → dark → auto → light
// We rewrite that to:
//   light → dark → light

(function () {
  function applyMode(mode) {
    // mode ∈ {"light", "dark"}
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.theme = mode;
    try { localStorage.setItem("mode", mode); } catch (_) {}
  }

  function onReady() {
    // If the page was loaded with "auto" stored, normalise to a concrete mode.
    const stored = (function () {
      try { return localStorage.getItem("mode"); } catch (_) { return null; }
    })();
    if (stored === "auto" || stored === null) {
      applyMode("light");
    }

    // Intercept clicks on the theme-switcher button(s).
    document.querySelectorAll(".theme-switch-button").forEach((btn) => {
      btn.addEventListener(
        "click",
        function (e) {
          e.preventDefault();
          e.stopImmediatePropagation();
          const current = document.documentElement.dataset.mode || "light";
          applyMode(current === "light" ? "dark" : "light");
        },
        true   // capture phase, before pydata's own listener runs
      );
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onReady);
  } else {
    onReady();
  }
})();
