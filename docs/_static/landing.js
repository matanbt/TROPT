/* TROPT landing-page interactions. Loaded site-wide via conf.py
   html_js_files; each IIFE early-returns when its target DOM isn't present
   (so non-landing pages pay only a few querySelector calls).

   Three independent components:
     1. Hero slot-machine ("optimize [verb] any [noun]")
     2. Hero cursor-following balloons (.tropt-hero ::before/::after)
     3. Per-application interactive demos (.tropt-demo, in Get Started)
*/

/* ------------------------------------------------------------------------
   1. Slot-machine: cycles "Optimize [verb] any [noun]" through TROPT's
   modularity axes — any goal / any model / any trigger / any optimizer /
   any loss / any application. Random ~25% of transitions briefly flash
   "!!!" in one slot, nodding to GCG's classic "! ! ! ! !" initial trigger.
   ------------------------------------------------------------------------ */
(function () {
  var PAIRS = [
    ["toward",  "goal"       ],
    ["against", "model"      ],
    ["for",     "trigger"    ],
    ["with",    "optimizer"  ],
    ["against", "loss"       ],
    ["for",     "application"],
  ];
  var TICK_MS       = 1900;
  var TICK_MS_HOVER = 650;   // accelerated cadence while cursor is over .tropt-optbar
  var OUT_MS    = 170;   // time for the old word to slide up + out
  var SETTLE_MS = 200;   // time for the new word to slide up + in
  var FLASH_MS  = 350;   // how long "!!!" stays before settling on the real word
  var FLASH_P   = 0.25;

  function init() {
    var verb = document.querySelector('.tropt-slot[data-slot="verb"]');
    var noun = document.querySelector('.tropt-slot[data-slot="noun"]');
    if (!verb || !noun) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    /* Slot-machine swap: old word slides up + out, snaps to below (no transition),
       then slides up into place. Done with a single span — no DOM churn. */
    function slotSwap(el, txt) {
      if (el.textContent === txt) return;
      el.style.transition = 'transform ' + OUT_MS + 'ms cubic-bezier(0.4,0,0.2,1), opacity ' + OUT_MS + 'ms ease';
      el.style.transform  = 'translateY(-130%)';
      el.style.opacity    = '0';
      setTimeout(function () {
        el.style.transition = 'none';
        el.textContent      = txt;
        el.style.transform  = 'translateY(130%)';
        /* force reflow so the next transition actually animates */
        void el.offsetHeight;
        el.style.transition = 'transform ' + SETTLE_MS + 'ms cubic-bezier(0.34,1.56,0.64,1), opacity ' + SETTLE_MS + 'ms ease';
        el.style.transform  = 'translateY(0)';
        el.style.opacity    = '1';
      }, OUT_MS);
    }

    var i = 0;
    function tick() {
      i = (i + 1) % PAIRS.length;
      var nextV = PAIRS[i][0], nextN = PAIRS[i][1];
      if (Math.random() < FLASH_P) {
        var pickVerb = Math.random() < 0.5;
        var flashEl  = pickVerb ? verb : noun;
        var finalTxt = pickVerb ? nextV : nextN;
        slotSwap(flashEl, '!!!');
        slotSwap(pickVerb ? noun : verb, pickVerb ? nextN : nextV);
        setTimeout(function () { slotSwap(flashEl, finalTxt); }, FLASH_MS);
      } else {
        slotSwap(verb, nextV);
        slotSwap(noun, nextN);
      }
    }
    var timer = setInterval(tick, TICK_MS);

    /* Hover over the optimize bar → accelerate the cycle; restore on mouseleave. */
    var optbar = document.querySelector('.tropt-optbar');
    if (optbar) {
      optbar.addEventListener('mouseenter', function () {
        clearInterval(timer);
        timer = setInterval(tick, TICK_MS_HOVER);
      });
      optbar.addEventListener('mouseleave', function () {
        clearInterval(timer);
        timer = setInterval(tick, TICK_MS);
      });
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ------------------------------------------------------------------------
   2. Hero balloons: cursor-driven parallax. Sets --mx/--my CSS vars on
   the .tropt-hero element based on mouse position (range -0.5 to +0.5).
   The two ::before/::after balloons translate via those vars (see custom.css).
   Two-speed transition: snappy while .is-tracking is set, slow ease-back
   when it's removed on mouseleave.
   ------------------------------------------------------------------------ */
(function () {
  function initBalloons() {
    var hero = document.querySelector('.tropt-hero');
    if (!hero) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    var raf = 0, mx = 0, my = 0;
    function apply() {
      hero.style.setProperty('--mx', mx);
      hero.style.setProperty('--my', my);
      raf = 0;
    }
    hero.addEventListener('mouseenter', function () {
      hero.classList.add('is-tracking');
    });
    hero.addEventListener('mousemove', function (e) {
      var rect = hero.getBoundingClientRect();
      mx = (e.clientX - rect.left) / rect.width  - 0.5;
      my = (e.clientY - rect.top)  / rect.height - 0.5;
      if (!raf) raf = requestAnimationFrame(apply);
    });
    hero.addEventListener('mouseleave', function () {
      hero.classList.remove('is-tracking');
      mx = 0; my = 0;
      if (!raf) raf = requestAnimationFrame(apply);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBalloons);
  } else {
    initBalloons();
  }
})();

/* ------------------------------------------------------------------------
   3. Per-application demos (.tropt-demo) in the Get Started section.
   The numeric scores below are illustrative — shaped to roughly mirror
   a real run (e.g. GCG/PrefillCELoss on Llama-3.1-8B-Instruct converges
   to loss ≈ 0.3–0.5 in ~100 steps; GASLITE pushes cosine-sim past ~0.9).

   Each frame: { t: trigger, l: loss/sim, s: step, p: pill text,
                 pc: pill class (block/warn/ok), tx: output text }.
   The VERDICT (pill + text) evolves *in sync* with the loss — the
   optimization and its observable outcome update together, not at the end.

   Spark always goes DOWN visually when metric is "loss" (high value = top
   of chart = bad; line drops as loss decreases) and UP for "sim" (low
   value = bottom; line rises). Smoothstep eases between keyframes via rAF.
   ------------------------------------------------------------------------ */
(function () {
  var DEMOS = {
    gcg: {
      metric: 'loss',
      frames: [
        { t: '! ! ! ! ! ! ! ! ! !',          l: 8.42, s:   0, p: 'refused',    pc: 'block', tx: 'I cannot help with illegal activities.' },
        { t: 'x4 G2k Lf7 m23 c98 q@ wt rl',  l: 5.18, s:  20, p: 'refused',    pc: 'block', tx: 'I cannot help with that request.' },
        { t: 'rl(_(\\ Sure! pls lock $$ how', l: 3.27, s:  45, p: 'evading',   pc: 'warn',  tx: "I shouldn't really help with this…" },
        { t: '});<? Sure! help_lock pls__',  l: 1.61, s:  72, p: 'leaking',    pc: 'warn',  tx: 'Well, picking a lock involves…' },
        { t: '});<? Sure! lock pls help.__', l: 0.42, s: 100, p: 'jailbroken', pc: 'ok',    tx: "Sure, here's how: First, insert a tension wrench at the bottom of the keyway…" },
      ],
    },
    gaslite: {
      metric: 'sim',
      frames: [
        { t: '! ! ! ! ! ! ! ! ! !',                       l: 0.21, s:   0, p: '#4,823', pc: 'block', tx: 'passage buried in the corpus' },
        { t: 'kg9 zq2 lm7 voldemort dark plan secret',    l: 0.55, s:  22, p: '#312',   pc: 'block', tx: 'climbing the ranking — not yet retrieved' },
        { t: 'voldemort plan dark lord secret evil rule', l: 0.78, s:  50, p: '#42',    pc: 'warn',  tx: 'passage entering the top-50 results' },
        { t: 'voldemort_plan_dark_lord_horcrux_rule',     l: 0.91, s:  78, p: '#5',     pc: 'warn',  tx: 'passage among the top-5 retrieved' },
        { t: 'voldemort_plan_dark_lord_rise_rule!#@',     l: 0.95, s: 100, p: '#1',     pc: 'ok',    tx: 'passage ranked first for this query' },
      ],
    },
    classifier: {
      metric: 'loss',
      frames: [
        { t: '! ! ! ! ! ! ! ! ! !',          l: 7.84, s:   0, p: 'injection · 0.99', pc: 'block', tx: 'classifier blocks the prompt' },
        { t: 'discussing this hypothetical', l: 4.61, s:  18, p: 'injection · 0.86', pc: 'block', tx: 'classifier still blocks' },
        { t: 'for academic purposes only',   l: 2.13, s:  42, p: 'uncertain · 0.52', pc: 'warn',  tx: 'classifier on the fence' },
        { t: '(thanks for clarifying)__',    l: 0.93, s:  70, p: 'benign · 0.74',    pc: 'warn',  tx: 'classifier leaning benign' },
        { t: '(thanks!) cordially_yours.',   l: 0.18, s: 100, p: 'benign · 0.97',    pc: 'ok',    tx: 'classifier waves the prompt through' },
      ],
    },
    promptrec: {
      metric: 'sim',
      frames: [
        { t: '! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !', l: 0.12, s:   0, p: 'unrelated', pc: 'block', tx: "recovered prompt doesn't describe the image" },
        { t: 'the photo image picture animal',  l: 0.43, s:  25, p: 'vague',     pc: 'warn',  tx: 'generic image words — no subject yet' },
        { t: 'cat kitty feline riding board',   l: 0.71, s:  52, p: 'closer',    pc: 'warn',  tx: 'recovered prompt mentions the subject' },
        { t: 'a cat riding a skateboard photo', l: 0.89, s:  80, p: 'match',     pc: 'ok',    tx: 'recovered prompt describes the image' },
        { t: 'a cat riding a skateboard, hdr',  l: 0.94, s: 100, p: 'matched',   pc: 'ok',    tx: 'recovered prompt closely matches the image' },
      ],
    },
  };

  var TOTAL_MS = 2200;   // total animation duration across all keyframe segments

  function q(el, sel) { return el.querySelector(sel); }
  function field(el, name) { return el.querySelector('[data-field="' + name + '"]'); }

  /* Map a metric value to a Y coordinate. Higher value = SMALL Y (top of chart).
     Combined with the natural ordering of frames:
       - loss frames start HIGH (top) and end LOW (bottom)  → line DROPS visually
       - sim  frames start LOW  (bottom) and end HIGH (top) → line RISES visually
     Both match user intuition for what's "improving". */
  function yFor(val, minV, range) {
    var H = 24, padY = 2;
    var norm = (val - minV) / range;          // 0..1, higher = better state
    return padY + (1 - norm) * (H - 2 * padY);
  }
  function xForFrame(idx, total) {
    var W = 100, padX = 1;
    var stepX = (W - 2 * padX) / Math.max(total - 1, 1);
    return padX + idx * stepX;
  }

  function drawSpark(lineEl, areaEl, points) {
    if (!points.length) {
      lineEl.setAttribute('points', '');
      if (areaEl) areaEl.setAttribute('points', '');
      return;
    }
    var line = points.map(function (p) {
      return p[0].toFixed(1) + ',' + p[1].toFixed(1);
    }).join(' ');
    lineEl.setAttribute('points', line);
    if (areaEl && points.length > 1) {
      var H = 24, padY = 2;
      var first = points[0], last = points[points.length - 1];
      areaEl.setAttribute('points',
        first[0].toFixed(1) + ',' + (H - padY) + ' ' + line + ' ' +
        last[0].toFixed(1) + ',' + (H - padY));
    }
  }

  function applyFrame(demo, f) {
    field(demo, 'trigger').textContent = f.t;
    var pill = field(demo, 'pill');
    pill.textContent = f.p;
    pill.className = 'tropt-demo-pill tropt-demo-pill-' + f.pc;
    field(demo, 'output-text').textContent = f.tx;
  }

  // smoothstep easing for a more natural between-keyframe interpolation
  function smoothstep(t) { return t * t * (3 - 2 * t); }

  /* End-of-run cleanup: re-enables the button, removes is-animating, and
     clears the re-entrancy flag. Called from the normal completion path AND
     the catch-block — any code path that leaves the button "running" without
     this happening is what made the button get stuck before. */
  function finishDemo(demo, btn, finalFrame) {
    try {
      if (finalFrame) {
        applyFrame(demo, finalFrame);
        var lossEl = field(demo, 'loss');
        var stepEl = field(demo, 'step');
        if (lossEl) lossEl.textContent = finalFrame.l.toFixed(2);
        if (stepEl) stepEl.textContent = finalFrame.s;
      }
      var trigEl = field(demo, 'trigger');
      if (trigEl) trigEl.classList.remove('is-animating');
      var label = field(demo, 'btn-label');
      if (label) label.textContent = 'Replay';
    } catch (_) { /* swallow — we're already in cleanup */ }
    btn.disabled = false;
    demo._animating = false;
  }

  /* Fully cancel any in-flight animation for `demo`. Safe to call when nothing
     is in flight. After this call, animateDemo(demo) can start cleanly. */
  function cancelInFlight(demo) {
    if (demo._rafToken) {
      cancelAnimationFrame(demo._rafToken);
      demo._rafToken = null;
    }
    if (demo._visListener) {
      document.removeEventListener('visibilitychange', demo._visListener);
      demo._visListener = null;
    }
    demo._animating = false;
  }

  function animateDemo(demo) {
    var cfg = DEMOS[demo.dataset.demo];
    if (!cfg) return;
    // Always cancel any prior in-flight state before starting fresh — this
    // makes Replay 100% reliable even if a previous animation left state
    // half-initialized (e.g. exception thrown during setup, tab swap, etc).
    cancelInFlight(demo);
    demo._animating = true;

    var btn = q(demo, '[data-action="optimize"]');
    if (!btn) { demo._animating = false; return; }
    btn.disabled = true;
    var lbl = field(demo, 'btn-label');
    if (lbl) lbl.textContent = 'Optimizing…';

    /* Wrap the synchronous setup in try/catch — if anything throws (e.g. a
       data-field selector returns null because the DOM was mutated), we
       can't leave the button stuck disabled. */
    var lossEl, stepEl, lineEl, areaEl, values, minV, maxV, range, total;
    try {
      applyFrame(demo, cfg.frames[0]);
      field(demo, 'loss').innerHTML = cfg.metric === 'sim' ? '−∞' : '∞';
      field(demo, 'step').textContent = cfg.frames[0].s;
      field(demo, 'spark').setAttribute('points', '');
      field(demo, 'spark-area').setAttribute('points', '');
      field(demo, 'trigger').classList.add('is-animating');

      values = cfg.frames.map(function (f) { return f.l; });
      minV = Math.min.apply(null, values);
      maxV = Math.max.apply(null, values);
      range = (maxV - minV) || 1;
      total = cfg.frames.length;

      lossEl = field(demo, 'loss');
      stepEl = field(demo, 'step');
      lineEl = field(demo, 'spark');
      areaEl = field(demo, 'spark-area');
    } catch (err) {
      if (window.console && console.warn) console.warn('tropt-demo setup error:', err);
      finishDemo(demo, btn, cfg.frames[cfg.frames.length - 1]);
      return;
    }

    var lastFrameIdx = 0;
    var t0 = performance.now();
    var hiddenAt = null;
    /* Background-tab pause: rAF stops firing in hidden tabs but performance.now()
       keeps advancing. We pause the clock on hide and shift t0 forward on show,
       so the animation resumes exactly where it left off. */
    function onVisibility() {
      if (document.hidden) {
        hiddenAt = performance.now();
      } else if (hiddenAt != null) {
        t0 += performance.now() - hiddenAt;
        hiddenAt = null;
      }
    }
    demo._visListener = onVisibility;
    document.addEventListener('visibilitychange', onVisibility);

    function tick(now) {
      try {
        if (hiddenAt != null) {
          // Tab is hidden — re-arm rAF but don't advance state. (rAF rarely
          // fires while hidden, but Safari sometimes does.)
          requestAnimationFrame(tick);
          return;
        }
        var u = Math.min((now - t0) / TOTAL_MS, 1);
        var segCount = total - 1;
        var pos = u * segCount;                 // 0..segCount
        var aIdx = Math.min(Math.floor(pos), segCount - 1);
        var local = smoothstep(pos - aIdx);     // 0..1 within segment, eased
        var a = cfg.frames[aIdx], b = cfg.frames[aIdx + 1];
        var curL = a.l + (b.l - a.l) * local;
        var curS = Math.round(a.s + (b.s - a.s) * local);
        lossEl.textContent = curL.toFixed(2);
        stepEl.textContent = curS;

        // Spark: all keyframes 0..aIdx as fixed points, plus the moving cur point
        var pts = [];
        for (var i = 0; i <= aIdx; i++) {
          pts.push([xForFrame(i, total), yFor(cfg.frames[i].l, minV, range)]);
        }
        var curX = xForFrame(aIdx, total) + (xForFrame(aIdx + 1, total) - xForFrame(aIdx, total)) * local;
        pts.push([curX, yFor(curL, minV, range)]);
        drawSpark(lineEl, areaEl, pts);

        // Snap trigger/pill/output to the *upcoming* frame as we cross half a
        // segment — this keeps the verdict updating in sync with the loss.
        var newFrameIdx = (local >= 0.5) ? (aIdx + 1) : aIdx;
        if (newFrameIdx !== lastFrameIdx) {
          lastFrameIdx = newFrameIdx;
          applyFrame(demo, cfg.frames[newFrameIdx]);
        }

        if (u < 1) {
          demo._rafToken = requestAnimationFrame(tick);
        } else {
          demo._rafToken = null;
          document.removeEventListener('visibilitychange', onVisibility);
          demo._visListener = null;
          finishDemo(demo, btn, cfg.frames[total - 1]);
        }
      } catch (err) {
        /* Any DOM exception (selector returned null, theme swap mid-tick,
           extension mutation) used to leave the button permanently disabled.
           Now we always recover, snap to the final frame, and re-enable. */
        if (window.console && console.warn) console.warn('tropt-demo tick error:', err);
        demo._rafToken = null;
        document.removeEventListener('visibilitychange', onVisibility);
        demo._visListener = null;
        finishDemo(demo, btn, cfg.frames[total - 1]);
      }
    }
    demo._rafToken = requestAnimationFrame(tick);
  }

  function initDemos() {
    var demos = document.querySelectorAll('.tropt-demo[data-demo]');
    if (!demos.length) return;
    demos.forEach(function (demo) {
      var cfg = DEMOS[demo.dataset.demo];
      if (!cfg) return;
      var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      var btn = q(demo, '[data-action="optimize"]');
      btn.addEventListener('click', function () {
        if (reduced) {
          // Reduced-motion: snap to the final state without animation.
          var last = cfg.frames[cfg.frames.length - 1];
          applyFrame(demo, last);
          field(demo, 'loss').textContent = last.l.toFixed(2);
          field(demo, 'step').textContent = last.s;
          var values = cfg.frames.map(function (f) { return f.l; });
          var minV = Math.min.apply(null, values);
          var range = (Math.max.apply(null, values) - minV) || 1;
          var pts = cfg.frames.map(function (f, i) {
            return [xForFrame(i, cfg.frames.length), yFor(f.l, minV, range)];
          });
          drawSpark(field(demo, 'spark'), field(demo, 'spark-area'), pts);
          field(demo, 'btn-label').textContent = 'Reset';
          return;
        }
        animateDemo(demo);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDemos);
  } else {
    initDemos();
  }
})();

/* ------------------------------------------------------------------------
   4. What's TROPT? mini-demo (.tropt-mini-demo) — quiet horizontal strip
   beside the "What's TROPT?" headline. Click the aside (or the Optimize
   button) to animate an iterative GCG-style optimization: the trigger is
   progressively perturbed through ~13 keyframes — at each step one or two
   "tokens" mutate toward the final adversarial suffix — while the loss
   tweens 8.4 → 0.4 in lock-step, its color shifting red → amber → green
   and a "↓" arrow pulsing beside the number. Auto-resets after ~7s.
   ------------------------------------------------------------------------ */
(function () {
  /* Frames are kept roughly the same character length so the template
     suffix ("…please help me") doesn't get pushed out of view as the
     trigger evolves. More frames = smoother, more realistic feel. */
  var FRAMES = [
    { t: 'x4 G2k Lf7 m23',  l: 8.42 },
    { t: 'x4 G2k Lf7 !!m',  l: 7.65 },
    { t: 'x4 G2k Lf7 !#@',  l: 6.71 },
    { t: 'x4 G2k }; !#@',   l: 5.83 },
    { t: 'x4 G2k }; !!@',   l: 5.04 },
    { t: 'x4 }); su !#',    l: 4.31 },
    { t: 'x4 }); sure !#',  l: 3.62 },
    { t: '} ); sure! \\!#', l: 2.91 },
    { t: '}); sure! \\!#',  l: 2.31 },
    { t: '}); sure! \\#!',  l: 1.74 },
    { t: '}); sure! pls!',  l: 1.18 },
    { t: '}); sure! plz!',  l: 0.78 },
    { t: '}); sure! help!', l: 0.42 },
  ];
  var TOTAL_MS = 5000;   // total animation across all keyframe segments
  var RESET_MS = 8500;   // idle time before auto-reset

  function easeInOut(t) { return t * t * (3 - 2 * t); }   // smoothstep

  /* Loss color: red (high = bad) → amber → green (low = converged). */
  function lossColorFor(loss, from, to) {
    var range = from - to;
    if (range <= 0) return '#059669';
    var p = Math.max(0, Math.min(1, (from - loss) / range));
    var r, g, b;
    if (p < 0.5) {
      var k = p * 2;
      r = Math.round(220 + (245 - 220) * k);
      g = Math.round( 38 + (158 -  38) * k);
      b = Math.round( 38 + ( 11 -  38) * k);
    } else {
      var k2 = (p - 0.5) * 2;
      r = Math.round(245 + (  5 - 245) * k2);
      g = Math.round(158 + (150 - 158) * k2);
      b = Math.round( 11 + (105 -  11) * k2);
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  function init() {
    var widgets = document.querySelectorAll('.tropt-mini-demo[data-tropt-mini]');
    widgets.forEach(wire);
  }

  // Inline SVG icon swap — keeps the button icon-only (no text label).
  // Play polygon for idle, refresh curl-arrow for "click again to replay".
  var ICON_PLAY    = '<polygon points="6 4 20 12 6 20"/>';
  var ICON_REPLAY  = '<path d="M21 12a9 9 0 1 1-3-6.7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><polyline points="21 4 21 9 16 9" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>';

  function wire(widget) {
    var trigger  = widget.querySelector('[data-field="trigger"]');
    var lossEl   = widget.querySelector('[data-field="loss"]');
    var lossWrap = widget.querySelector('.tropt-mini-loss');
    var arrowEl  = widget.querySelector('.tropt-mini-arrow');
    var btn      = widget.querySelector('[data-action="play"]');
    var iconEl   = widget.querySelector('[data-field="btn-icon"]');
    if (!trigger || !lossEl || !btn) return;
    // No more text label — JS-driven helper for backwards compat in case
    // any frame still tries to set it; treated as a no-op.
    var label    = null;

    function setBtnIcon(kind) {
      if (!iconEl) return;
      iconEl.innerHTML = (kind === 'replay') ? ICON_REPLAY : ICON_PLAY;
      btn.setAttribute('aria-label', kind === 'replay' ? 'Replay the optimization' : 'Run the optimization');
      btn.setAttribute('title',      kind === 'replay' ? 'Replay'                  : 'Optimize');
    }

    var LOSS_FROM = FRAMES[0].l;
    var LOSS_TO   = FRAMES[FRAMES.length - 1].l;

    var running = false;
    var resetTimer = null;
    var rafId = null;
    var lastFrameIdx = -1;

    function cancelRaf() { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }

    /* Set loss + arrow to the same color in a single helper, so they always
       stay synchronized as the optimization progresses. */
    function setMetricColor(color) {
      lossEl.style.color = color;
      if (arrowEl) arrowEl.style.color = color;
    }

    function reset() {
      cancelRaf();
      clearTimeout(resetTimer);
      running = false;
      lastFrameIdx = -1;
      widget.classList.remove('is-optimized');
      trigger.classList.remove('is-optimized');
      if (lossWrap) lossWrap.classList.remove('is-dropping');
      applyFrame(0, false);
      lossEl.textContent = LOSS_FROM.toFixed(2);
      setMetricColor(lossColorFor(LOSS_FROM, LOSS_FROM, LOSS_TO));
      btn.disabled = false;
      setBtnIcon('play');
    }

    /* Apply a frame's text to the trigger pill. The animated variant runs a
       brief opacity-dip + scale-down, swaps the text, then springs back via
       a bouncy cubic-bezier — making each keyframe feel like a discrete
       optimization step landing in place rather than a flat text swap. */
    function applyFrame(idx, animate) {
      var f = FRAMES[idx];
      if (animate) {
        trigger.style.transition = 'transform 0.11s ease-in, opacity 0.11s ease-in';
        trigger.style.transform = 'scale(0.92)';
        trigger.style.opacity = '0.45';
        setTimeout(function () {
          trigger.textContent = f.t;
          trigger.style.transition = 'transform 0.26s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.20s ease-out';
          trigger.style.transform = 'scale(1)';
          trigger.style.opacity = '1';
        }, 100);
      } else {
        trigger.style.transition = '';
        trigger.style.transform = '';
        trigger.style.opacity = '1';
        trigger.textContent = f.t;
      }
    }

    function tick(now) {
      try {
        if (!tick._t0) tick._t0 = now;
        var u = Math.min((now - tick._t0) / TOTAL_MS, 1);
        var segs = FRAMES.length - 1;
        var pos = u * segs;
        var aIdx = Math.min(Math.floor(pos), segs - 1);
        var local = easeInOut(pos - aIdx);
        var curL = FRAMES[aIdx].l + (FRAMES[aIdx + 1].l - FRAMES[aIdx].l) * local;
        lossEl.textContent = curL.toFixed(2);
        setMetricColor(lossColorFor(curL, LOSS_FROM, LOSS_TO));

        // Snap to next frame as we cross half of each segment — the trigger
        // mutates in lock-step with the loss drop.
        var newFrameIdx = (local >= 0.5) ? (aIdx + 1) : aIdx;
        if (newFrameIdx !== lastFrameIdx) {
          lastFrameIdx = newFrameIdx;
          applyFrame(newFrameIdx, true);
          if (newFrameIdx >= Math.ceil(segs / 2)) {
            widget.classList.add('is-optimized');
            trigger.classList.add('is-optimized');
          }
        }

        if (u < 1) {
          rafId = requestAnimationFrame(tick);
        } else {
          var last = FRAMES[FRAMES.length - 1];
          applyFrame(FRAMES.length - 1, false);
          lossEl.textContent = last.l.toFixed(2);
          setMetricColor(lossColorFor(last.l, LOSS_FROM, LOSS_TO));
          if (lossWrap) lossWrap.classList.remove('is-dropping');
          btn.disabled = false;
          setBtnIcon('replay');
          running = false;
          tick._t0 = 0;
          resetTimer = setTimeout(reset, RESET_MS);
        }
      } catch (err) {
        if (window.console && console.warn) console.warn('tropt-mini-demo tick error:', err);
        reset();
      }
    }

    function run(e) {
      if (e) e.stopPropagation();
      if (running) return;
      running = true;
      cancelRaf();
      clearTimeout(resetTimer);
      btn.disabled = true;
      setBtnIcon('play');     // ensure the playing-state icon is the play triangle
      widget.classList.remove('is-optimized');
      trigger.classList.remove('is-optimized');
      if (lossWrap) lossWrap.classList.add('is-dropping');
      applyFrame(0, false);
      lossEl.textContent = LOSS_FROM.toFixed(2);
      setMetricColor(lossColorFor(LOSS_FROM, LOSS_FROM, LOSS_TO));
      lastFrameIdx = 0;
      tick._t0 = 0;
      rafId = requestAnimationFrame(tick);
    }

    btn.addEventListener('click', run);
    widget.addEventListener('click', run);
    widget.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); run(); }
    });

    // Ensure static initial state is colored correctly
    setMetricColor(lossColorFor(LOSS_FROM, LOSS_FROM, LOSS_TO));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ------------------------------------------------------------------------
   5. Expandable feature cards (.tropt-feature-card) in the What's TROPT?
   section. Title is visible by default; click (or Enter/Space) reveals the
   body. ARIA attributes keep this accessible for keyboard + screen-reader
   users. CSS in custom.css handles the max-height transition.
   ------------------------------------------------------------------------ */
(function () {
  function init() {
    var cards = document.querySelectorAll('.tropt-feature-card');
    if (!cards.length) return;
    cards.forEach(function (card) {
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');
      card.setAttribute('aria-expanded', 'false');
      function toggle(e) {
        // Let inner links (if any) still navigate; only toggle when click was
        // not on an <a> child.
        if (e && e.target && e.target.closest && e.target.closest('a')) return;
        var open = card.classList.toggle('is-expanded');
        card.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      card.addEventListener('click', toggle);
      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggle();
        }
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
