/* ════════════════════════════════════════════════════════════════════
   IndiaFinBench — main.js
   Chart, tables, canvas, scroll choreography, live RAG, explorer, submit.
   Vanilla JS, no dependencies.
   ════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var MODELS = window.IFB_MODELS || [];
  var HUMAN = window.IFB_HUMAN || null;
  var CLAUDE = window.IFB_CLAUDE || null;
  var DIFF = window.IFB_DIFF || [];
  var TASKS = window.IFB_TASKS || [];
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var $ = function (s, c) { return (c || document).querySelector(s); };
  var $$ = function (s, c) { return Array.prototype.slice.call((c || document).querySelectorAll(s)); };

  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  };

  /* ── Deep links: jump instantly on load instead of animating from top ─ */
  if (location.hash) {
    var deepTarget = document.getElementById(location.hash.slice(1));
    if (deepTarget) {
      requestAnimationFrame(function () {
        deepTarget.scrollIntoView({ behavior: 'instant', block: 'start' });
      });
    }
  }

  /* ── Masthead drawer ──────────────────────────────────────────────── */
  var menuBtn = $('#menuBtn');
  var drawer = $('#menuDrawer');
  if (menuBtn && drawer) {
    menuBtn.addEventListener('click', function () {
      var open = drawer.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', String(open));
    });
    $$('a', drawer).forEach(function (a) {
      a.addEventListener('click', function () {
        drawer.classList.remove('open');
        menuBtn.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ── Scroll reveal ────────────────────────────────────────────────── */
  var revealIO = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('shown');
        revealIO.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  function watch(el) { if (el) revealIO.observe(el); }
  $$('.reveal').forEach(watch);

  /* ── Count-up numbers ─────────────────────────────────────────────── */
  function countUp(el) {
    var target = parseFloat(el.dataset.count);
    var dec = parseInt(el.dataset.dec || '0', 10);
    var suffix = el.dataset.suffix || '';
    if (REDUCED) { el.textContent = target.toFixed(dec) + suffix; return; }
    var t0 = null, DUR = 1400;
    function frame(t) {
      if (!t0) t0 = t;
      var p = Math.min((t - t0) / DUR, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  var countIO = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { countUp(e.target); countIO.unobserve(e.target); }
    });
  }, { threshold: 0.4 });
  $$('[data-count]').forEach(function (el) { countIO.observe(el); });

  /* ── Progress rail ────────────────────────────────────────────────── */
  var railFill = $('#railFill');
  var railItems = $$('.rail-list li');
  var chapterIds = railItems.map(function (li) { return li.dataset.ch; });

  window.addEventListener('scroll', function () {
    if (!railFill) return;
    var h = document.documentElement.scrollHeight - window.innerHeight;
    railFill.style.height = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
  }, { passive: true });

  var chapterIO = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        railItems.forEach(function (li) {
          li.classList.toggle('on', li.dataset.ch === e.target.id);
        });
        $$('.mast-nav a').forEach(function (a) {
          a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id);
        });
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  chapterIds.forEach(function (id) {
    var sec = document.getElementById(id);
    if (sec) chapterIO.observe(sec);
  });

  /* ── §02 Corpus: 192 document marks ───────────────────────────────── */
  (function docDots() {
    var holder = $('#docDots');
    if (!holder) return;
    var frag = document.createDocumentFragment();
    for (var i = 0; i < 192; i++) {
      var el = document.createElement('i');
      el.className = i < 92 ? 'sebi' : 'rbi';
      el.style.setProperty('--d', (i * 7) + 'ms');
      frag.appendChild(el);
    }
    holder.appendChild(frag);
    watchShown(holder);
  })();

  function watchShown(el) {
    new IntersectionObserver(function (entries, io) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('shown'); io.unobserve(e.target); }
      });
    }, { threshold: 0.25 }).observe(el);
  }

  /* ── §03 Task cards ───────────────────────────────────────────────── */
  (function taskCards() {
    var grid = $('#taskGrid');
    if (!grid || !TASKS.length) return;
    grid.innerHTML = TASKS.map(function (t) {
      var share = (t.n / 406) * 100;
      return '<article class="task-card reveal">' +
        '<span class="task-code" style="color:' + t.color + '">' + esc(t.code) + '</span>' +
        '<h3>' + esc(t.label) + '</h3>' +
        '<p>' + esc(t.desc) + '</p>' +
        '<div class="task-n"><b>' + t.n + '</b> items · ' + share.toFixed(1) + '% of benchmark</div>' +
        '<div class="task-meter"><i style="--w:' + share.toFixed(1) + '%;background:' + t.color + '"></i></div>' +
        '</article>';
    }).join('');
    $$('.task-card', grid).forEach(function (card) { watch(card); watchShown(card); });
  })();

  /* ── §03 difficulty strip + diff segments need .shown ─────────────── */
  $$('.diff-strip, .rag-diagram').forEach(watchShown);

  /* ── §04 Chart ────────────────────────────────────────────────────── */
  var METRIC_META = {
    overall: { title: 'Overall accuracy', desc: 'All 406 items · zero-shot · 95% Wilson confidence intervals on hover' },
    reg: { title: 'Regulatory Interpretation', desc: '174 items · precision reading of rules, thresholds and applicability' },
    num: { title: 'Numerical Reasoning', desc: '92 items · arithmetic over figures embedded in regulatory prose' },
    con: { title: 'Contradiction Detection', desc: '62 items · do two passages conflict on the stated issue?' },
    tmp: { title: 'Temporal Reasoning', desc: '78 items · supersession chains and operative-date resolution' }
  };

  var chartEl = $('#barChart');
  var chartShown = false;

  function tipHTML(m) {
    return '<b>' + esc(m.label) + '</b>' +
      '<span class="tt-grid">' +
      '<i>REG</i><span>' + m.reg.toFixed(1) + '%</span>' +
      '<i>NUM</i><span>' + m.num.toFixed(1) + '%</span>' +
      '<i>CON</i><span>' + m.con.toFixed(1) + '%</span>' +
      '<i>TMP</i><span>' + m.tmp.toFixed(1) + '%</span>' +
      '<i>Overall</i><span>' + m.overall.toFixed(1) + '%</span>' +
      '</span>' +
      (m.ci ? '<span class="tt-ci">95% CI ' + esc(m.ci) + ' · n = ' + m.n_items + '</span>' : '');
  }

  function buildChart(metric) {
    if (!chartEl) return;
    var rows = MODELS.slice();
    if (HUMAN) rows.push(HUMAN);
    rows.sort(function (a, b) { return b[metric] - a[metric]; });

    var html = rows.map(function (m, idx) {
      var v = m[metric];
      var cls = m.is_human ? 'human' : ('t' + (m.tier || 2));
      return '<div class="crow ' + cls + '">' +
        '<div class="crow-label">' +
        '<span class="crow-rank">' + (m.is_human ? '—' : (idx + 1)) + '</span>' +
        '<span class="crow-name' + (m.is_human ? ' human' : '') + '">' + esc(m.label) + '</span>' +
        '</div>' +
        '<div class="crow-track">' +
        '<div class="crow-fill" data-w="' + v + '"><span class="crow-val">' + v.toFixed(1) + '%</span></div>' +
        '<div class="crow-tip" role="tooltip">' + tipHTML(m) + '</div>' +
        '</div>' +
        '</div>';
    }).join('');
    html += '<div class="chart-baseline" id="chartBaseline"><em>human baseline · ' +
      (HUMAN ? HUMAN[metric].toFixed(1) : '—') + '%</em></div>';
    chartEl.innerHTML = html;

    function animate() {
      $$('.crow-fill', chartEl).forEach(function (f) {
        f.style.width = f.dataset.w + '%';
        f.classList.add('shown');
      });
      positionBaseline(metric);
    }
    if (chartShown) {
      requestAnimationFrame(function () { requestAnimationFrame(animate); });
    } else {
      new IntersectionObserver(function (entries, io) {
        if (entries[0].isIntersecting) {
          chartShown = true;
          animate();
          io.disconnect();
        }
      }, { threshold: 0.15 }).observe(chartEl);
    }
  }

  function positionBaseline(metric) {
    var line = $('#chartBaseline');
    var track = $('.crow-track', chartEl);
    if (!line || !track || !HUMAN) return;
    var v = HUMAN[metric];
    var left = track.offsetLeft + (track.offsetWidth * v / 100);
    line.style.left = left + 'px';
  }
  window.addEventListener('resize', function () {
    var active = $('.tab.active');
    if (chartShown) positionBaseline(active ? active.dataset.t : 'overall');
  });

  $$('#taskTabs .tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      $$('#taskTabs .tab').forEach(function (t) {
        t.classList.toggle('active', t === tab);
        t.setAttribute('aria-selected', String(t === tab));
      });
      var k = tab.dataset.t;
      $('#chartTitle').textContent = METRIC_META[k].title;
      $('#chartDesc').textContent = METRIC_META[k].desc;
      buildChart(k);
    });
  });
  buildChart('overall');

  /* ── §04 Results table ────────────────────────────────────────────── */
  var sortKey = 'overall', sortAsc = false;

  function scoreCell(v, isBest) {
    var cls = v >= 85 ? 's-hi' : v >= 70 ? 's-md' : 's-lo';
    return '<td class="c"><span class="score ' + cls + (isBest ? ' score-best' : '') + '">' + v.toFixed(1) + '</span></td>';
  }

  function buildTable() {
    var body = $('#tBody');
    if (!body) return;
    var rows = MODELS.slice();
    rows.sort(function (a, b) {
      var x = a[sortKey], y = b[sortKey];
      if (sortKey === 'params') { x = parseFloat(x) || 0; y = parseFloat(y) || 0; }
      if (sortKey === 'rank') return sortAsc ? a.rank - b.rank : b.rank - a.rank;
      return sortAsc ? x - y : y - x;
    });

    var best = {};
    ['reg', 'num', 'con', 'tmp', 'overall'].forEach(function (k) {
      best[k] = Math.max.apply(null, MODELS.map(function (m) { return m[k]; }));
    });

    var html = rows.map(function (m) {
      return '<tr>' +
        '<td class="c rank-cell' + (m.rank === 1 ? ' rank-1' : '') + '">' + m.rank + '</td>' +
        '<td><div class="model-cell-name">' + esc(m.label) + '</div><div class="model-cell-id">' + esc(m.hf_id) + '</div></td>' +
        '<td class="mono">' + esc(m.params) + '</td>' +
        '<td><span class="access-tag">' + esc(m.type) + '</span></td>' +
        scoreCell(m.reg, m.reg === best.reg) +
        scoreCell(m.num, m.num === best.num) +
        scoreCell(m.con, m.con === best.con) +
        scoreCell(m.tmp, m.tmp === best.tmp) +
        scoreCell(m.overall, m.overall === best.overall) +
        '<td class="c ci-cell">' + esc(m.ci) + '</td>' +
        '</tr>';
    }).join('');

    if (HUMAN) {
      html += '<tr class="tr-human">' +
        '<td class="c rank-cell">—</td>' +
        '<td><div class="model-cell-name">' + esc(HUMAN.label) + '</div><div class="model-cell-id">' + esc(HUMAN.hf_id) + '</div></td>' +
        '<td class="mono">—</td><td><span class="access-tag">Human baseline</span></td>' +
        scoreCell(HUMAN.reg) + scoreCell(HUMAN.num) + scoreCell(HUMAN.con) + scoreCell(HUMAN.tmp) + scoreCell(HUMAN.overall) +
        '<td class="c ci-cell">' + esc(HUMAN.ci) + '</td></tr>';
    }
    if (CLAUDE) {
      html += '<tr class="tr-subset">' +
        '<td class="c rank-cell">†</td>' +
        '<td><div class="model-cell-name">' + esc(CLAUDE.label.replace('†', '')) + '</div><div class="model-cell-id">' + esc(CLAUDE.hf_id) + '</div></td>' +
        '<td class="mono">—</td><td><span class="access-tag">150-item subset</span></td>' +
        scoreCell(CLAUDE.reg) + scoreCell(CLAUDE.num) + scoreCell(CLAUDE.con) + scoreCell(CLAUDE.tmp) + scoreCell(CLAUDE.overall) +
        '<td class="c ci-cell">' + esc(CLAUDE.ci) + '</td></tr>';
    }
    body.innerHTML = html;
  }

  $$('#resultsTable th.sortable').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.dataset.k;
      if (sortKey === k) sortAsc = !sortAsc;
      else { sortKey = k; sortAsc = (k === 'rank'); }
      $$('#resultsTable th').forEach(function (h) { h.classList.toggle('sorted', h === th); });
      buildTable();
    });
  });
  buildTable();

  /* ── §04 Difficulty table ─────────────────────────────────────────── */
  (function diffTable() {
    var body = $('#diffBody');
    if (!body || !DIFF.length) return;
    body.innerHTML = DIFF.map(function (m) {
      var d = m.hard - m.easy;
      var sign = d >= 0 ? '+' : '−';
      var cls = d >= 0 ? 'delta-up' : 'delta-down';
      return '<tr>' +
        '<td><div class="model-cell-name">' + esc(m.label) + '</div></td>' +
        '<td class="c mono">' + m.easy.toFixed(1) + '%</td>' +
        '<td class="c mono">' + m.med.toFixed(1) + '%</td>' +
        '<td class="c mono">' + m.hard.toFixed(1) + '%</td>' +
        '<td class="c mono ' + cls + '">' + sign + Math.abs(d).toFixed(1) + '</td>' +
        '</tr>';
    }).join('');
  })();

  /* ── §06 Live RAG ─────────────────────────────────────────────────── */
  (function rag() {
    var input = $('#ragQ'), btn = $('#ragBtn');
    var out = $('#ragOut'), status = $('#ragStatus'), statusText = $('#ragStatusText');
    var answerEl = $('#ragAnswer'), sourcesEl = $('#ragSources');
    if (!input || !btn) return;

    var PHASES = ['Encoding query…', 'Searching 4,347 chunks…', 'Fusing dense and sparse ranks…', 'Drafting cited answer…'];
    var phaseTimer = null;

    function startPhases() {
      var i = 0;
      statusText.textContent = PHASES[0];
      phaseTimer = setInterval(function () {
        i = Math.min(i + 1, PHASES.length - 1);
        statusText.textContent = PHASES[i];
      }, 1500);
    }
    function stopPhases() { clearInterval(phaseTimer); }

    function run() {
      var q = input.value.trim();
      if (!q) return;
      btn.disabled = true;
      out.hidden = false;
      status.hidden = false;
      answerEl.textContent = '';
      sourcesEl.innerHTML = '';
      startPhases();

      fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      }).then(function (r) { return r.json(); }).then(function (data) {
        stopPhases();
        status.hidden = true;
        if (data.error) {
          answerEl.textContent = '⚠ ' + data.error;
        } else {
          answerEl.textContent = data.answer || '';
          (data.sources || []).forEach(function (s, i) {
            var card = document.createElement('div');
            card.className = 'rag-src';
            card.innerHTML =
              '<div class="rag-src-no">Source ' + (i + 1) + ' · ' + esc(s.source || s.doc_id || '') + '</div>' +
              '<div class="rag-src-title">' + esc(s.title || s.doc_id || 'Document') + '</div>' +
              '<div class="rag-src-text">' + esc(s.text || '') + '</div>' +
              '<div class="rag-src-scores">' +
              '<span>RRF <b>' + (s.rrf_score != null ? Number(s.rrf_score).toFixed(4) : '—') + '</b></span>' +
              '<span>dense <b>' + (s.dense_score != null ? Number(s.dense_score).toFixed(3) : '—') + '</b></span>' +
              '<span>BM25 <b>' + (s.bm25_score != null ? Number(s.bm25_score).toFixed(2) : '—') + '</b></span>' +
              '</div>';
            sourcesEl.appendChild(card);
          });
        }
        btn.disabled = false;
      }).catch(function (e) {
        stopPhases();
        status.hidden = true;
        answerEl.textContent = '⚠ Network error: ' + e.message;
        btn.disabled = false;
      });
    }

    btn.addEventListener('click', run);
    input.addEventListener('keydown', function (e) { if (e.key === 'Enter') run(); });
    $$('.rag-examples .chip').forEach(function (chip) {
      chip.addEventListener('click', function () {
        input.value = chip.dataset.q || chip.textContent.trim();
        run();
      });
    });
  })();

  /* ── §07 Specimen explorer ────────────────────────────────────────── */
  (function explorer() {
    var btn = $('#exBtn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      btn.disabled = true;
      var task = encodeURIComponent($('#exTask').value);
      var diff = encodeURIComponent($('#exDiff').value);
      fetch('/api/example?task=' + task + '&diff=' + diff)
        .then(function (r) { return r.json(); })
        .then(function (q) {
          btn.disabled = false;
          var card = $('#exCard');
          if (q.error) {
            card.hidden = false;
            $('#exId').textContent = '';
            $('#exTaskBadge').textContent = '';
            $('#exDiffBadge').textContent = '';
            $('#exContext').textContent = q.error;
            $('#exQuestion').textContent = '';
            $('#exReveal').hidden = true;
            $('#exAnswer').hidden = true;
            return;
          }
          card.hidden = false;
          $('#exId').textContent = q.id || '';
          $('#exTaskBadge').textContent = q.task_type || '';
          $('#exDiffBadge').textContent = q.difficulty || '';
          $('#exContext').textContent = q.context || '';
          $('#exQuestion').textContent = q.question || '';
          $('#exReveal').hidden = false;
          $('#exAnswer').hidden = true;
          $('#exAnswerText').textContent = q.answer || '';
        })
        .catch(function () { btn.disabled = false; });
    });
    $('#exReveal').addEventListener('click', function () {
      $('#exAnswer').hidden = false;
      this.hidden = true;
    });
  })();

  /* ── §07 Submit ───────────────────────────────────────────────────── */
  (function submit() {
    var btn = $('#subBtn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var hfId = $('#hfId').value.trim();
      var box = $('#statusBox');
      box.hidden = false;
      box.className = 'status';
      if (!hfId || hfId.indexOf('/') < 1) {
        box.classList.add('err');
        box.textContent = 'Enter a valid HuggingFace model ID (org/model).';
        return;
      }
      btn.disabled = true;
      box.textContent = 'Preparing submission…';
      fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hf_id: hfId,
          label: $('#dispName').value.trim(),
          params: $('#modelParams').value.trim(),
          model_type: $('#mtype').value
        })
      }).then(function (r) { return r.json(); }).then(function (data) {
        btn.disabled = false;
        if (data.issue_url) {
          box.classList.add('ok');
          box.textContent = 'Submission issue opened in a new tab — submit it on GitHub and the model joins the evaluation queue.';
          window.open(data.issue_url, '_blank', 'noopener');
        } else {
          box.classList.add('err');
          box.textContent = data.error || 'Submission failed.';
        }
      }).catch(function (e) {
        btn.disabled = false;
        box.classList.add('err');
        box.textContent = 'Network error: ' + e.message;
      });
    });
  })();

  /* ── §07 Copy citation ────────────────────────────────────────────── */
  (function cite() {
    var btn = $('#copyBtn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText($('#citeText').textContent).then(function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      });
    });
  })();

})();
