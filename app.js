(function () {
  'use strict';

  // ========== CONFIGURATION ==========
  var BASE_PATH = detectBasePath();
  var POSITIONS = ['QB', 'RB', 'WR', 'TE'];
  var POS_COLORS = { QB: '#00f5ff', RB: '#a78bfa', WR: '#10b981', TE: '#fbbf24' };

  // ========== STATE ==========
  var allPlayers = [];
  var filteredPlayers = [];
  var modelMetadata = null;
  var scheduleImpact = null;
  var modelPerformance = null;
  var perfFiltered = [];
  var currentPosition = 'ALL';
  var currentSort = 'projection_points_total';
  var searchQuery = '';
  var perfPosition = 'ALL';
  var perfSearch = '';

  // ========== UTILITIES ==========
  function detectBasePath() {
    var path = window.location.pathname;
    var idx = path.indexOf('/nfl-player-projections/');
    if (idx !== -1) return path.substring(0, idx) + '/nfl-player-projections/';
    return './';
  }

  function escapeHtml(str) {
    if (!str) return '';
    var div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  }

  function formatFeatureName(name) {
    if (!name) return '';
    return name
      .replace(/_/g, ' ')
      .replace(/\bnorm\b/gi, '(normalized)')
      .replace(/\broll(\d+)\b/gi, 'rolling $1-wk')
      .replace(/\bmean\b/gi, 'avg')
      .replace(/\bstd\b/gi, 'volatility')
      .replace(/\blag(\d+)\b/gi, '$1-wk lag')
      .replace(/\bpct\b/gi, '%')
      .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function riskClass(score) {
    if (score == null) return '';
    if (score <= 30) return 'risk-low';
    if (score <= 60) return 'risk-med';
    return 'risk-high';
  }

  function posClass(pos) {
    return 'pos-' + (pos || '').toLowerCase();
  }

  function num(val, decimals) {
    if (val == null || isNaN(val)) return 'N/A';
    return Number(val).toFixed(decimals == null ? 1 : decimals);
  }

  function errorClass(error) {
    if (error == null) return '';
    var abs = Math.abs(error);
    if (abs <= 5) return 'risk-low';
    if (abs <= 15) return 'risk-med';
    return 'risk-high';
  }

  // ========== DATA LOADING ==========
  function fetchJSON(path) {
    return fetch(BASE_PATH + path).then(function (r) {
      if (!r.ok) throw new Error('Failed to load ' + path + ': ' + r.status);
      return r.json();
    });
  }

  function loadAllData() {
    return Promise.all([
      fetchJSON('data/players_QB.json'),
      fetchJSON('data/players_RB.json'),
      fetchJSON('data/players_WR.json'),
      fetchJSON('data/players_TE.json'),
      fetchJSON('data/draft_model_metadata.json'),
      fetchJSON('data/schedule_impact.json'),
      fetchJSON('data/model_performance.json').catch(function () { return null; }),
    ]).then(function (results) {
      var qb = results[0], rb = results[1], wr = results[2], te = results[3];
      modelMetadata = results[4];
      scheduleImpact = results[5];
      modelPerformance = results[6];
      allPlayers = [].concat(qb, rb, wr, te);
    });
  }

  // ========== TAB NAVIGATION ==========
  function initTabs() {
    var tabs = document.querySelectorAll('.tab-nav__btn');
    var panels = document.querySelectorAll('.tab-panel');

    function activateTab(tabEl) {
      tabs.forEach(function (t) {
        t.classList.remove('tab-nav__btn--active');
        t.setAttribute('aria-selected', 'false');
      });
      panels.forEach(function (p) {
        p.classList.remove('tab-panel--active');
        p.hidden = true;
      });
      tabEl.classList.add('tab-nav__btn--active');
      tabEl.setAttribute('aria-selected', 'true');
      var targetId = 'section-' + tabEl.dataset.tab;
      var panel = document.getElementById(targetId);
      if (panel) {
        panel.classList.add('tab-panel--active');
        panel.hidden = false;
      }
      history.replaceState(null, '', '#' + tabEl.dataset.tab);
    }

    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () { activateTab(tab); });
    });

    var hash = window.location.hash.replace('#', '');
    if (hash) {
      var matchingTab = document.querySelector('[data-tab="' + hash + '"]');
      if (matchingTab) activateTab(matchingTab);
    }
  }

  // ========== DYNAMIC HEADER ==========
  function updateHeader() {
    var badge = document.getElementById('header-badge');
    var subtitle = document.getElementById('header-subtitle');
    var upcomingSeason = (modelMetadata && modelMetadata.upcoming_season) || '';
    var prevSeason = (modelMetadata && modelMetadata.prev_season) || '';
    var hasPred = modelMetadata && modelMetadata.has_ml_predictions;
    var hasPerf = modelPerformance && modelPerformance.per_player_season_totals && modelPerformance.per_player_season_totals.length > 0;

    if (badge) {
      badge.textContent = hasPred ? upcomingSeason + ' Predictions' : prevSeason + ' Model Validation';
    }
    if (subtitle) {
      if (hasPred) {
        subtitle.textContent = upcomingSeason + ' ML predictions \u00b7 ' + prevSeason + ' model performance';
      } else if (hasPerf) {
        subtitle.textContent = prevSeason + ' out-of-sample predictions vs actuals \u00b7 ' + upcomingSeason + ' predictions pending schedule';
      } else {
        subtitle.textContent = prevSeason + ' model performance \u00b7 ' + upcomingSeason + ' predictions pending schedule';
      }
    }
  }

  // ========== OVERVIEW SECTION ==========
  function renderOverview() {
    var hasPredictions = allPlayers.some(function (p) { return p.projection_points_total != null; });
    var hasPred = modelMetadata && modelMetadata.has_ml_predictions;

    // If off-season (no ML predictions) and we have model performance data, show off-season overview
    if (!hasPred && modelPerformance && modelPerformance.per_player_season_totals && modelPerformance.per_player_season_totals.length > 0) {
      renderOffseasonOverview();
      return;
    }

    // In-season or predictions available: show normal overview
    var totalEl = document.getElementById('stat-total-players');
    if (totalEl) totalEl.textContent = allPlayers.length.toLocaleString();

    var featuresEl = document.getElementById('stat-features');
    if (featuresEl && modelMetadata && modelMetadata.n_features_per_position) {
      var vals = Object.values(modelMetadata.n_features_per_position);
      if (vals.length > 0) featuresEl.textContent = vals[0];
    }

    var rangeEl = document.getElementById('stat-training-range');
    if (rangeEl && modelMetadata) {
      rangeEl.textContent = modelMetadata.training_data_range || '2006-2024';
    }

    // Update notice
    var noticeEl = document.getElementById('data-notice');
    if (noticeEl && modelMetadata) {
      var upcoming = modelMetadata.upcoming_season || '';
      var prev = modelMetadata.prev_season || '';
      if (hasPred) {
        noticeEl.innerHTML = '<strong>How to read this board:</strong> ' +
          'Player projections are ML model predictions for the ' + upcoming + ' season. ' +
          'The <em>Model Performance</em> tab shows how the model performed on the ' +
          prev + ' season (out-of-sample predictions vs actual results).';
      } else {
        noticeEl.innerHTML = '<strong>Awaiting ' + upcoming + ' schedule:</strong> ' +
          'The ' + upcoming + ' NFL schedule has not been released. Once available, ML predictions will appear. ' +
          'No extrapolations are used.';
      }
    }

    // Top 10
    if (hasPredictions) {
      var top10 = allPlayers.slice().sort(function (a, b) {
        return (b.projection_points_total || 0) - (a.projection_points_total || 0);
      }).slice(0, 10);
      renderTop10List(top10);
    } else {
      var top10Container = document.getElementById('top10-list');
      if (top10Container) {
        top10Container.innerHTML = '<p style="color:#94a3b8;text-align:center;padding:1rem">' +
          'Projections pending \u2014 the upcoming season schedule has not been released.</p>';
      }
    }

    renderPositionCards();
  }

  function renderOffseasonOverview() {
    var section = document.getElementById('section-overview');
    if (!section) return;

    var prev = modelPerformance.season || '';
    var upcoming = (modelMetadata && modelMetadata.upcoming_season) || '';
    var m = modelPerformance.aggregate_metrics || {};
    var byPos = modelPerformance.by_position || {};
    var players = modelPerformance.per_player_season_totals || [];
    var trainRange = (modelMetadata && modelMetadata.training_data_range) || '2006-' + (prev - 1);

    var html = '';

    // Hero banner
    html += '<div class="section-card" style="text-align:center;padding:2rem;border:1px solid var(--color-accent-cyan);border-radius:var(--radius)">' +
      '<h2 style="color:var(--color-accent-cyan);font-size:1.5rem;margin-bottom:0.5rem">' +
        prev + ' Season: Model Predictions vs Reality</h2>' +
      '<p style="color:var(--color-text-secondary);margin-bottom:0.25rem">' +
        'Out-of-sample predictions &mdash; model trained on ' + trainRange + ', never saw ' + prev + ' data</p>' +
      '<span class="pill pill--success" style="margin-top:0.5rem;display:inline-block">' +
        'Zero Data Leakage Verified</span>' +
    '</div>';

    // Key metrics
    html += '<div class="hero-stats" style="margin-top:1.5rem">';
    if (m.rmse != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.rmse) + '</div><div class="stat-card__label">RMSE</div></div>';
    if (m.mae != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.mae) + '</div><div class="stat-card__label">MAE</div></div>';
    if (m.r2 != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.r2, 3) + '</div><div class="stat-card__label">R\u00b2</div></div>';
    if (m.correlation != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.correlation, 3) + '</div><div class="stat-card__label">Correlation</div></div>';
    if (m.within_7_pts_pct != null) html += '<div class="stat-card stat-card--accent"><div class="stat-card__value">' + num(m.within_7_pts_pct) + '%</div><div class="stat-card__label">Within 7 Pts</div></div>';
    html += '</div>';

    // Upcoming season notice
    html += '<div class="notice notice--info" style="margin-top:1.5rem">' +
      '<strong>' + upcoming + ' Predictions:</strong> ' +
      'The ' + upcoming + ' NFL schedule has not been released (typically May). ' +
      'Once available, ML predictions will automatically appear here. ' +
      'Below is how our model performed on the ' + prev + ' season.' +
    '</div>';

    // Top 10 Most Accurate Predictions
    var sorted = players.slice().sort(function (a, b) {
      return Math.abs(a.error) - Math.abs(b.error);
    });
    var top10Accurate = sorted.slice(0, 10);
    html += '<div class="section-card" style="margin-top:1.5rem">' +
      '<h2>Top 10 Most Accurate Predictions</h2>' +
      '<p class="section-desc">Players where the model\'s season-total prediction was closest to the actual result.</p>' +
      '<div class="top10-list">';
    top10Accurate.forEach(function (p, i) {
      var errCls = errorClass(p.error);
      html += '<div class="top10-item">' +
        '<span class="top10-rank">' + (i + 1) + '</span>' +
        '<span class="top10-name">' + escapeHtml(p.name) + '</span>' +
        '<span class="pos-badge ' + posClass(p.position) + '">' + escapeHtml(p.position) + '</span>' +
        '<span class="top10-team">' + escapeHtml(p.team) + '</span>' +
        '<span class="top10-pts" style="font-size:0.8rem">' +
          'Pred: ' + num(p.predicted_total) + ' &middot; Actual: ' + num(p.actual_total) +
          ' &middot; <span class="risk-badge ' + errCls + '">' + (p.error > 0 ? '+' : '') + num(p.error) + '</span>' +
        '</span>' +
      '</div>';
    });
    html += '</div></div>';

    // Position accuracy cards
    html += '<h2 class="section-heading">Position Accuracy</h2>';
    html += '<div class="position-grid">';
    POSITIONS.forEach(function (pos) {
      var r = byPos[pos];
      if (!r) return;
      var color = POS_COLORS[pos] || '#fff';
      html += '<div class="position-card">' +
        '<div class="position-card__header">' +
          '<span class="position-card__title" style="color:' + color + '">' + pos + '</span>' +
        '</div>' +
        '<ul class="position-card__list" style="list-style:none">' +
          '<li><span class="player-name">RMSE</span><span class="player-pts" style="color:' + color + '">' + num(r.rmse) + '</span></li>' +
          '<li><span class="player-name">MAE</span><span class="player-pts" style="color:' + color + '">' + num(r.mae) + '</span></li>' +
          '<li><span class="player-name">R\u00b2</span><span class="player-pts" style="color:' + color + '">' + num(r.r2, 3) + '</span></li>' +
          '<li><span class="player-name">Within 7 pts</span><span class="player-pts" style="color:' + color + '">' + num(r.within_7_pts_pct) + '%</span></li>' +
          '<li><span class="player-name">Within 10 pts</span><span class="player-pts" style="color:' + color + '">' + num(r.within_10_pts_pct) + '%</span></li>' +
        '</ul>' +
      '</div>';
    });
    html += '</div>';

    section.innerHTML = html;
  }

  function renderTop10List(players) {
    var container = document.getElementById('top10-list');
    if (!container) return;
    container.innerHTML = players.map(function (p, i) {
      var pts = p.projection_points_total != null ? num(p.projection_points_total) : 'Pending';
      return '<div class="top10-item">' +
        '<span class="top10-rank">' + (i + 1) + '</span>' +
        '<span class="top10-name">' + escapeHtml(p.name) + '</span>' +
        '<span class="pos-badge ' + posClass(p.position) + '">' + escapeHtml(p.position) + '</span>' +
        '<span class="top10-team">' + escapeHtml(p.team) + '</span>' +
        '<span class="top10-pts">' + pts + '</span>' +
        '</div>';
    }).join('');
  }

  function renderPositionCards() {
    var container = document.getElementById('overview-positions');
    if (!container) return;
    var hasPredictions = allPlayers.some(function (p) { return p.projection_points_total != null; });
    container.innerHTML = POSITIONS.map(function (pos) {
      var posPlayers = allPlayers.filter(function (p) { return p.position === pos; });
      posPlayers.sort(function (a, b) {
        return (b.projection_points_total || 0) - (a.projection_points_total || 0);
      });
      var top5 = posPlayers.slice(0, 5);
      var color = POS_COLORS[pos] || '#fff';
      return '<div class="position-card">' +
        '<div class="position-card__header">' +
          '<span class="position-card__title" style="color:' + color + '">' + pos + '</span>' +
          '<span class="position-card__count">' + posPlayers.length + ' players</span>' +
        '</div>' +
        '<ul class="position-card__list">' +
          top5.map(function (p) {
            var pts = hasPredictions && p.projection_points_total != null
              ? num(p.projection_points_total)
              : (p.prev_season_ppg != null ? num(p.prev_season_ppg) + ' PPG (prev)' : 'N/A');
            return '<li>' +
              '<span class="player-name">' + escapeHtml(p.name) + '</span>' +
              '<span class="player-pts" style="color:' + color + '">' + pts + '</span>' +
            '</li>';
          }).join('') +
        '</ul>' +
      '</div>';
    }).join('');
  }

  // ========== MODEL PERFORMANCE SECTION ==========
  function renderScatterPlot(players) {
    if (!players || players.length === 0) return '';
    var W = 500, H = 400, PAD = 50;

    // Compute axis range
    var allVals = [];
    players.forEach(function (p) {
      if (p.predicted_total != null) allVals.push(p.predicted_total);
      if (p.actual_total != null) allVals.push(p.actual_total);
    });
    if (allVals.length === 0) return '';
    var minVal = 0;
    var maxVal = Math.ceil(Math.max.apply(null, allVals) / 50) * 50;
    if (maxVal <= 0) maxVal = 100;

    function sx(v) { return PAD + (v - minVal) / (maxVal - minVal) * (W - 2 * PAD); }
    function sy(v) { return H - PAD - (v - minVal) / (maxVal - minVal) * (H - 2 * PAD); }

    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;max-width:600px;display:block;margin:1rem auto" xmlns="http://www.w3.org/2000/svg">';

    // Background
    svg += '<rect width="' + W + '" height="' + H + '" fill="#0f1322" rx="8"/>';

    // Grid lines
    for (var g = 0; g <= maxVal; g += 50) {
      var gx = sx(g), gy = sy(g);
      svg += '<line x1="' + PAD + '" y1="' + gy + '" x2="' + (W - PAD) + '" y2="' + gy + '" stroke="#1e293b" stroke-width="1"/>';
      svg += '<line x1="' + gx + '" y1="' + PAD + '" x2="' + gx + '" y2="' + (H - PAD) + '" stroke="#1e293b" stroke-width="1"/>';
      svg += '<text x="' + (PAD - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#64748b" font-size="10">' + g + '</text>';
      svg += '<text x="' + gx + '" y="' + (H - PAD + 15) + '" text-anchor="middle" fill="#64748b" font-size="10">' + g + '</text>';
    }

    // Perfect prediction line
    svg += '<line x1="' + sx(minVal) + '" y1="' + sy(minVal) + '" x2="' + sx(maxVal) + '" y2="' + sy(maxVal) + '" stroke="#00f5ff" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.5"/>';

    // Data points
    players.forEach(function (p) {
      if (p.predicted_total == null || p.actual_total == null) return;
      var color = POS_COLORS[p.position] || '#fff';
      svg += '<circle cx="' + sx(p.predicted_total) + '" cy="' + sy(p.actual_total) + '" r="3.5" fill="' + color + '" opacity="0.7">' +
        '<title>' + p.name + ' (' + p.position + '): Pred=' + num(p.predicted_total) + ', Actual=' + num(p.actual_total) + '</title>' +
      '</circle>';
    });

    // Axis labels
    svg += '<text x="' + (W / 2) + '" y="' + (H - 5) + '" text-anchor="middle" fill="#94a3b8" font-size="12">Predicted Total (PPR)</text>';
    svg += '<text x="12" y="' + (H / 2) + '" text-anchor="middle" fill="#94a3b8" font-size="12" transform="rotate(-90,12,' + (H / 2) + ')">Actual Total (PPR)</text>';

    // Legend
    var lx = W - PAD - 100;
    svg += '<text x="' + lx + '" y="' + (PAD + 10) + '" fill="#64748b" font-size="10">Position:</text>';
    POSITIONS.forEach(function (pos, i) {
      svg += '<circle cx="' + (lx + 5) + '" cy="' + (PAD + 22 + i * 14) + '" r="4" fill="' + POS_COLORS[pos] + '"/>';
      svg += '<text x="' + (lx + 14) + '" y="' + (PAD + 25 + i * 14) + '" fill="' + POS_COLORS[pos] + '" font-size="10">' + pos + '</text>';
    });

    svg += '</svg>';
    return svg;
  }

  function renderAccuracyBars(m) {
    if (!m) return '';
    var bars = [
      { label: 'Within 3 pts', pct: m.within_3_pts_pct, color: '#10b981' },
      { label: 'Within 5 pts', pct: m.within_5_pts_pct, color: '#00f5ff' },
      { label: 'Within 7 pts', pct: m.within_7_pts_pct, color: '#a78bfa' },
      { label: 'Within 10 pts', pct: m.within_10_pts_pct, color: '#fbbf24' },
    ];

    var html = '<div style="margin:1rem 0">';
    bars.forEach(function (bar) {
      if (bar.pct == null) return;
      html += '<div style="display:flex;align-items:center;margin-bottom:0.5rem">' +
        '<span style="width:100px;font-size:0.8rem;color:#94a3b8">' + bar.label + '</span>' +
        '<div style="flex:1;background:#1e293b;border-radius:4px;height:20px;position:relative;overflow:hidden">' +
          '<div style="width:' + Math.min(bar.pct, 100) + '%;height:100%;background:' + bar.color + ';border-radius:4px;transition:width 0.5s"></div>' +
        '</div>' +
        '<span style="width:50px;text-align:right;font-size:0.85rem;color:' + bar.color + ';font-weight:600">' + num(bar.pct) + '%</span>' +
      '</div>';
    });
    html += '</div>';
    return html;
  }

  function initModelPerformance() {
    if (!modelPerformance) return;

    var heading = document.getElementById('perf-heading');
    if (heading) {
      heading.textContent = 'Model Performance \u2014 ' + modelPerformance.season + ' Season (Out-of-Sample)';
    }

    var summary = document.getElementById('perf-summary');
    if (summary) {
      summary.innerHTML = '<span class="pill pill--success" style="margin-right:0.5rem">Zero Data Leakage</span>' +
        'These are genuine out-of-sample predictions: the model never saw ' +
        modelPerformance.season + ' data during training. ' +
        'Predicted values are compared against actual game results.';
    }

    // Aggregate metrics
    var metricsEl = document.getElementById('perf-metrics');
    if (metricsEl) {
      var m = modelPerformance.aggregate_metrics || {};
      var byPos = modelPerformance.by_position || {};
      var html = '';
      if (m.rmse || m.mae) {
        html += '<div class="hero-stats" style="margin-bottom:1rem">';
        if (m.rmse) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.rmse) + '</div><div class="stat-card__label">RMSE</div></div>';
        if (m.mae) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.mae) + '</div><div class="stat-card__label">MAE</div></div>';
        if (m.r2 != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.r2, 3) + '</div><div class="stat-card__label">R\u00b2</div></div>';
        if (m.correlation != null) html += '<div class="stat-card"><div class="stat-card__value">' + num(m.correlation, 3) + '</div><div class="stat-card__label">Correlation</div></div>';
        html += '</div>';
      }

      // Accuracy distribution bars
      html += '<h3 style="margin:1rem 0 0.5rem;color:var(--color-text-primary)">Prediction Accuracy Distribution</h3>';
      html += renderAccuracyBars(m);

      // Scatter plot
      var players = modelPerformance.per_player_season_totals || [];
      if (players.length > 0) {
        html += '<h3 style="margin:1.5rem 0 0.5rem;color:var(--color-text-primary)">Predicted vs Actual (Season Totals)</h3>';
        html += '<p style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem">Each dot is one player. Dashed line = perfect prediction. Closer to the line = more accurate.</p>';
        html += renderScatterPlot(players);
      }

      // Per-position metrics table
      var posKeys = Object.keys(byPos).filter(function (k) { return POSITIONS.indexOf(k) !== -1; });
      if (posKeys.length > 0) {
        html += '<h3 style="margin:1.5rem 0 0.5rem;color:var(--color-text-primary)">Per-Position Metrics</h3>';
        html += '<div class="table-wrap" style="margin-bottom:1rem"><table class="draft-table"><thead><tr>' +
          '<th>Position</th><th>RMSE</th><th>MAE</th><th>R\u00b2</th><th>Correlation</th>' +
          '<th>Within 7pts</th><th>Within 10pts</th>' +
          '</tr></thead><tbody>';
        posKeys.forEach(function (pos) {
          var r = byPos[pos];
          html += '<tr>' +
            '<td><span class="pos-badge ' + posClass(pos) + '">' + pos + '</span></td>' +
            '<td class="td--pts">' + num(r.rmse) + '</td>' +
            '<td class="td--pts">' + num(r.mae) + '</td>' +
            '<td class="td--pts">' + num(r.r2, 3) + '</td>' +
            '<td class="td--pts">' + num(r.correlation, 3) + '</td>' +
            '<td class="td--pts">' + num(r.within_7_pts_pct) + '%</td>' +
            '<td class="td--pts">' + num(r.within_10_pts_pct) + '%</td>' +
          '</tr>';
        });
        html += '</tbody></table></div>';
      }
      metricsEl.innerHTML = html;
    }

    // Position filter
    var filtersContainer = document.getElementById('perf-position-filters');
    if (filtersContainer) {
      filtersContainer.addEventListener('click', function (e) {
        var btn = e.target.closest('[data-position]');
        if (!btn) return;
        perfPosition = btn.dataset.position;
        filtersContainer.querySelectorAll('.btn').forEach(function (b) {
          b.classList.toggle('btn--active', b.dataset.position === perfPosition);
        });
        applyPerfFilters();
      });
    }

    // Search
    var searchInput = document.getElementById('perf-search');
    var debounceTimer;
    if (searchInput) {
      searchInput.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function () {
          perfSearch = searchInput.value.trim().toLowerCase();
          applyPerfFilters();
        }, 200);
      });
    }

    applyPerfFilters();
  }

  function applyPerfFilters() {
    if (!modelPerformance || !modelPerformance.per_player_season_totals) return;
    perfFiltered = modelPerformance.per_player_season_totals.filter(function (p) {
      if (perfPosition !== 'ALL' && p.position !== perfPosition) return false;
      if (perfSearch && p.name.toLowerCase().indexOf(perfSearch) === -1) return false;
      return true;
    });
    renderPerfTable();
  }

  function renderPerfTable() {
    var tbody = document.getElementById('perf-table-body');
    if (!tbody) return;

    var maxRows = 200;
    var displayed = perfFiltered.slice(0, maxRows);

    tbody.innerHTML = displayed.map(function (p, i) {
      var errCls = errorClass(p.error);
      return '<tr>' +
        '<td class="td--rank">' + (i + 1) + '</td>' +
        '<td class="td--name">' + escapeHtml(p.name) + '</td>' +
        '<td class="td--pos"><span class="pos-badge ' + posClass(p.position) + '">' + p.position + '</span></td>' +
        '<td class="td--team">' + escapeHtml(p.team) + '</td>' +
        '<td class="td--pts">' + num(p.predicted_total) + '</td>' +
        '<td class="td--pts">' + num(p.actual_total) + '</td>' +
        '<td class="td--pts"><span class="risk-badge ' + errCls + '">' + (p.error > 0 ? '+' : '') + num(p.error) + '</span></td>' +
        '<td class="td--ppg">' + num(p.predicted_ppg) + '</td>' +
        '<td class="td--ppg">' + num(p.actual_ppg) + '</td>' +
        '<td class="td--rank">' + p.games + '</td>' +
      '</tr>';
    }).join('');

    var footerEl = document.getElementById('perf-table-footer');
    if (footerEl) {
      footerEl.textContent = perfFiltered.length + ' player' +
        (perfFiltered.length !== 1 ? 's' : '') + ' shown \u00b7 ' +
        modelPerformance.season + ' out-of-sample predictions vs actual results';
    }
  }

  // ========== DRAFT BOARD (UPCOMING SEASON) ==========
  function initDraftBoard() {
    var hasPred = modelMetadata && modelMetadata.has_ml_predictions;
    var upcoming = (modelMetadata && modelMetadata.upcoming_season) || '';
    var prev = (modelMetadata && modelMetadata.prev_season) || '';

    // Off-season: show preview table with OOS comparison instead of empty draft board
    if (!hasPred && modelPerformance && modelPerformance.per_player_season_totals && modelPerformance.per_player_season_totals.length > 0) {
      renderOffseasonDraftBoard(upcoming, prev);
      return;
    }

    var filtersContainer = document.getElementById('position-filters');
    if (filtersContainer) {
      filtersContainer.addEventListener('click', function (e) {
        var btn = e.target.closest('[data-position]');
        if (!btn) return;
        currentPosition = btn.dataset.position;
        filtersContainer.querySelectorAll('.btn').forEach(function (b) {
          b.classList.toggle('btn--active', b.dataset.position === currentPosition);
        });
        applyFiltersAndRender();
      });
    }

    var searchInput = document.getElementById('player-search');
    var debounceTimer;
    if (searchInput) {
      searchInput.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function () {
          searchQuery = searchInput.value.trim().toLowerCase();
          applyFiltersAndRender();
        }, 200);
      });
    }

    var sortSelect = document.getElementById('sort-select');
    if (sortSelect) {
      sortSelect.addEventListener('change', function () {
        currentSort = sortSelect.value;
        applyFiltersAndRender();
      });
    }

    var closeBtn = document.getElementById('detail-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', function () {
        var panel = document.getElementById('player-detail');
        if (panel) panel.hidden = true;
      });
    }

    // Draft summary
    var summaryEl = document.getElementById('draft-summary');
    if (summaryEl && modelMetadata) {
      if (hasPred) {
        summaryEl.textContent = upcoming + ' ML model predictions \u00b7 ' +
          (modelMetadata.methodology ? modelMetadata.methodology.scoring_format : 'PPR scoring') +
          ' \u00b7 Schedule incorporated';
      } else {
        summaryEl.textContent = upcoming + ' season \u2014 predictions pending schedule release \u00b7 ' +
          'No extrapolations \u00b7 ' +
          'Players listed based on ' + prev + ' performance';
      }
    }
  }

  function renderOffseasonDraftBoard(upcoming, prev) {
    var section = document.getElementById('section-draft');
    if (!section) return;

    var players = modelPerformance.per_player_season_totals || [];

    var html = '<div class="section-card">' +
      '<h2>Upcoming Season</h2>' +
      '<div class="notice notice--info" style="margin-bottom:1rem">' +
        '<strong>' + upcoming + ' predictions pending</strong> &mdash; ' +
        'The NFL schedule is typically released in May. Predictions will automatically appear once incorporated. ' +
        'No extrapolations are ever used.' +
      '</div>' +
      '<p class="section-desc" style="margin-bottom:1rem">' +
        'While waiting for the ' + upcoming + ' schedule, here\'s how the model predicted the ' + prev + ' season ' +
        '(out-of-sample). Use this to gauge model accuracy for each player.' +
      '</p>';

    // Table showing players with OOS predictions
    html += '<div class="table-wrap"><table class="draft-table">' +
      '<thead><tr>' +
        '<th class="th--rank">#</th>' +
        '<th class="th--name">Player</th>' +
        '<th class="th--pos">Pos</th>' +
        '<th class="th--team">Team</th>' +
        '<th class="th--ppg">' + prev + ' PPG</th>' +
        '<th class="th--pts">' + prev + ' Total</th>' +
        '<th class="th--pts">Model Prediction</th>' +
        '<th class="th--pts">Error</th>' +
        '<th class="th--rank">Games</th>' +
      '</tr></thead><tbody>';

    players.slice(0, 100).forEach(function (p, i) {
      var errCls = errorClass(p.error);
      html += '<tr>' +
        '<td class="td--rank">' + (i + 1) + '</td>' +
        '<td class="td--name">' + escapeHtml(p.name) + '</td>' +
        '<td class="td--pos"><span class="pos-badge ' + posClass(p.position) + '">' + p.position + '</span></td>' +
        '<td class="td--team">' + escapeHtml(p.team) + '</td>' +
        '<td class="td--ppg">' + num(p.actual_ppg) + '</td>' +
        '<td class="td--pts">' + num(p.actual_total) + '</td>' +
        '<td class="td--pts">' + num(p.predicted_total) + '</td>' +
        '<td class="td--pts"><span class="risk-badge ' + errCls + '">' + (p.error > 0 ? '+' : '') + num(p.error) + '</span></td>' +
        '<td class="td--rank">' + p.games + '</td>' +
      '</tr>';
    });

    html += '</tbody></table></div>';
    html += '<div class="table-footer">' + players.length + ' players shown &middot; ' +
      prev + ' out-of-sample predictions vs actual results</div>';
    html += '</div>';

    section.innerHTML = html;
  }

  function applyFiltersAndRender() {
    filteredPlayers = allPlayers.filter(function (p) {
      if (currentPosition !== 'ALL' && p.position !== currentPosition) return false;
      if (searchQuery && p.name.toLowerCase().indexOf(searchQuery) === -1) return false;
      return true;
    });

    filteredPlayers.sort(function (a, b) {
      if (currentSort === 'risk_score') return (a.risk_score || 100) - (b.risk_score || 100);
      if (currentSort === 'adp') return (a.adp || 999) - (b.adp || 999);
      if (currentSort === 'prev_season_ppg') return (b.prev_season_ppg || 0) - (a.prev_season_ppg || 0);
      return ((b[currentSort] || 0) - (a[currentSort] || 0));
    });

    renderDraftTable();
  }

  function renderDraftTable() {
    var tbody = document.getElementById('draft-table-body');
    if (!tbody) return;

    var hasPredictions = allPlayers.some(function (p) { return p.projection_points_total != null; });
    var maxRows = 300;
    var displayed = filteredPlayers.slice(0, maxRows);

    tbody.innerHTML = displayed.map(function (p, i) {
      var pts = p.projection_points_total != null ? num(p.projection_points_total) : '<span style="color:#64748b">Pending</span>';
      var ppg = p.projection_points_per_game != null ? num(p.projection_points_per_game) : '<span style="color:#64748b">Pending</span>';
      var riskBadge = p.risk_score != null
        ? '<span class="risk-badge ' + riskClass(p.risk_score) + '">' + p.risk_score + '</span>'
        : '<span style="color:#64748b">N/A</span>';
      return '<tr class="draft-row" data-idx="' + i + '">' +
        '<td class="td--rank">' + (i + 1) + '</td>' +
        '<td class="td--name">' + escapeHtml(p.name) + '</td>' +
        '<td class="td--pos"><span class="pos-badge ' + posClass(p.position) + '">' + p.position + '</span></td>' +
        '<td class="td--team">' + escapeHtml(p.team) + '</td>' +
        '<td class="td--pts">' + pts + '</td>' +
        '<td class="td--ppg">' + ppg + '</td>' +
        '<td class="td--risk">' + riskBadge + '</td>' +
        '<td class="td--adp">' + (p.adp || 'N/A') + '</td>' +
      '</tr>';
    }).join('');

    tbody.querySelectorAll('.draft-row').forEach(function (row) {
      row.addEventListener('click', function () {
        var idx = parseInt(row.dataset.idx, 10);
        var player = filteredPlayers[idx];
        if (player) showPlayerDetail(player);
      });
    });

    var footerEl = document.getElementById('table-footer');
    if (footerEl) {
      if (filteredPlayers.length > maxRows) {
        footerEl.textContent = 'Showing ' + maxRows + ' of ' + filteredPlayers.length + ' players.';
      } else {
        footerEl.textContent = filteredPlayers.length + ' player' + (filteredPlayers.length !== 1 ? 's' : '') + ' shown';
      }
    }
  }

  // ========== PLAYER DETAIL ==========
  function showPlayerDetail(player) {
    var panel = document.getElementById('player-detail');
    var content = document.getElementById('detail-content');
    if (!panel || !content) return;

    var hasPred = player.projection_points_total != null;
    var upcoming = (modelMetadata && modelMetadata.upcoming_season) || '';

    var statsHtml = '<div class="detail-stats">';
    if (hasPred) {
      var ceiling = player.projection_ceiling || 1;
      statsHtml +=
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Total Projected</div>' +
          '<div class="detail-stat__value">' + num(player.projection_points_total) + '</div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Per Game</div>' +
          '<div class="detail-stat__value">' + num(player.projection_points_per_game) + '</div>' +
        '</div>';
      if (player.projection_floor != null) {
        statsHtml +=
          '<div class="detail-stat">' +
            '<div class="detail-stat__label">Floor (Season)</div>' +
            '<div class="detail-stat__value">' + num(player.projection_floor) + '</div>' +
          '</div>' +
          '<div class="detail-stat">' +
            '<div class="detail-stat__label">Ceiling (Season)</div>' +
            '<div class="detail-stat__value">' + num(player.projection_ceiling) + '</div>' +
          '</div>';
      }
    } else {
      statsHtml += '<div class="detail-stat">' +
          '<div class="detail-stat__label">' + upcoming + ' Projection</div>' +
          '<div class="detail-stat__value" style="color:#64748b">Pending schedule release</div>' +
        '</div>';
    }
    if (player.risk_score != null) {
      statsHtml +=
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Risk Score</div>' +
          '<div class="detail-stat__value"><span class="risk-badge ' + riskClass(player.risk_score) + '">' + player.risk_score + ' / 100</span></div>' +
        '</div>';
    }
    statsHtml += '</div>';

    // Previous season basis
    var prevHtml = '';
    if (player.prev_season_ppg != null) {
      prevHtml = '<div class="detail-actuals">' +
        '<p class="detail-actuals__note">' + (player.prev_season || '') + ' season: ' +
          (player.prev_season_games || 0) + ' games, ' +
          num(player.prev_season_total_fp) + ' total PPR points, ' +
          num(player.prev_season_ppg) + ' PPG</p>' +
      '</div>';
    }

    // Features
    var featuresHtml = '';
    if (player.key_features && player.key_features.length > 0) {
      featuresHtml = '<div class="detail-features">' +
        '<h4>Key Predictive Features (' + escapeHtml(player.position) + ')</h4>' +
        '<ul class="feature-list">' +
        player.key_features.map(function (f) {
          var imp = player.feature_importance_rank && player.feature_importance_rank[f];
          var impText = imp != null ? (imp * 100).toFixed(1) + '%' : '--';
          return '<li>' +
            '<span class="feature-name">' + formatFeatureName(f) + '</span>' +
            '<span class="feature-imp">' + impText + '</span>' +
          '</li>';
        }).join('') +
        '</ul></div>';
    }

    content.innerHTML =
      '<div class="detail-header">' +
        '<span class="pos-badge ' + posClass(player.position) + '">' + player.position + '</span>' +
        '<h3>' + escapeHtml(player.name) + '</h3>' +
        '<span class="detail-team">' + escapeHtml(player.team) + '</span>' +
      '</div>' +
      statsHtml +
      prevHtml +
      featuresHtml +
      '<div class="detail-schedule">' +
        '<span class="pill ' + (player.uses_schedule ? 'pill--success' : 'pill--warning') + '">' +
          (player.uses_schedule ? 'Schedule Incorporated' : 'Schedule Not Available') +
        '</span>' +
        '<p style="margin-top:0.5rem;font-size:0.8rem;color:#94a3b8">' +
          (player.uses_schedule
            ? 'Predictions include ' + upcoming + ' schedule data.'
            : 'Predictions will be available once the ' + upcoming + ' NFL schedule is released. No extrapolations are used.') +
        '</p>' +
      '</div>';

    panel.hidden = false;
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ========== METHODOLOGY SECTION ==========
  function renderMethodology() {
    var container = document.getElementById('methodology-content');
    if (!container || !modelMetadata) return;

    var m = modelMetadata.methodology || {};
    var hasPred = modelMetadata.has_ml_predictions;
    var upcoming = modelMetadata.upcoming_season || '';
    var prev = modelMetadata.prev_season || '';
    var html = '';

    // Off-season validation section (when no predictions yet)
    if (!hasPred) {
      html += '<div class="content-section" style="border-left:3px solid var(--color-accent-cyan);padding-left:1rem">' +
        '<h3>Off-Season Validation</h3>' +
        '<p>During the off-season, the primary purpose of this app is to demonstrate model accuracy ' +
        'through <strong>out-of-sample validation</strong> on the ' + prev + ' season.</p>' +
        '<div style="display:flex;gap:0.5rem;align-items:center;margin:1rem 0;flex-wrap:wrap">' +
          '<div style="background:#1e293b;padding:0.5rem 1rem;border-radius:8px;text-align:center;flex:1;min-width:120px">' +
            '<div style="color:#64748b;font-size:0.75rem">TRAIN</div>' +
            '<div style="color:var(--color-accent-cyan);font-weight:600">' + (m.training_window || '2006-2024') + '</div>' +
          '</div>' +
          '<span style="color:#64748b;font-size:1.2rem">\u2192</span>' +
          '<div style="background:#1e293b;padding:0.5rem 1rem;border-radius:8px;text-align:center;flex:1;min-width:120px">' +
            '<div style="color:#64748b;font-size:0.75rem">TEST (OOS)</div>' +
            '<div style="color:var(--color-accent-emerald);font-weight:600">' + prev + '</div>' +
          '</div>' +
          '<span style="color:#64748b;font-size:1.2rem">\u2192</span>' +
          '<div style="background:#1e293b;padding:0.5rem 1rem;border-radius:8px;text-align:center;flex:1;min-width:120px">' +
            '<div style="color:#64748b;font-size:0.75rem">PREDICT</div>' +
            '<div style="color:var(--color-accent-amber);font-weight:600">' + upcoming + ' (pending)</div>' +
          '</div>' +
        '</div>' +
        '<p style="font-size:0.85rem;color:#94a3b8">The model was trained on all data before ' + prev + '. ' +
        'It predicted each week of ' + prev + ' without seeing any ' + prev + ' data. ' +
        'Results are shown in the Overview and Model Performance tabs.</p>' +
      '</div>';
    }

    html += '<div class="content-section">' +
      '<h3>Prediction Target</h3>' +
      '<p>' + escapeHtml(modelMetadata.target_definition || 'Fantasy points per game') + '</p>' +
    '</div>';

    html += '<div class="content-section">' +
      '<h3>Modeling Approach</h3>' +
      '<p><strong>Model:</strong> ' + escapeHtml(m.model_type || 'LightGBM ensemble') + '</p>' +
      '<p><strong>Training Window:</strong> ' + escapeHtml(m.training_window || 'N/A') +
        ' (' + escapeHtml(m.test_season || 'N/A') + ' held out for evaluation)</p>' +
      '<p><strong>Scoring Format:</strong> ' + escapeHtml(m.scoring_format || 'PPR') + '</p>' +
      '<p><strong>Prediction Horizons:</strong> ' + (m.horizons || []).join(', ') + '</p>' +
    '</div>';

    // Features
    html += '<div class="content-section">' +
      '<h3>Features Used</h3>' +
      '<p>' + escapeHtml(m.features_description || '') + '</p>';
    if (modelMetadata.top_features) {
      html += '<div class="metric-grid">';
      POSITIONS.forEach(function (pos) {
        var feats = modelMetadata.top_features[pos];
        if (!feats || feats.length === 0) return;
        html += '<div class="metric-item">' +
          '<div class="metric-item__label" style="color:' + POS_COLORS[pos] + '">' + pos + ' Top Features</div>' +
          '<div style="margin-top:0.375rem;font-size:0.8rem">' +
            feats.slice(0, 5).map(function (f) {
              return formatFeatureName(f.feature) + ' (' + (f.importance * 100).toFixed(1) + '%)';
            }).join('<br>') +
          '</div>' +
        '</div>';
      });
      html += '</div>';
    }
    html += '</div>';

    // Training & Evaluation
    html += '<div class="content-section">' +
      '<h3>ML Model Training &amp; Evaluation</h3>';
    var bt = modelMetadata.backtest_results || {};
    var posKeys = Object.keys(bt).filter(function (k) { return POSITIONS.indexOf(k) !== -1; });
    if (posKeys.length > 0) {
      html += '<p>Backtest results on the ' + (modelMetadata.test_season || '') + ' held-out season:</p>' +
        '<div class="table-wrap"><table class="draft-table" style="margin-top:0.75rem">' +
        '<thead><tr><th>Position</th><th>RMSE</th><th>MAE</th><th>R\u00b2</th><th>Correlation</th></tr></thead><tbody>';
      posKeys.forEach(function (pos) {
        var r = bt[pos];
        html += '<tr>' +
          '<td><span class="pos-badge ' + posClass(pos) + '">' + pos + '</span></td>' +
          '<td class="td--pts">' + num(r.rmse) + '</td>' +
          '<td class="td--pts">' + num(r.mae) + '</td>' +
          '<td class="td--pts">' + num(r.r2, 3) + '</td>' +
          '<td class="td--pts">' + num(r.correlation, 3) + '</td>' +
        '</tr>';
      });
      html += '</tbody></table></div>';
    }
    html += '</div>';

    // Overfitting Prevention
    if (m.overfitting_prevention && m.overfitting_prevention.length > 0) {
      html += '<div class="content-section">' +
        '<h3>Overfitting Prevention</h3>' +
        '<ul>' + m.overfitting_prevention.map(function (s) {
          return '<li>' + escapeHtml(s) + '</li>';
        }).join('') + '</ul>' +
      '</div>';
    }

    // Data Basis Note
    if (modelMetadata.data_basis_note) {
      html += '<div class="notice notice--info" style="margin-top:1rem">' +
        '<strong>Data Source:</strong> ' + escapeHtml(modelMetadata.data_basis_note) +
      '</div>';
    }

    container.innerHTML = html;
  }

  // ========== DATA & ASSUMPTIONS SECTION ==========
  function renderDataSection() {
    var container = document.getElementById('data-content');
    if (!container) return;

    var upcoming = (modelMetadata && modelMetadata.upcoming_season) || '';
    var prev = (modelMetadata && modelMetadata.prev_season) || '';
    var hasPred = modelMetadata && modelMetadata.has_ml_predictions;

    var html = '<div class="content-section">' +
      '<h3>Data Sources</h3>' +
      '<ul>' +
        '<li><strong>Player Statistics:</strong> Weekly player performance data from nfl-data-py (official NFL play-by-play and boxscore data)</li>' +
        '<li><strong>Time Range:</strong> ' + (modelMetadata ? modelMetadata.training_data_range : '2006-2024') + ' for ML model training</li>' +
        '<li><strong>Play-by-Play:</strong> Advanced metrics including EPA, WPA, and success rate</li>' +
        '<li><strong>Team Context:</strong> Team-level offensive stats, play volume, pass/rush ratios</li>' +
      '</ul>' +
    '</div>';

    html += '<div class="content-section">' +
      '<h3>Scoring Format</h3>' +
      '<p>All projections use <strong>PPR (Points Per Reception)</strong> scoring:</p>' +
      '<div class="metric-grid">' +
        '<div class="metric-item"><div class="metric-item__label">Passing Yard</div><div class="metric-item__value">0.04</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Passing TD</div><div class="metric-item__value">4</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Interception</div><div class="metric-item__value">-2</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Rushing Yard</div><div class="metric-item__value">0.1</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Rushing TD</div><div class="metric-item__value">6</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Reception</div><div class="metric-item__value">1.0</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Receiving Yard</div><div class="metric-item__value">0.1</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Receiving TD</div><div class="metric-item__value">6</div></div>' +
        '<div class="metric-item"><div class="metric-item__label">Fumble Lost</div><div class="metric-item__value">-2</div></div>' +
      '</div>' +
    '</div>';

    html += '<div class="content-section">' +
      '<h3>Key Principles</h3>' +
      '<ul>' +
        '<li><strong>No Extrapolation:</strong> This app never extrapolates past performance to future seasons. ' +
           'Upcoming season projections come exclusively from the ML model.</li>' +
        '<li><strong>Model Performance:</strong> The Model Performance tab shows genuine out-of-sample ' +
           'predictions vs actual results for the ' + prev + ' season.</li>' +
        (hasPred
          ? '<li><strong>Current Projections:</strong> The Upcoming Season tab shows ML model predictions for ' + upcoming + '.</li>'
          : '<li><strong>Pending:</strong> The ' + upcoming + ' schedule has not been released. ' +
             'Projections will appear once the schedule is available.</li>') +
      '</ul>' +
    '</div>';

    html += '<div class="content-section">' +
      '<h3>Season Transition</h3>' +
      '<p>The app automatically handles the NFL calendar:</p>' +
      '<ul>' +
        '<li><strong>Off-Season (Feb\u2013May):</strong> Shows previous season out-of-sample predictions vs actuals. ' +
           'No forward projections until the schedule is released.</li>' +
        '<li><strong>Schedule Release (~May):</strong> Once the ' + upcoming + ' schedule is available, ' +
           'the model retrains with ' + prev + ' data included and generates ' + upcoming + ' predictions. ' +
           'The front page automatically switches to show upcoming season projections.</li>' +
        '<li><strong>In-Season:</strong> Weekly predictions with schedule-adjusted matchup quality.</li>' +
      '</ul>' +
    '</div>';

    if (modelMetadata && modelMetadata.last_updated) {
      html += '<div class="content-section">' +
        '<h3>Data Freshness</h3>' +
        '<p>Model last trained: ' + escapeHtml(modelMetadata.last_updated) + '</p>' +
      '</div>';
    }

    html += '<div class="notice notice--info">' +
      '<strong>Disclaimer:</strong> Fantasy football projections are inherently uncertain. ' +
      'Injuries, coaching changes, roster moves, and schedule difficulty can all significantly ' +
      'impact actual outcomes.' +
    '</div>';

    container.innerHTML = html;
  }

  // ========== SCHEDULE SECTION ==========
  function renderScheduleSection() {
    var container = document.getElementById('schedule-content');
    if (!container) return;

    var isIncorporated = scheduleImpact && scheduleImpact.schedule_incorporated;
    var upcoming = (scheduleImpact && scheduleImpact.season) || '';
    var html = '';

    if (!isIncorporated) {
      html += '<div class="schedule-status">' +
        '<span class="schedule-status__icon" aria-hidden="true">&#128197;</span>' +
        '<div class="schedule-status__text">' +
          '<h3>Schedule Not Yet Released</h3>' +
          '<p>' + escapeHtml(scheduleImpact ? scheduleImpact.reason : 'The ' + upcoming + ' NFL schedule has not been released.') + '</p>' +
        '</div>' +
      '</div>';

      html += '<div class="content-section">' +
        '<h3>Expected Timeline</h3>' +
        '<p>The NFL typically releases the regular season schedule in <strong>mid-May</strong>. ' +
        'Once the ' + upcoming + ' schedule is released:</p>' +
        '<ol>' +
          '<li>Schedule data is automatically detected by the pipeline</li>' +
          '<li>Models are retrained to include the latest completed season data</li>' +
          '<li>Full-season ML predictions are generated for ' + upcoming + '</li>' +
          '<li>The app front page switches to show ' + upcoming + ' projections</li>' +
        '</ol>' +
      '</div>';

      html += '<div class="content-section">' +
        '<h3>What Schedule Incorporation Changes</h3>' +
        '<p>When the schedule is available, the following adjustments are applied to predictions:</p>' +
        '<ul>' +
          '<li><strong>Matchup quality:</strong> Opponent defense strength per position (e.g., facing a top-5 vs bottom-5 pass defense)</li>' +
          '<li><strong>Home/away splits:</strong> Historical performance differences at home vs on the road</li>' +
          '<li><strong>Bye week identification:</strong> Rest advantages and post-bye performance patterns</li>' +
          '<li><strong>Short-week adjustments:</strong> Thursday Night Football scheduling impacts</li>' +
        '</ul>' +
      '</div>';
    } else {
      html += '<div class="schedule-status">' +
        '<span class="schedule-status__icon" aria-hidden="true">&#9989;</span>' +
        '<div class="schedule-status__text">' +
          '<h3>Schedule Incorporated</h3>' +
          '<p>The ' + upcoming + ' NFL schedule has been incorporated into ML predictions.</p>' +
        '</div>' +
      '</div>';
    }

    container.innerHTML = html;
  }

  // ========== FAQ SECTION ==========
  function renderFAQ() {
    var container = document.getElementById('faq-content');
    if (!container) return;

    var upcoming = (modelMetadata && modelMetadata.upcoming_season) || '';
    var prev = (modelMetadata && modelMetadata.prev_season) || '';
    var hasPred = modelMetadata && modelMetadata.has_ml_predictions;
    var isScheduleUsed = scheduleImpact && scheduleImpact.schedule_incorporated;

    var faqs = [
      {
        q: 'What am I looking at during the off-season?',
        a: hasPred
          ? 'The ' + upcoming + ' schedule has been incorporated, so you\'re seeing ML model predictions for the upcoming season.'
          : 'During the off-season, the app showcases how the model predicted the ' + prev + ' season. ' +
            'These are genuine <strong>out-of-sample predictions</strong>: the model was trained only on data before ' + prev + ' ' +
            'and predicted each week of ' + prev + ' without seeing any of that season\'s data. ' +
            'This lets you evaluate model accuracy before the ' + upcoming + ' predictions are available.'
      },
      {
        q: 'How do I know the model didn\'t cheat on the ' + prev + ' predictions?',
        a: 'The model uses a strict <strong>expanding-window backtest</strong> with multiple leakage safeguards: ' +
          '(1) Training data only includes seasons before ' + prev + '. ' +
          '(2) Each week\'s prediction is made using only data available up to that point. ' +
          '(3) An automated leakage check verifies no future data appears in training before every fold. ' +
          '(4) Feature scaling is fit on training data only \u2014 never on test data. ' +
          '(5) Rolling features are re-computed per fold to prevent look-ahead bias. ' +
          'These measures ensure genuinely out-of-sample predictions with zero data leakage.'
      },
      {
        q: 'What does "Model Performance" show?',
        a: 'The Model Performance tab displays genuine out-of-sample predictions vs actual results for the ' +
          prev + ' season. The model was trained on data through ' + (prev - 1) +
          ' and never saw ' + prev + ' data during training. This demonstrates how accurately the model ' +
          'would have predicted fantasy outcomes for a season it had never seen.'
      },
      {
        q: 'Why are upcoming season projections blank?',
        a: hasPred
          ? 'They aren\'t! The ' + upcoming + ' schedule has been incorporated and ML predictions are available.'
          : 'The ' + upcoming + ' NFL schedule has not been released yet (typically released in mid-May). ' +
            'Rather than extrapolating past performance, we wait for the actual schedule so the ML model can ' +
            'generate proper predictions that account for matchup quality, home/away splits, and bye weeks. ' +
            'Once the schedule drops, predictions will automatically appear on this app.'
      },
      {
        q: 'How is this different from simple extrapolation?',
        a: 'Many sites take last season\'s per-game average and multiply by 17 games. We do NOT do this. ' +
          'Our ML model (LightGBM ensemble with XGBoost and Ridge regression) uses 50+ features per position ' +
          'including rolling averages, utilization metrics, team context, and advanced play-by-play data ' +
          'to generate genuine forward predictions. When the model can\'t generate proper predictions ' +
          '(e.g., no schedule available), we show "pending" rather than an extrapolation.'
      },
      {
        q: 'Are these projections schedule-adjusted?',
        a: isScheduleUsed
          ? 'Yes. The ' + upcoming + ' NFL schedule has been incorporated.'
          : 'Not yet. The ' + upcoming + ' NFL schedule has not been released. Once available (typically May), ' +
            'matchup-quality adjustments will be applied.'
      },
      {
        q: 'What do the risk scores mean?',
        a: 'Risk scores range from 0 (lowest risk) to 100 (highest risk), calculated relative to each position group. ' +
          'They combine: weekly scoring volatility (30%), coefficient of variation (25%), ' +
          'inverse consistency (25%), and games played penalty (20%).'
      },
      {
        q: 'What scoring format is used?',
        a: 'All values use PPR (Points Per Reception) scoring.'
      },
      {
        q: 'Why are some players missing?',
        a: 'Players must have played in the ' + prev + ' NFL season to appear. This excludes: ' +
          '(1) Incoming rookies, (2) Players who missed the entire ' + prev + ' season, (3) Retired players.'
      }
    ];

    container.innerHTML = faqs.map(function (faq) {
      return '<div class="faq-item">' +
        '<h3>' + escapeHtml(faq.q) + '</h3>' +
        '<p>' + faq.a + '</p>' +
      '</div>';
    }).join('');
  }

  // ========== INITIALIZATION ==========
  function init() {
    loadAllData()
      .then(function () {
        initTabs();
        updateHeader();
        renderOverview();
        initModelPerformance();
        initDraftBoard();
        applyFiltersAndRender();
        renderMethodology();
        renderDataSection();
        renderScheduleSection();
        renderFAQ();
      })
      .catch(function (err) {
        console.error('Failed to load app data:', err);
        var main = document.querySelector('.app-main');
        if (main) {
          main.innerHTML = '<div class="section-card">' +
            '<p style="color:#fb7185">Error loading data: ' + escapeHtml(err.message) +
            '. Please try refreshing the page.</p></div>';
        }
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
