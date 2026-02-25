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
  var currentPosition = 'ALL';
  var currentSort = 'projection_points_total';
  var searchQuery = '';

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
    ]).then(function (results) {
      var qb = results[0], rb = results[1], wr = results[2], te = results[3];
      modelMetadata = results[4];
      scheduleImpact = results[5];
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

    // Handle initial hash
    var hash = window.location.hash.replace('#', '');
    if (hash) {
      var matchingTab = document.querySelector('[data-tab="' + hash + '"]');
      if (matchingTab) activateTab(matchingTab);
    }
  }

  // ========== OVERVIEW SECTION ==========
  function renderOverview() {
    // Update stat cards
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

    // Top 10 overall
    var top10 = allPlayers.slice().sort(function (a, b) {
      return b.projection_points_total - a.projection_points_total;
    }).slice(0, 10);
    renderTop10List(top10);

    // Position cards
    renderPositionCards();
  }

  function renderTop10List(players) {
    var container = document.getElementById('top10-list');
    if (!container) return;
    container.innerHTML = players.map(function (p, i) {
      return '<div class="top10-item">' +
        '<span class="top10-rank">' + (i + 1) + '</span>' +
        '<span class="top10-name">' + escapeHtml(p.name) + '</span>' +
        '<span class="pos-badge ' + posClass(p.position) + '">' + escapeHtml(p.position) + '</span>' +
        '<span class="top10-team">' + escapeHtml(p.team) + '</span>' +
        '<span class="top10-pts">' + num(p.projection_points_total) + '</span>' +
        '</div>';
    }).join('');
  }

  function renderPositionCards() {
    var container = document.getElementById('overview-positions');
    if (!container) return;
    container.innerHTML = POSITIONS.map(function (pos) {
      var posPlayers = allPlayers.filter(function (p) { return p.position === pos; });
      posPlayers.sort(function (a, b) { return b.projection_points_total - a.projection_points_total; });
      var top5 = posPlayers.slice(0, 5);
      var color = POS_COLORS[pos] || '#fff';
      return '<div class="position-card">' +
        '<div class="position-card__header">' +
          '<span class="position-card__title" style="color:' + color + '">' + pos + '</span>' +
          '<span class="position-card__count">' + posPlayers.length + ' players</span>' +
        '</div>' +
        '<ul class="position-card__list">' +
          top5.map(function (p) {
            return '<li>' +
              '<span class="player-name">' + escapeHtml(p.name) + '</span>' +
              '<span class="player-pts" style="color:' + color + '">' + num(p.projection_points_total) + '</span>' +
            '</li>';
          }).join('') +
        '</ul>' +
      '</div>';
    }).join('');
  }

  // ========== DRAFT BOARD ==========
  function initDraftBoard() {
    // Position filter buttons
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

    // Search input with debounce
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

    // Sort dropdown
    var sortSelect = document.getElementById('sort-select');
    if (sortSelect) {
      sortSelect.addEventListener('change', function () {
        currentSort = sortSelect.value;
        applyFiltersAndRender();
      });
    }

    // Close detail panel
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
      var schedText = scheduleImpact && !scheduleImpact.schedule_incorporated
        ? 'Schedule not yet incorporated.'
        : 'Schedule incorporated.';
      var draftSrc = modelMetadata.data_source || 'extrapolation';
      var draftBasis = modelMetadata.basis_season || 2025;
      var srcLabel = draftSrc === 'ml_model'
        ? 'ML model predictions for ' + (modelMetadata.target_season || draftBasis + 1) + ' season'
        : draftBasis + ' actuals extrapolated to 17 games';
      summaryEl.textContent = srcLabel + ' \u00b7 ' +
        (modelMetadata.methodology ? modelMetadata.methodology.scoring_format : 'PPR scoring') +
        ' \u00b7 ' + schedText +
        ' \u00b7 Last updated: ' + (modelMetadata.last_updated || 'N/A').split('T')[0];
    }
  }

  function applyFiltersAndRender() {
    filteredPlayers = allPlayers.filter(function (p) {
      if (currentPosition !== 'ALL' && p.position !== currentPosition) return false;
      if (searchQuery && p.name.toLowerCase().indexOf(searchQuery) === -1) return false;
      return true;
    });

    filteredPlayers.sort(function (a, b) {
      if (currentSort === 'risk_score') return a.risk_score - b.risk_score;
      if (currentSort === 'adp') return a.adp - b.adp;
      return (b[currentSort] || 0) - (a[currentSort] || 0);
    });

    renderDraftTable();
  }

  function renderDraftTable() {
    var tbody = document.getElementById('draft-table-body');
    if (!tbody) return;

    var maxRows = 300;
    var displayed = filteredPlayers.slice(0, maxRows);

    tbody.innerHTML = displayed.map(function (p, i) {
      return '<tr class="draft-row" data-idx="' + i + '">' +
        '<td class="td--rank">' + (i + 1) + '</td>' +
        '<td class="td--name">' + escapeHtml(p.name) + '</td>' +
        '<td class="td--pos"><span class="pos-badge ' + posClass(p.position) + '">' + p.position + '</span></td>' +
        '<td class="td--team">' + escapeHtml(p.team) + '</td>' +
        '<td class="td--pts">' + num(p.projection_points_total) + '</td>' +
        '<td class="td--ppg">' + num(p.projection_points_per_game) + '</td>' +
        '<td class="td--risk"><span class="risk-badge ' + riskClass(p.risk_score) + '">' + p.risk_score + '</span></td>' +
        '<td class="td--adp">' + p.adp + '</td>' +
      '</tr>';
    }).join('');

    // Row click handlers
    tbody.querySelectorAll('.draft-row').forEach(function (row) {
      row.addEventListener('click', function () {
        var idx = parseInt(row.dataset.idx, 10);
        var player = filteredPlayers[idx];
        if (player) showPlayerDetail(player);
      });
    });

    // Table footer
    var footerEl = document.getElementById('table-footer');
    if (footerEl) {
      if (filteredPlayers.length > maxRows) {
        footerEl.textContent = 'Showing ' + maxRows + ' of ' + filteredPlayers.length + ' players. Use search or position filter to narrow results.';
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

    var ceiling = player.projection_ceiling || 1;
    var floorPct = Math.max(0, (player.projection_floor / ceiling) * 100).toFixed(1);
    var projPct = Math.min(100, Math.max(0, (player.projection_points_total / ceiling) * 100)).toFixed(1);
    var fillWidth = (100 - parseFloat(floorPct)).toFixed(1);

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

      '<div class="detail-stats">' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Total Projected</div>' +
          '<div class="detail-stat__value">' + num(player.projection_points_total) + '</div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Per Game</div>' +
          '<div class="detail-stat__value">' + num(player.projection_points_per_game) + '</div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Floor (Season)</div>' +
          '<div class="detail-stat__value">' + num(player.projection_floor) + '</div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Ceiling (Season)</div>' +
          '<div class="detail-stat__value">' + num(player.projection_ceiling) + '</div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">Risk Score</div>' +
          '<div class="detail-stat__value"><span class="risk-badge ' + riskClass(player.risk_score) + '">' + player.risk_score + ' / 100</span></div>' +
        '</div>' +
        '<div class="detail-stat">' +
          '<div class="detail-stat__label">ADP (Rank)</div>' +
          '<div class="detail-stat__value">' + player.adp + '</div>' +
        '</div>' +
      '</div>' +

      '<div class="range-bar">' +
        '<div class="range-bar__labels">' +
          '<span>Floor: ' + num(player.projection_floor) + '</span>' +
          '<span>Projected: ' + num(player.projection_points_total) + '</span>' +
          '<span>Ceiling: ' + num(player.projection_ceiling) + '</span>' +
        '</div>' +
        '<div class="range-bar__track">' +
          '<div class="range-bar__fill" style="left:' + floorPct + '%;width:' + fillWidth + '%"></div>' +
          '<div class="range-bar__marker" style="left:' + projPct + '%"></div>' +
        '</div>' +
      '</div>' +

      featuresHtml +

      '<div class="detail-schedule">' +
        '<span class="pill ' + (player.uses_schedule ? 'pill--success' : 'pill--warning') + '">' +
          (player.uses_schedule ? 'Schedule Incorporated' : 'Schedule Not Available') +
        '</span>' +
        '<p style="margin-top:0.5rem;font-size:0.8rem;color:#94a3b8">' +
          (player.uses_schedule
            ? 'These estimates include the schedule, including bye weeks and matchup strength.'
            : 'These estimates are schedule-neutral; the NFL schedule is not yet available.') +
        '</p>' +
      '</div>' +

      '<div class="detail-actuals">' +
        '<p class="detail-actuals__note">' +
          (player.basis_season || 2025) + ' season reference: ' +
          (player.games_played_basis || player.games_played_2025) + ' games, ' +
          num(player.total_fp_basis || player.total_fp_2025) + ' total PPR points' +
          (player.ppg_basis ? ' (' + num(player.ppg_basis) + ' PPG)' : '') +
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
    var html = '';

    // Prediction Target
    html += '<div class="content-section">' +
      '<h3>Prediction Target</h3>' +
      '<p>' + escapeHtml(modelMetadata.target_definition || 'Fantasy points per game') + '</p>' +
    '</div>';

    // Modeling Approach
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
    var methSrc = modelMetadata.data_source || 'extrapolation';
    var methBasis = modelMetadata.basis_season || 2025;
    html += '<div class="content-section">' +
      '<h3>ML Model Training &amp; Evaluation</h3>';
    if (methSrc === 'ml_model') {
      html += '<div class="notice notice--info" style="margin-bottom:0.75rem">' +
        '<strong>Draft Board uses ML predictions.</strong> The model was trained on ' +
        (modelMetadata.training_data_range || '') + ' and is predicting ' +
        (modelMetadata.target_season || methBasis + 1) + ' season performance. ' +
        'The backtest metrics below show how the model performed on the ' +
        (modelMetadata.test_season || methBasis) + ' held-out validation season.' +
      '</div>';
    } else {
      html += '<div class="notice notice--warning" style="margin-bottom:0.75rem">' +
        '<strong>Note:</strong> The metrics below reflect the ML model validated against the ' +
        (modelMetadata.test_season || methBasis) + ' season (held-out test set). ' +
        'The Draft Board tab currently shows ' + methBasis + ' actuals extrapolated to 17 games, ' +
        'not these ML model outputs. Run <code>python scripts/prepare_new_season.py</code> to generate ML predictions.' +
      '</div>';
    }

    var bt = modelMetadata.backtest_results || {};
    var posKeys = Object.keys(bt).filter(function (k) { return POSITIONS.indexOf(k) !== -1; });
    if (posKeys.length > 0) {
      html += '<p>ML model backtest results on the ' + (modelMetadata.test_season || 2025) + ' held-out season:</p>' +
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
      html += '<div class="notice notice--warning" style="margin-top:1rem">' +
        '<strong>Draft Board Data Source:</strong> ' + escapeHtml(modelMetadata.data_basis_note) +
      '</div>';
    }

    container.innerHTML = html;
  }

  // ========== DATA & ASSUMPTIONS SECTION ==========
  function renderDataSection() {
    var container = document.getElementById('data-content');
    if (!container) return;

    var dataBasis = (modelMetadata && modelMetadata.basis_season) || 2025;
    var dataTarget = (modelMetadata && modelMetadata.target_season) || (dataBasis + 1);
    var dataSrc = (modelMetadata && modelMetadata.data_source) || 'extrapolation';

    var html = '<div class="content-section">' +
      '<h3>Data Sources</h3>' +
      '<ul>' +
        '<li><strong>Player Statistics:</strong> Weekly player performance data from nfl-data-py (official NFL play-by-play and boxscore data)</li>' +
        '<li><strong>Time Range:</strong> ' + (modelMetadata ? modelMetadata.training_data_range : '2006-2024') + ' for ML model training; ' + dataBasis + ' season actuals for draft board ' + (dataSrc === 'ml_model' ? 'reference' : 'estimates') + '</li>' +
        '<li><strong>Play-by-Play:</strong> Advanced metrics including EPA (Expected Points Added), WPA (Win Probability Added), and success rate derived from play-by-play data</li>' +
        '<li><strong>Team Context:</strong> Team-level offensive stats, play volume, pass/rush ratios, and scoring tendencies</li>' +
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
      '<h3>Handling of Edge Cases</h3>' +
      '<ul>' +
        '<li><strong>Missing Games:</strong> Players with fewer than 4 games of data in a season are included but flagged with higher risk scores due to small sample size.</li>' +
        '<li><strong>Team Changes:</strong> Team assignments reflect the player\'s most recent ' + dataBasis + ' team. ' + dataTarget + ' free agency moves and trades may not yet be reflected.</li>' +
        '<li><strong>Rookies:</strong> ' + dataTarget + ' rookies (draft class) are not included since they have no NFL data to extrapolate from.</li>' +
        '<li><strong>Injuries:</strong> No injury model is currently incorporated. Injury history is not factored into projections or risk scores.</li>' +
      '</ul>' +
    '</div>';

    html += '<div class="content-section">' +
      '<h3>Key Assumptions</h3>' +
      '<ul>' +
        '<li>Draft board values assume a 17-game regular season</li>' +
        (dataSrc === 'ml_model'
          ? '<li>ML model predicts ' + dataTarget + ' season performance using ' + dataBasis + ' data as features</li>'
          : '<li>Each player\'s ' + dataBasis + ' per-game average is projected to 17 games (no regression, no ML adjustment)</li>') +
        '<li>Floor and ceiling are calculated as PPG &plusmn; 1.5 standard deviations over 17 games</li>' +
        '<li>ADP values are proxy rankings based on projected total points (not actual draft data)</li>' +
        '<li>Risk scores are relative within each position group and reflect week-to-week consistency, not injury risk</li>' +
      '</ul>' +
    '</div>';

    html += '<div class="notice notice--info">' +
      '<strong>Disclaimer:</strong> Fantasy football projections are inherently uncertain. ' +
      'These numbers should be used as one input among many when making draft decisions. ' +
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
    var html = '';

    if (!isIncorporated) {
      html += '<div class="schedule-status">' +
        '<span class="schedule-status__icon" aria-hidden="true">&#128197;</span>' +
        '<div class="schedule-status__text">' +
          '<h3>Schedule Not Yet Incorporated</h3>' +
          '<p>' + escapeHtml(scheduleImpact ? scheduleImpact.reason : 'The 2026 NFL schedule has not been released.') + '</p>' +
        '</div>' +
      '</div>';

      html += '<div class="content-section">' +
        '<h3>What This Means</h3>' +
        '<p>All projections treat each week as an average matchup based on historical team tendencies. ' +
        'No opponent-specific adjustments have been applied. This means:</p>' +
        '<ul>' +
          '<li>No strength-of-schedule adjustments to player projections</li>' +
          '<li>No home/away performance splits applied</li>' +
          '<li>No bye week information available</li>' +
          '<li>No short-week (Thursday game) adjustments</li>' +
        '</ul>' +
      '</div>';

      html += '<div class="content-section">' +
        '<h3>When the Schedule Is Available</h3>' +
        '<p>Once the NFL releases the schedule (typically in May), the following adjustments will be incorporated:</p>' +
        '<ul>' +
          '<li><strong>Matchup Quality:</strong> Opponent defense strength adjustments per position (e.g., how many fantasy points allowed to QBs/RBs/WRs/TEs)</li>' +
          '<li><strong>Home/Away Splits:</strong> Historical performance differences when playing at home vs. away</li>' +
          '<li><strong>Bye Weeks:</strong> Each player\'s bye week identified for draft planning</li>' +
          '<li><strong>Short Weeks:</strong> Adjustments for Thursday Night Football and other short-week games</li>' +
          '<li><strong>Divisional Matchups:</strong> Historical patterns in divisional rivalry games</li>' +
        '</ul>' +
      '</div>';
    } else {
      html += '<div class="schedule-status">' +
        '<span class="schedule-status__icon" aria-hidden="true">&#9989;</span>' +
        '<div class="schedule-status__text">' +
          '<h3>Schedule Incorporated</h3>' +
          '<p>The 2026 NFL schedule has been incorporated into projections, including matchup strength, bye weeks, and home/away adjustments.</p>' +
        '</div>' +
      '</div>';
    }

    container.innerHTML = html;
  }

  // ========== FAQ SECTION ==========
  function renderFAQ() {
    var container = document.getElementById('faq-content');
    if (!container) return;

    var isScheduleUsed = scheduleImpact && scheduleImpact.schedule_incorporated;
    var faqSrc = (modelMetadata && modelMetadata.data_source) || 'extrapolation';
    var faqBasis = (modelMetadata && modelMetadata.basis_season) || 2025;
    var faqTarget = (modelMetadata && modelMetadata.target_season) || (faqBasis + 1);
    var faqRange = (modelMetadata && modelMetadata.training_data_range) || '';

    var faqs = [
      {
        q: 'How should I use these projections?',
        a: faqSrc === 'ml_model'
          ? 'The Draft Board shows ML model predictions for the ' + faqTarget + ' season, trained on ' + faqRange + ' historical data. ' + faqBasis + ' actual performance is shown in the player detail panel for reference. Use these as a data-driven starting point alongside your own research, expert rankings, and league-specific scoring rules.'
          : 'The Draft Board shows each player\'s ' + faqBasis + ' per-game average projected over 17 games. Use it as a data-driven starting point alongside your own research, expert rankings, and league-specific scoring rules. Because these are straight extrapolations (not ML forecasts), factors like regression to the mean, coaching changes, offseason moves, and age curves are not accounted for.'
      },
      {
        q: 'Are these actual ML model predictions?',
        a: faqSrc === 'ml_model'
          ? 'Yes. The Draft Board uses predictions from the ML pipeline (LightGBM ensemble trained on ' + faqRange + ') to project ' + faqTarget + ' season performance. The Methodology tab shows the model\'s backtest accuracy on the ' + faqBasis + ' held-out validation season.'
          : 'No. The Draft Board currently shows ' + faqBasis + ' actual fantasy points extrapolated to a 17-game season &mdash; essentially assuming each player repeats their ' + faqBasis + ' performance. A separate ML pipeline (LightGBM ensemble trained on ' + faqRange + ') exists and powers the prediction API, but its forward ' + faqTarget + ' projections have not yet replaced the draft board extrapolations. Run <code>python scripts/prepare_new_season.py</code> to generate them.'
      },
      {
        q: 'What do the training/validation seasons mean?',
        a: 'The ML model learned patterns from historical data (' + faqRange + '). It was evaluated against the ' + faqBasis + ' season, which the model had never seen during training. The backtest metrics (RMSE, R\u00b2, etc.) in the Methodology tab measure how well the model predicted ' + faqBasis + ' outcomes. ' + (faqSrc === 'ml_model' ? 'For forward prediction, all available data (including ' + faqBasis + ') is used for training.' : 'This is a standard validation step &mdash; it does not mean the Draft Board numbers come from this model.')
      },
      {
        q: 'Are these projections schedule-adjusted?',
        a: isScheduleUsed
          ? 'Yes. The ' + faqTarget + ' NFL schedule has been incorporated, including opponent defensive strength, home/away adjustments, and bye week identification.'
          : 'No. The ' + faqTarget + ' NFL schedule has not yet been released. All estimates are schedule-neutral, treating each week as an average matchup. Once the schedule is available (typically May), matchup-quality adjustments can be applied.'
      },
      {
        q: 'What do the risk scores mean?',
        a: 'Risk scores range from 0 (lowest risk) to 100 (highest risk) and are calculated relative to each position group. They combine four factors: (1) Weekly scoring volatility (30% weight), (2) Coefficient of variation (25%), (3) Inverse consistency score (25%), and (4) Games played penalty (20%). A low-risk player (0\u201330) is highly consistent; a high-risk player (61\u2013100) has significant week-to-week variance or limited game history.'
      },
      {
        q: 'What scoring format is used?',
        a: 'All values use PPR (Points Per Reception) scoring: 0.04 points per passing yard, 4 points per passing TD, 0.1 points per rushing/receiving yard, 6 points per rushing/receiving TD, 1 point per reception, \u22122 for interceptions and fumbles lost.'
      },
      {
        q: 'Why are some players missing?',
        a: 'Players must have at least one game of ' + faqBasis + ' NFL data to appear. This excludes: (1) ' + faqTarget + ' rookies who haven\'t played an NFL game, (2) Players who missed the entire ' + faqBasis + ' season, (3) Retired players. Free agents and players who changed teams in the ' + faqTarget + ' offseason still show their ' + faqBasis + ' team.'
      },
      {
        q: 'What are the known limitations?',
        a: faqSrc === 'ml_model'
          ? 'Key limitations: (1) No injury model, (2) ADP values are rank-based proxies (not actual draft data), (3) ' + faqTarget + ' roster updates may not be fully reflected, (4) No schedule adjustments (waiting for NFL schedule release), (5) Rookies excluded (no historical data), (6) ML model does not account for coaching changes or scheme shifts.'
          : 'Key limitations: (1) Draft board uses ' + faqBasis + ' extrapolations, not ML forward predictions, (2) No regression-to-the-mean or age adjustments, (3) No injury model, (4) ADP values are rank-based proxies (not actual draft data), (5) No ' + faqTarget + ' roster updates (free agency, trades not reflected), (6) No schedule adjustments (waiting for NFL schedule release), (7) Rookies excluded (no historical data).'
      }
    ];

    container.innerHTML = faqs.map(function (faq) {
      return '<div class="faq-item">' +
        '<h3>' + escapeHtml(faq.q) + '</h3>' +
        '<p>' + faq.a + '</p>' +
      '</div>';
    }).join('');
  }

  // ========== DYNAMIC CHROME (subtitle, notice, footer) ==========
  function updateDynamicChrome() {
    var src = (modelMetadata && modelMetadata.data_source) || 'extrapolation';
    var basis = (modelMetadata && modelMetadata.basis_season) || 2025;
    var target = (modelMetadata && modelMetadata.target_season) || (basis + 1);
    var trainingRange = (modelMetadata && modelMetadata.training_data_range) || '';

    // Subtitle
    var subtitleEl = document.getElementById('header-subtitle');
    if (subtitleEl) {
      if (src === 'ml_model') {
        subtitleEl.textContent = target + ' ML projections trained on ' + trainingRange;
      } else {
        subtitleEl.textContent = target + ' projections based on ' + basis + ' season performance';
      }
    }

    // Badge
    var badgeEl = document.getElementById('header-badge');
    if (badgeEl) {
      badgeEl.textContent = 'Pre-Draft ' + target;
    }

    // Notice
    var noticeEl = document.getElementById('data-notice');
    if (noticeEl) {
      if (src === 'ml_model') {
        noticeEl.innerHTML =
          '<strong>ML-powered projections:</strong> Player values are generated by the ML model ' +
          '(trained on ' + escapeHtml(trainingRange) + ' historical data) predicting ' + target +
          ' season performance. ' + basis + ' actual stats shown in the player detail panel for reference.';
      } else {
        noticeEl.innerHTML =
          '<strong>How to read this board:</strong> Player values are estimated by taking their ' +
          'actual ' + basis + ' per-game fantasy points and projecting over a 17-game season. These are ' +
          '<em>not</em> ML model predictions &mdash; they assume ' + target + ' performance will match ' + basis +
          '. The ML model\'s backtest metrics are shown in the Methodology tab.';
      }
    }

    // Footer
    var footerEl = document.getElementById('app-footer-text');
    if (footerEl) {
      if (src === 'ml_model') {
        footerEl.textContent = 'NFL Fantasy Draft Board \u00b7 ML model: trained ' +
          trainingRange + ', predicting ' + target + ' season';
      } else {
        footerEl.textContent = 'NFL Fantasy Draft Board \u00b7 Draft board: ' +
          basis + ' actuals extrapolated to 17 games \u00b7 ML model backtest available in Methodology';
      }
    }
  }

  // ========== INITIALIZATION ==========
  function init() {
    loadAllData()
      .then(function () {
        updateDynamicChrome();
        initTabs();
        renderOverview();
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
