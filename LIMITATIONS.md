# Fantasy Predictor - Limitations & Improvement Opportunities

## Prediction Horizon Analysis

This document identifies limitations for different prediction horizons (1 week, 5 weeks, 18 weeks) from both **Football SME** and **Data Scientist** perspectives.

---

## 1-Week Prediction Horizon (Weekly Start/Sit Decisions)

### Current Capabilities ✅
- Rolling 3-week averages for recent form
- Utilization score (target share, rush share, snap share)
- QB-specific features (passer rating, completion %)
- Uncertainty quantification (floor/ceiling)

### Limitations & Improvements Needed

#### Football SME Perspective 🏈

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No injury status integration** | Can't account for questionable/doubtful players | Integrate injury reports from ESPN/NFL API |
| **No weather data** | Outdoor games affected by rain/wind/cold | Add weather API for game-time conditions |
| **No Vegas lines/implied totals** | Missing game script expectations | Integrate The Odds API for spreads/totals |
| **No opponent defense ranking** | Doesn't adjust for matchup difficulty | Add defense-vs-position rankings |
| **No bye week handling** | Players returning from bye may perform differently | Add bye week indicator and post-bye adjustment |
| **No primetime game adjustment** | Some players perform differently in primetime | Add game time slot feature |

#### Data Scientist Perspective 📊

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **Single point prediction** | Users need ranges, not just averages | ✅ Already implemented (floor/ceiling) |
| **No confidence intervals by player tier** | Elite players have different variance than backups | Add tier-specific uncertainty bounds |
| **No recent news/sentiment** | Can't capture breaking news impact | Add NLP on recent news articles |

---

## 5-Week Prediction Horizon (Trade Deadline / Mid-Season)

### Current Capabilities ✅
- Season-to-date averages
- Trend features (improving/declining)
- Consistency scores

### Limitations & Improvements Needed

#### Football SME Perspective 🏈

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No schedule strength analysis** | Next 5 weeks may be easy or hard | Add remaining schedule difficulty score |
| **No coaching/scheme changes** | Mid-season OC changes affect usage | Track coaching changes and adjust |
| **No trade deadline impact** | Players traded mid-season need adjustment | Detect team changes and reset projections |
| **No playoff implications** | Teams may rest starters late season | Add playoff clinch/elimination status |

#### Data Scientist Perspective 📊

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No multi-week aggregation model** | Currently just multiplies weekly projection | Train separate model for 5-week totals |
| **No injury probability modeling** | Can't account for injury risk over 5 weeks | Add injury probability based on position/usage |
| **No regression to mean over time** | Hot streaks don't last 5 weeks | Add mean reversion factor for longer horizons |

---

## 18-Week Prediction Horizon (Season-Long / Draft)

### Current Capabilities ✅
- Historical season totals
- Value Over Replacement (VOR)
- Position-specific rankings

### Limitations & Improvements Needed

#### Football SME Perspective 🏈

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No ADP (Average Draft Position) integration** | Can't identify value picks | Add ADP data from FantasyPros/ESPN |
| **No strength of schedule** | Season-long matchup difficulty matters | Calculate full-season SOS |
| **No rookie projections** | New players have no history | Use draft capital + college stats |
| **No offseason changes** | Free agency/trades affect projections | Track offseason moves and adjust |
| **No suspension risk** | Some players have suspension history | Add suspension probability |
| **No age/decline curves** | Older RBs decline faster | Add age-based adjustment by position |

#### Data Scientist Perspective 📊

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No games played projection** | Assumes 17 games, but injuries happen | Model expected games played |
| **No touchdown regression** | TDs are high-variance, should regress | ✅ Partially implemented (expected TDs) |
| **No opportunity share projection** | Usage can change during season | Model expected usage changes |
| **No team-level projections** | Individual projections without team context | Add team offensive projections |

---

## Cross-Horizon Limitations

### Data Limitations

| Issue | Current State | Needed |
|-------|---------------|--------|
| **2025 season data** | Pending in nflverse | Will auto-load when available |
| **Real-time updates** | Manual refresh | Automatic scheduled refresh |
| **Snap count data** | Available but not fully integrated | Full snap count integration |
| **Red zone data** | Estimated from TDs | Actual red zone opportunities |
| **Air yards data** | Estimated | Actual air yards from nflverse |

### Model Limitations

| Issue | Current State | Needed |
|-------|---------------|--------|
| **Single model per position** | Same model for all horizons | Horizon-specific models |
| **No ensemble weighting by horizon** | Fixed weights | Adaptive weights by prediction length |
| **No player-specific models** | Same model for all players | Tier-based or player-specific adjustments |

---

## Priority Improvements by Use Case

### For Weekly Start/Sit (1 Week)
1. 🔴 **HIGH**: Injury status integration
2. 🔴 **HIGH**: Opponent defense rankings
3. 🟡 **MEDIUM**: Weather data
4. 🟡 **MEDIUM**: Vegas implied totals

### For Trade Analysis (5 Weeks)
1. 🔴 **HIGH**: Schedule strength analysis
2. 🔴 **HIGH**: Multi-week aggregation model
3. 🟡 **MEDIUM**: Injury probability modeling

### For Draft Preparation (18 Weeks)
1. 🔴 **HIGH**: ADP integration for value analysis
2. 🔴 **HIGH**: Rookie projections
3. 🔴 **HIGH**: Games played projection
4. 🟡 **MEDIUM**: Age/decline curves

---

## Implementation Roadmap

### Phase 1: Data Enrichment (Recommended First)
- [ ] Integrate injury reports (ESPN API)
- [ ] Add defense-vs-position rankings
- [ ] Integrate Vegas lines (The Odds API)
- [ ] Add ADP data (FantasyPros)

### Phase 2: Model Improvements
- [ ] Train horizon-specific models (1w, 5w, 18w)
- [ ] Add games played projection
- [ ] Implement schedule strength scoring
- [ ] Add rookie projection module

### Phase 3: Advanced Features
- [ ] Weather integration
- [ ] Real-time news sentiment
- [ ] Player-specific uncertainty bounds
- [ ] Age/decline curve modeling

---

*Last updated: 2026-01-31*
