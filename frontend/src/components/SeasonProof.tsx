import { useEffect, useState, useMemo } from 'react'
import {
  api,
  type BacktestSeasonResponse,
  type TSBacktestPredictionRow,
} from '../api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, CartesianGrid, Legend,
  ComposedChart, Line,
} from 'recharts'

const POSITIONS = ['QB', 'RB', 'WR', 'TE'] as const

const POS_COLORS: Record<string, string> = {
  QB: 'var(--color-accent-cyan)',
  RB: 'var(--color-accent-emerald)',
  WR: 'var(--color-accent-purple)',
  TE: 'var(--color-accent-amber)',
}

export function SeasonProof() {
  const [availableSeasons, setAvailableSeasons] = useState<number[]>([])
  const [selectedSeason, setSelectedSeason] = useState<number | null>(null)
  const [seasonData, setSeasonData] = useState<BacktestSeasonResponse | null>(null)
  const [predictions, setPredictions] = useState<TSBacktestPredictionRow[]>([])
  const [loading, setLoading] = useState(true)
  const [predLoading, setPredLoading] = useState(false)
  const [playerSearch, setPlayerSearch] = useState('')
  const [selectedPos, setSelectedPos] = useState<string>('ALL')

  const season = selectedSeason

  // Discover available seasons from backtest files
  useEffect(() => {
    api.backtest()
      .then((meta) => {
        const seasons = Array.from(new Set(
          (meta.files ?? []).map(f => f.season).filter((s): s is number => s != null)
        )).sort((a, b) => b - a) // newest first
        setAvailableSeasons(seasons)
        if (seasons.length > 0 && selectedSeason == null) {
          setSelectedSeason(seasons[0])
        }
      })
      .catch(() => {})
  }, [])

  // Load season-specific backtest data when season changes
  useEffect(() => {
    if (selectedSeason == null) return
    setLoading(true)
    setSeasonData(null)
    api.backtestSeason(selectedSeason)
      .then(setSeasonData)
      .catch(() => setSeasonData(null))
      .finally(() => setLoading(false))
  }, [selectedSeason])

  // Load player-level predictions from TS backtest
  useEffect(() => {
    if (!season) return
    setPredLoading(true)
    setPredictions([])
    api.tsBacktestPredictions(season)
      .then((d) => setPredictions(d.rows ?? []))
      .catch(() => setPredictions([]))
      .finally(() => setPredLoading(false))
  }, [season])

  const metrics = seasonData?.metrics ?? {}
  const topPerformers = seasonData?.top_performers ?? {}
  const biggestMisses = seasonData?.biggest_misses ?? []
  const rankingAccuracy = seasonData?.ranking_accuracy ?? {}

  // Filtered player predictions for search
  const filteredPredictions = useMemo(() => {
    let rows = predictions
    if (selectedPos !== 'ALL') {
      rows = rows.filter(r => r.position?.toUpperCase() === selectedPos)
    }
    if (playerSearch.trim()) {
      const q = playerSearch.trim().toLowerCase()
      rows = rows.filter(r => r.name?.toLowerCase().includes(q))
    }
    return rows
  }, [predictions, playerSearch, selectedPos])

  // Player names for autocomplete
  const playerNames = useMemo(() => {
    const names = new Set<string>()
    for (const r of predictions) {
      if (r.name) names.add(r.name)
    }
    return Array.from(names).sort()
  }, [predictions])

  // Suggestions for search
  const suggestions = useMemo(() => {
    if (!playerSearch.trim() || playerSearch.length < 2) return []
    const q = playerSearch.toLowerCase()
    return playerNames.filter(n => n.toLowerCase().includes(q)).slice(0, 8)
  }, [playerSearch, playerNames])

  // Per-player weekly data for the searched player
  const playerWeeklyData = useMemo(() => {
    if (!playerSearch.trim()) return []
    const q = playerSearch.trim().toLowerCase()
    const exact = filteredPredictions.filter(r => r.name?.toLowerCase() === q)
    const rows = exact.length > 0 ? exact : filteredPredictions.filter(r => r.name?.toLowerCase().includes(q))
    if (rows.length === 0) return []
    // Group by the first matching player name
    const targetName = rows[0].name
    return rows
      .filter(r => r.name === targetName && r.predicted != null && r.actual != null)
      .map(r => ({
        week: r.week ?? 0,
        predicted: r.predicted!,
        actual: r.actual!,
        name: r.name ?? '',
        position: r.position ?? '',
      }))
      .sort((a, b) => a.week - b.week)
  }, [filteredPredictions, playerSearch])

  // Scatter data for all predictions (sampled)
  const scatterData = useMemo(() => {
    let rows = predictions.filter(r => r.predicted != null && r.actual != null)
    if (selectedPos !== 'ALL') {
      rows = rows.filter(r => r.position?.toUpperCase() === selectedPos)
    }
    if (rows.length > 2500) {
      const step = Math.ceil(rows.length / 2500)
      rows = rows.filter((_, i) => i % step === 0)
    }
    return rows.map(r => ({
      predicted: r.predicted!,
      actual: r.actual!,
      name: r.name ?? '',
      position: r.position ?? '',
      week: r.week ?? 0,
    }))
  }, [predictions, selectedPos])

  if (loading) {
    return (
      <div className="section-card">
        <h2>Last Season Results</h2>
        <div className="skeleton" style={{ height: 400 }} />
      </div>
    )
  }

  if (!seasonData && !loading) {
    return (
      <div className="section-card">
        <h2>Last Season Results</h2>
        <p style={{ color: 'var(--color-text-muted)' }}>
          No backtest results available. Run the backtester to generate last-season proof data.
        </p>
      </div>
    )
  }

  const rmse = (metrics as Record<string, number>).rmse
  const mae = (metrics as Record<string, number>).mae
  const within10 = (metrics as Record<string, number>).within_10_pts_pct
  const within5 = (metrics as Record<string, number>).within_5_pts_pct
  const dirAcc = (metrics as Record<string, number>).directional_accuracy_pct
  const corr = (metrics as Record<string, number>).correlation
  const nPred = seasonData?.n_predictions

  return (
    <div className="section-card">
      {/* ───── Hero Banner ───── */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(0,245,255,0.08) 0%, rgba(167,139,250,0.08) 100%)',
        border: '1px solid rgba(0,245,255,0.2)',
        borderRadius: 12,
        padding: '1.5rem 2rem',
        marginBottom: '2rem',
        textAlign: 'center',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
          <h2 style={{ fontSize: 'var(--text-h2)', color: 'var(--color-text-primary)', margin: 0 }}>
            {season ? `${season} Season Backtest` : 'Last Season Results'}
          </h2>
          {availableSeasons.length > 1 && (
            <select
              value={selectedSeason ?? ''}
              onChange={(e) => setSelectedSeason(Number(e.target.value))}
              style={{
                background: 'var(--color-card)',
                color: 'var(--color-text-primary)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 6,
                padding: '0.35rem 0.6rem',
                fontSize: 'var(--text-small)',
              }}
            >
              {availableSeasons.map((s) => (
                <option key={s} value={s}>{s} Season</option>
              ))}
            </select>
          )}
        </div>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: '1.25rem', fontSize: 'var(--text-body)' }}>
          Out-of-sample predictions on a full season the model never trained on. Every number below is a genuine test — no data leakage, no peeking.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
          {[
            { label: 'Predictions', value: nPred?.toLocaleString(), color: 'var(--color-accent-cyan)' },
            { label: 'RMSE', value: rmse?.toFixed(2), color: 'var(--color-accent-purple)' },
            { label: 'MAE', value: mae?.toFixed(2), color: 'var(--color-accent-emerald)' },
            { label: 'Within 10 pts', value: within10 != null ? `${within10.toFixed(1)}%` : undefined, color: 'var(--color-accent-amber)' },
            { label: 'Directional Acc', value: dirAcc != null ? `${dirAcc.toFixed(1)}%` : undefined, color: 'var(--color-accent-sky)' },
            { label: 'Correlation', value: corr?.toFixed(3), color: 'var(--color-accent-lime)' },
          ].filter(c => c.value != null).map((card) => (
            <div key={card.label}>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: card.color }}>
                {card.value}
              </div>
              <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>{card.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ───── Ranking Accuracy Cards ───── */}
      {Object.keys(rankingAccuracy).length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '1rem' }}>
            Ranking Accuracy by Position
          </h3>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', fontSize: 'var(--text-small)' }}>
            How often did players we ranked in the top-N actually finish in the top-N?
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            {POSITIONS.filter(p => rankingAccuracy[p]).map((pos) => {
              const ra = rankingAccuracy[pos]!
              return (
                <div key={pos} style={{
                  background: 'var(--color-bg)',
                  border: '1px solid var(--color-card-border)',
                  borderRadius: 10,
                  padding: '1rem',
                }}>
                  <div style={{ fontWeight: 700, fontSize: '1.1rem', color: POS_COLORS[pos], marginBottom: '0.75rem' }}>
                    {pos}
                  </div>
                  {([
                    ['Top 5', ra.top_5_hit_rate],
                    ['Top 10', ra.top_10_hit_rate],
                    ['Top 20', ra.top_20_hit_rate],
                  ] as [string, number | undefined][]).map(([label, val]) => (
                    <div key={label} style={{ marginBottom: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--text-small)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
                        <span>{label}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{val != null ? `${val.toFixed(1)}%` : '–'}</span>
                      </div>
                      <div style={{ height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.min(val ?? 0, 100)}%`,
                          background: POS_COLORS[pos],
                          borderRadius: 3,
                          transition: 'width 0.6s ease',
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ───── Top Calls (Top Performers) ───── */}
      {Object.keys(topPerformers).length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Top Calls: Actual Top 10 vs Our Predictions
          </h3>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', fontSize: 'var(--text-small)' }}>
            How did we rank the players who actually finished in the top 10 at each position?
          </p>
          {POSITIONS.filter(p => topPerformers[p]?.top_10_actual?.length).map((pos) => {
            const perf = topPerformers[pos]!
            const players = perf.top_10_actual ?? []
            return (
              <div key={pos} style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 700, color: POS_COLORS[pos], fontSize: '1rem' }}>{pos}</span>
                  {perf.avg_pred_rank_of_top_10 != null && (
                    <span style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                      Avg predicted rank of top 10: <strong style={{ color: 'var(--color-text-secondary)' }}>#{perf.avg_pred_rank_of_top_10.toFixed(1)}</strong>
                    </span>
                  )}
                  {perf.top_10_in_our_top_20 != null && (
                    <span style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                      · {perf.top_10_in_our_top_20} of 10 in our top 20
                    </span>
                  )}
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-small)' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--color-card-border)' }}>
                        <th style={thStyle}>Actual Rank</th>
                        <th style={thStyle}>Player</th>
                        <th style={thStyle}>Our Rank</th>
                        <th style={thStyle}>Actual Pts</th>
                        <th style={thStyle}>Predicted Pts</th>
                        <th style={thStyle}>Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {players.map((p, i) => {
                        const rankDiff = Math.abs(p.actual_rank - p.pred_rank)
                        const closeRank = rankDiff <= 3
                        return (
                          <tr key={i} style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
                            <td style={tdStyle}>#{p.actual_rank}</td>
                            <td style={{ ...tdStyle, fontWeight: 600, color: 'var(--color-text-primary)' }}>{p.name}</td>
                            <td style={{ ...tdStyle, color: closeRank ? 'var(--color-accent-emerald)' : rankDiff <= 8 ? 'var(--color-accent-amber)' : 'var(--color-accent-rose)' }}>
                              #{p.pred_rank}
                              {closeRank && ' ✓'}
                            </td>
                            <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{p.fantasy_points.toFixed(1)}</td>
                            <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{p.predicted_points.toFixed(1)}</td>
                            <td style={tdStyle}>
                              <span style={{
                                display: 'inline-block',
                                padding: '2px 8px',
                                borderRadius: 4,
                                fontSize: '0.7rem',
                                fontWeight: 600,
                                background: closeRank ? 'rgba(16,185,129,0.15)' : rankDiff <= 8 ? 'rgba(251,191,36,0.15)' : 'rgba(251,113,133,0.15)',
                                color: closeRank ? 'var(--color-accent-emerald)' : rankDiff <= 8 ? 'var(--color-accent-amber)' : 'var(--color-accent-rose)',
                              }}>
                                {closeRank ? 'Nailed it' : rankDiff <= 8 ? 'Close' : `Off by ${rankDiff}`}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ───── Biggest Misses ───── */}
      {biggestMisses.length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Biggest Misses (Transparency)
          </h3>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', fontSize: 'var(--text-small)' }}>
            Our worst single-week predictions. Fantasy football is inherently unpredictable — here are the outlier games no model could see coming.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-small)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-card-border)' }}>
                  <th style={thStyle}>Player</th>
                  <th style={thStyle}>Pos</th>
                  <th style={thStyle}>Team</th>
                  <th style={thStyle}>Week</th>
                  <th style={thStyle}>Predicted</th>
                  <th style={thStyle}>Actual</th>
                  <th style={thStyle}>Error</th>
                </tr>
              </thead>
              <tbody>
                {biggestMisses.slice(0, 10).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, color: 'var(--color-text-primary)' }}>{m.name}</td>
                    <td style={{ ...tdStyle, color: POS_COLORS[m.position] ?? 'var(--color-text-secondary)' }}>{m.position}</td>
                    <td style={tdStyle}>{m.team}</td>
                    <td style={tdStyle}>{m.week}</td>
                    <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{m.predicted_points.toFixed(1)}</td>
                    <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{m.fantasy_points.toFixed(1)}</td>
                    <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)', color: 'var(--color-accent-rose)' }}>
                      {m.error.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ───── Player Search ───── */}
      <div style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
          Player Lookup: Week-by-Week Accuracy
        </h3>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', fontSize: 'var(--text-small)' }}>
          Search any player to see how our model performed week by week throughout the {season} season.
        </p>
        {!predLoading && predictions.length === 0 && (
          <div style={{
            padding: '1rem',
            background: 'var(--color-bg)',
            border: '1px solid var(--color-card-border)',
            borderRadius: 8,
            color: 'var(--color-text-muted)',
            fontSize: 'var(--text-small)',
            marginBottom: '1rem',
          }}>
            Player-level week-by-week data is not available for the {season} season.
            {availableSeasons.length > 1 && ' Try switching to another season using the dropdown above.'}
          </div>
        )}
        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: '1 1 280px' }}>
            <input
              type="text"
              placeholder="Search player name..."
              value={playerSearch}
              onChange={(e) => setPlayerSearch(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem 0.75rem',
                background: 'var(--color-bg)',
                color: 'var(--color-text-primary)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 8,
                fontSize: 'var(--text-body)',
                outline: 'none',
              }}
            />
            {suggestions.length > 0 && playerSearch.length >= 2 && (
              <div style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                background: 'var(--color-card)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 8,
                zIndex: 10,
                maxHeight: 200,
                overflowY: 'auto',
                marginTop: 2,
              }}>
                {suggestions.map((name) => (
                  <div
                    key={name}
                    onClick={() => setPlayerSearch(name)}
                    style={{
                      padding: '0.4rem 0.75rem',
                      cursor: 'pointer',
                      fontSize: 'var(--text-small)',
                      color: 'var(--color-text-secondary)',
                      borderBottom: '1px solid rgba(51,65,85,0.3)',
                    }}
                    onMouseEnter={(e) => {
                      (e.target as HTMLDivElement).style.background = 'rgba(0,245,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      (e.target as HTMLDivElement).style.background = 'transparent'
                    }}
                  >
                    {name}
                  </div>
                ))}
              </div>
            )}
          </div>
          <select
            value={selectedPos}
            onChange={(e) => setSelectedPos(e.target.value)}
            style={{
              background: 'var(--color-card)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-card-border)',
              borderRadius: 6,
              padding: '0.5rem 0.75rem',
              fontSize: 'var(--text-small)',
            }}
          >
            <option value="ALL">All Positions</option>
            {POSITIONS.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        {predLoading && <p style={{ color: 'var(--color-text-muted)', textAlign: 'center' }}>Loading player predictions...</p>}

        {/* Player weekly chart */}
        {playerWeeklyData.length > 0 && (
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ fontSize: '1rem', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
              {playerWeeklyData[0].name} ({playerWeeklyData[0].position}) — Predicted vs Actual
            </h4>
            <div style={{ height: 280 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={playerWeeklyData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                  <XAxis dataKey="week" stroke="var(--color-text-muted)" label={{ value: 'Week', position: 'bottom', fill: 'var(--color-text-muted)' }} />
                  <YAxis stroke="var(--color-text-muted)" label={{ value: 'Fantasy Pts', angle: -90, position: 'insideLeft', fill: 'var(--color-text-muted)' }} />
                  <Tooltip
                    contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                    formatter={(value: number, name: string) => [value.toFixed(1), name === 'predicted' ? 'Predicted' : 'Actual']}
                    labelFormatter={(l) => `Week ${l}`}
                  />
                  <Legend formatter={(v) => v === 'predicted' ? 'Predicted' : 'Actual'} />
                  <Bar dataKey="actual" fill="var(--color-accent-emerald)" fillOpacity={0.5} radius={[3, 3, 0, 0]} name="actual" />
                  <Line type="monotone" dataKey="predicted" stroke="var(--color-accent-cyan)" strokeWidth={2.5} dot={{ fill: 'var(--color-accent-cyan)', r: 4 }} name="predicted" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            {/* Stats summary for this player */}
            {(() => {
              const errs = playerWeeklyData.map(d => Math.abs(d.predicted - d.actual))
              const avgErr = errs.reduce((a, b) => a + b, 0) / errs.length
              const within5Count = errs.filter(e => e <= 5).length
              return (
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.5rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                  <span>Weeks: <strong style={{ color: 'var(--color-text-secondary)' }}>{playerWeeklyData.length}</strong></span>
                  <span>Avg Error: <strong style={{ color: 'var(--color-text-secondary)' }}>{avgErr.toFixed(1)} pts</strong></span>
                  <span>Within 5 pts: <strong style={{ color: 'var(--color-text-secondary)' }}>{within5Count}/{playerWeeklyData.length} ({(within5Count / playerWeeklyData.length * 100).toFixed(0)}%)</strong></span>
                </div>
              )
            })()}
          </div>
        )}

        {/* Player weekly table */}
        {playerWeeklyData.length > 0 && (
          <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-small)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-card-border)', position: 'sticky', top: 0, background: 'var(--color-card)' }}>
                  <th style={thStyle}>Week</th>
                  <th style={thStyle}>Predicted</th>
                  <th style={thStyle}>Actual</th>
                  <th style={thStyle}>Error</th>
                </tr>
              </thead>
              <tbody>
                {playerWeeklyData.map((d) => {
                  const err = Math.abs(d.predicted - d.actual)
                  return (
                    <tr key={d.week} style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
                      <td style={tdStyle}>{d.week}</td>
                      <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{d.predicted.toFixed(1)}</td>
                      <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)' }}>{d.actual.toFixed(1)}</td>
                      <td style={{
                        ...tdStyle,
                        fontFamily: 'var(--font-mono)',
                        color: err <= 3 ? 'var(--color-accent-emerald)' : err <= 7 ? 'var(--color-accent-amber)' : 'var(--color-accent-rose)',
                      }}>
                        {err.toFixed(1)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        {playerSearch.trim() && playerWeeklyData.length === 0 && !predLoading && (
          <p style={{ color: 'var(--color-text-muted)', textAlign: 'center', padding: '1rem' }}>
            No predictions found for "{playerSearch}" {selectedPos !== 'ALL' ? `at ${selectedPos}` : ''}.
          </p>
        )}
      </div>

      {/* ───── Predicted vs Actual Scatter ───── */}
      {scatterData.length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Predicted vs Actual — Full Season ({scatterData.length.toLocaleString()} predictions)
          </h3>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem', fontSize: 'var(--text-small)' }}>
            Each dot is one player-week. The closer to the diagonal, the more accurate our prediction.
            {selectedPos !== 'ALL' ? ` Showing ${selectedPos} only.` : ''}
          </p>
          <div style={{ height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                <XAxis
                  dataKey="predicted" name="Predicted" type="number"
                  stroke="var(--color-text-muted)"
                  label={{ value: 'Predicted', position: 'bottom', fill: 'var(--color-text-muted)' }}
                />
                <YAxis
                  dataKey="actual" name="Actual" type="number"
                  stroke="var(--color-text-muted)"
                  label={{ value: 'Actual', angle: -90, position: 'insideLeft', fill: 'var(--color-text-muted)' }}
                />
                <Tooltip
                  contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                  formatter={(value: number, name: string) => [value.toFixed(1), name]}
                  labelFormatter={() => ''}
                  content={({ payload }) => {
                    if (!payload?.length) return null
                    const d = payload[0]?.payload as { name: string; position: string; week: number; predicted: number; actual: number } | undefined
                    if (!d) return null
                    return (
                      <div style={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8, padding: '0.5rem 0.75rem', fontSize: 'var(--text-small)' }}>
                        <div style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>{d.name} ({d.position})</div>
                        <div style={{ color: 'var(--color-text-muted)' }}>Week {d.week}</div>
                        <div style={{ color: 'var(--color-accent-cyan)' }}>Predicted: {d.predicted.toFixed(1)}</div>
                        <div style={{ color: 'var(--color-accent-emerald)' }}>Actual: {d.actual.toFixed(1)}</div>
                      </div>
                    )
                  }}
                />
                <Scatter data={scatterData} fill="var(--color-accent-cyan)" fillOpacity={0.4} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ───── Within-N Accuracy Chart ───── */}
      {(within5 != null || within10 != null) && (
        <div>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Prediction Accuracy Thresholds
          </h3>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[
                  ...(metrics.within_3_pts_pct != null ? [{ name: 'Within 3 pts', value: metrics.within_3_pts_pct }] : []),
                  ...(within5 != null ? [{ name: 'Within 5 pts', value: within5 }] : []),
                  ...(within10 != null ? [{ name: 'Within 10 pts', value: within10 }] : []),
                  ...(dirAcc != null ? [{ name: 'Direction correct', value: dirAcc }] : []),
                ]}
                margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                <XAxis dataKey="name" stroke="var(--color-text-muted)" />
                <YAxis stroke="var(--color-text-muted)" domain={[0, 105]} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Rate']}
                />
                <Bar dataKey="value" fill="var(--color-accent-purple)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '0.4rem 0.6rem',
  color: 'var(--color-text-muted)',
  fontWeight: 600,
  fontSize: 'var(--text-small)',
  whiteSpace: 'nowrap',
}

const tdStyle: React.CSSProperties = {
  padding: '0.4rem 0.6rem',
  color: 'var(--color-text-secondary)',
  whiteSpace: 'nowrap',
}
