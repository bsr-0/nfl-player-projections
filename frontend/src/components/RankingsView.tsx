import { useEffect, useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList, CartesianGrid } from 'recharts'
import { api, type PredictionsResponse, type PredictionRow, type TSBacktestPredictionRow } from '../api'

const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'] as const
type HorizonValue = 1 | 4 | 18
const HORIZONS: { value: HorizonValue; label: string }[] = [
  { value: 1, label: '1 Week' },
  { value: 4, label: '4 Weeks' },
  { value: 18, label: 'Rest of Season' },
]

const POSITION_COLORS: Record<string, string> = {
  QB: '#00f5ff',
  RB: '#a78bfa',
  WR: '#10b981',
  TE: '#fbbf24',
  K: '#f472b6',
  DST: '#fb923c',
}

function getTier(rank: number, total: number): { label: string; color: string; bg: string; num: number } {
  const pct = rank / total
  if (pct <= 0.08) return { label: 'Must Start', color: '#10b981', bg: 'rgba(16,185,129,0.15)', num: 1 }
  if (pct <= 0.25) return { label: 'Start', color: '#00f5ff', bg: 'rgba(0,245,255,0.1)', num: 2 }
  if (pct <= 0.5) return { label: 'Flex', color: '#fbbf24', bg: 'rgba(251,191,36,0.1)', num: 3 }
  if (pct <= 0.75) return { label: 'Bench', color: '#fb923c', bg: 'rgba(251,146,60,0.08)', num: 4 }
  return { label: 'Sit', color: '#94a3b8', bg: 'rgba(148,163,184,0.08)', num: 5 }
}

function getProjectedPoints(r: PredictionRow, horizon: HorizonValue): number | null {
  if (horizon === 1) {
    const v = r.projected_points ?? r.predicted_points ?? r.projection_1w
    return v != null && !Number.isNaN(Number(v)) ? Number(v) : null
  }
  if (horizon === 4) {
    const v = r.projection_4w ?? r.projected_points
    return v != null && !Number.isNaN(Number(v)) ? Number(v) : null
  }
  const v = r.projection_18w ?? r.projected_points
  return v != null && !Number.isNaN(Number(v)) ? Number(v) : null
}

function formatMatchup(row: PredictionRow): string {
  const opp = row.upcoming_opponent ?? ''
  const ha = row.upcoming_home_away ?? ''
  if (!opp) return '—'
  if (ha === 'home') return `vs ${opp}`
  if (ha === 'away') return `@ ${opp}`
  return opp
}

interface RankingsViewProps {
  data: PredictionsResponse | null
  position: string
  horizon: HorizonValue
  loading: boolean
  error: string | null
  onPositionChange: (pos: string) => void
  onHorizonChange: (h: HorizonValue) => void
}

type DataMode = 'live' | 'backtest'

export function RankingsView({ data, position, horizon, loading, error, onPositionChange, onHorizonChange }: RankingsViewProps) {
  const [searchName, setSearchName] = useState('')
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table')
  const [dataMode, setDataMode] = useState<DataMode>('live')
  const [backtestSeasons, setBacktestSeasons] = useState<number[]>([])
  const [selectedBtSeason, setSelectedBtSeason] = useState<number | null>(null)
  const [btRows, setBtRows] = useState<TSBacktestPredictionRow[]>([])
  const [btLoading, setBtLoading] = useState(false)
  const isFPPosition = position === 'QB' || position === 'K' || position === 'DST'

  // Fetch available backtest seasons once
  useEffect(() => {
    api.tsBacktest()
      .then((d) => {
        setBacktestSeasons(d.available_seasons ?? [])
        if (d.available_seasons?.length) setSelectedBtSeason(d.available_seasons[d.available_seasons.length - 1])
      })
      .catch(() => {})
  }, [])

  // Fetch backtest predictions when mode/season/position changes
  useEffect(() => {
    if (dataMode !== 'backtest' || selectedBtSeason == null) return
    setBtLoading(true)
    api.tsBacktestPredictions(selectedBtSeason, position)
      .then((d) => setBtRows(d.rows ?? []))
      .catch(() => setBtRows([]))
      .finally(() => setBtLoading(false))
  }, [dataMode, selectedBtSeason, position])

  const rows = data?.rows ?? []

  const enriched = useMemo(() => {
    let items = rows
      .map((r) => {
        const pts = getProjectedPoints(r, horizon)
        const util = r.utilization_score != null ? Number(r.utilization_score) : null
        const std = r.prediction_std != null ? Number(r.prediction_std) : (r.weekly_volatility != null ? Number(r.weekly_volatility) : null)
        const chartVal = isFPPosition ? pts : (util ?? pts)
        return { r, pts, util, std, chartVal, matchup: formatMatchup(r) }
      })
      .filter((x) => x.chartVal != null)
      .sort((a, b) => b.chartVal! - a.chartVal!)

    if (searchName.trim()) {
      const q = searchName.trim().toLowerCase()
      items = items.filter((x) => String(x.r.name ?? '').toLowerCase().includes(q))
    }

    const total = items.length
    return items.map((x, i) => ({
      ...x,
      rank: i + 1,
      tier: getTier(i + 1, total),
      floor: x.pts != null && x.std != null ? Math.max(0, x.pts - 1.5 * x.std) : null,
      ceiling: x.pts != null && x.std != null ? x.pts + 1.5 * x.std : null,
    }))
  }, [rows, horizon, isFPPosition, searchName])

  const chartData = useMemo(
    () =>
      enriched.slice(0, 30).map((x) => ({
        name: `${x.r.name ?? 'Unknown'} — ${x.r.team ?? '—'}`,
        value: Number(x.chartVal),
        position: x.r.position,
        tier: x.tier,
      })),
    [enriched],
  )

  const downloadCsv = () => {
    const headers = ['Rank', 'Name', 'Position', 'Team', 'Projected Pts', 'Floor', 'Ceiling', 'Tier', 'Matchup']
    const csvRows = enriched.slice(0, 200).map((x) => [
      x.rank,
      x.r.name ?? '',
      x.r.position ?? '',
      x.r.team ?? '',
      x.pts?.toFixed(1) ?? '',
      x.floor?.toFixed(1) ?? '',
      x.ceiling?.toFixed(1) ?? '',
      x.tier.label,
      x.matchup,
    ])
    const csv = [headers.join(','), ...csvRows.map((row) => row.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${position}_${horizon}w_rankings.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Backtest-mode enriched rows
  const btEnriched = useMemo(() => {
    if (dataMode !== 'backtest') return []
    let items = btRows
      .filter((r) => r.predicted != null && r.actual != null)
      .map((r) => {
        const predicted = Number(r.predicted)
        const actual = Number(r.actual)
        const err = predicted - actual
        return { r, predicted, actual, error: err, absError: Math.abs(err), week: r.week ?? 0 }
      })
      .sort((a, b) => b.predicted - a.predicted)

    if (searchName.trim()) {
      const q = searchName.trim().toLowerCase()
      items = items.filter((x) => String(x.r.name ?? '').toLowerCase().includes(q))
    }
    return items.map((x, i) => ({ ...x, rank: i + 1, tier: getTier(i + 1, items.length) }))
  }, [btRows, dataMode, searchName])

  const horizonLabel = HORIZONS.find((h) => h.value === horizon)?.label ?? `${horizon} weeks`
  const isBacktest = dataMode === 'backtest'
  const activeLoading = isBacktest ? btLoading : loading

  return (
    <div className="rankings">
      {/* Controls */}
      <div className="rankings__controls section-card">
        <div className="rankings__control-row">
          {/* Data Mode toggle */}
          <div className="rankings__control-group">
            <label className="rankings__label">Mode</label>
            <div className="rankings__btn-group">
              <button
                className={`rankings__view-btn ${dataMode === 'live' ? 'rankings__view-btn--active' : ''}`}
                onClick={() => setDataMode('live')}
              >
                Live
              </button>
              <button
                className={`rankings__view-btn ${dataMode === 'backtest' ? 'rankings__view-btn--active' : ''}`}
                onClick={() => setDataMode('backtest')}
                disabled={backtestSeasons.length === 0}
                title={backtestSeasons.length === 0 ? 'No backtest data available' : 'View historical backtest'}
              >
                Backtest
              </button>
            </div>
          </div>
          {/* Season selector (backtest mode) */}
          {isBacktest && backtestSeasons.length > 0 && (
            <div className="rankings__control-group">
              <label className="rankings__label">Season</label>
              <select
                value={selectedBtSeason ?? ''}
                onChange={(e) => setSelectedBtSeason(Number(e.target.value))}
                style={{
                  background: 'var(--color-card)', color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-card-border)', borderRadius: 6,
                  padding: '0.35rem 0.6rem', fontSize: 'var(--text-small)',
                }}
              >
                {backtestSeasons.map((s) => (
                  <option key={s} value={s}>{s}/{s + 1}</option>
                ))}
              </select>
            </div>
          )}
          <div className="rankings__control-group">
            <label className="rankings__label">Position</label>
            <div className="rankings__btn-group">
              {POSITIONS.map((p) => (
                <button
                  key={p}
                  className={`rankings__pos-btn ${position === p ? 'rankings__pos-btn--active' : ''}`}
                  style={position === p ? { borderColor: POSITION_COLORS[p], color: POSITION_COLORS[p] } : {}}
                  onClick={() => onPositionChange(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
          {!isBacktest && (
            <div className="rankings__control-group">
              <label className="rankings__label">Horizon</label>
              <div className="rankings__btn-group">
                {HORIZONS.map((h) => (
                  <button
                    key={h.value}
                    className={`rankings__horizon-btn ${horizon === h.value ? 'rankings__horizon-btn--active' : ''}`}
                    onClick={() => onHorizonChange(h.value)}
                  >
                    {h.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="rankings__control-group">
            <label className="rankings__label">View</label>
            <div className="rankings__btn-group">
              <button className={`rankings__view-btn ${viewMode === 'table' ? 'rankings__view-btn--active' : ''}`} onClick={() => setViewMode('table')}>
                Table
              </button>
              <button className={`rankings__view-btn ${viewMode === 'chart' ? 'rankings__view-btn--active' : ''}`} onClick={() => setViewMode('chart')}>
                Chart
              </button>
            </div>
          </div>
        </div>
        <div className="rankings__search-row">
          <input
            type="text"
            placeholder="Search player name..."
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            className="rankings__search"
          />
          <button type="button" onClick={downloadCsv} className="rankings__export-btn">
            Export CSV
          </button>
        </div>
      </div>

      {/* Header info */}
      <div className="rankings__info">
        <span className="rankings__count">{isBacktest ? btEnriched.length : enriched.length} players</span>
        <span className="rankings__metric">
          {isBacktest
            ? `Historical Backtest · ${selectedBtSeason}/${(selectedBtSeason ?? 0) + 1} · Predicted vs Actual`
            : `${isFPPosition ? 'Ranked by Fantasy Points' : 'Ranked by Utilization Score'} · ${horizonLabel}`}
        </span>
        {!isBacktest && data?.schedule_available && <span className="pill pill--success" style={{ fontSize: 12 }}>Schedule ✓</span>}
        {isBacktest && <span className="pill" style={{ fontSize: 12, background: 'rgba(167,139,250,0.15)', color: '#a78bfa', border: '1px solid rgba(167,139,250,0.3)' }}>Backtest</span>}
      </div>

      {/* Start/Sit Quick Summary — shown on 1-week horizon for lineup decisions */}
      {horizon === 1 && !loading && enriched.length > 0 && (
        <div className="rankings__startsit section-card">
          <h3 className="section-heading" style={{ fontSize: '1rem' }}>
            Quick Start/Sit — {position} · {horizonLabel}
          </h3>
          <div className="rankings__startsit-grid">
            <div className="rankings__startsit-col rankings__startsit-col--start">
              <div className="rankings__startsit-header" style={{ color: '#10b981' }}>Must Starts</div>
              {enriched.filter((x) => x.tier.num <= 2).slice(0, 5).map((x, i) => (
                <div key={i} className="rankings__startsit-row">
                  <span className="rankings__startsit-rank">#{x.rank}</span>
                  <span className="rankings__startsit-name">{x.r.name}</span>
                  <span className="rankings__startsit-matchup">
                    {x.r.upcoming_home_away === 'home' ? '🏠' : x.r.upcoming_home_away === 'away' ? '✈️' : ''}{' '}
                    {x.matchup}
                  </span>
                  <span className="rankings__startsit-pts" style={{ color: '#10b981' }}>{x.chartVal!.toFixed(1)}</span>
                </div>
              ))}
            </div>
            <div className="rankings__startsit-col rankings__startsit-col--sit">
              <div className="rankings__startsit-header" style={{ color: '#fb923c' }}>Consider Sitting</div>
              {enriched.filter((x) => x.tier.num >= 4).slice(0, 5).map((x, i) => (
                <div key={i} className="rankings__startsit-row">
                  <span className="rankings__startsit-rank">#{x.rank}</span>
                  <span className="rankings__startsit-name">{x.r.name}</span>
                  <span className="rankings__startsit-matchup">
                    {x.r.upcoming_home_away === 'home' ? '🏠' : x.r.upcoming_home_away === 'away' ? '✈️' : ''}{' '}
                    {x.matchup}
                  </span>
                  <span className="rankings__startsit-pts" style={{ color: '#94a3b8' }}>{x.chartVal!.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {error && !isBacktest && (
        <div className="section-card" style={{ borderColor: 'var(--color-accent-amber)', textAlign: 'center' }}>
          <p style={{ color: 'var(--color-accent-amber)' }}>{error}</p>
        </div>
      )}

      {activeLoading ? (
        <div className="section-card">
          <div className="skeleton" style={{ height: 400 }} />
        </div>
      ) : isBacktest ? (
        /* Backtest Table View */
        btEnriched.length === 0 ? (
          <div className="section-card" style={{ textAlign: 'center', padding: '3rem' }}>
            <p style={{ color: 'var(--color-text-muted)' }}>
              No backtest predictions available for this season. Run:
            </p>
            <pre style={{ background: 'var(--color-bg)', padding: '0.75rem', borderRadius: 8, fontSize: 'var(--text-small)', display: 'inline-block', marginTop: '0.5rem' }}>
              python scripts/run_ts_backtest.py --season {selectedBtSeason ?? 2024}
            </pre>
          </div>
        ) : viewMode === 'chart' ? (
          <div className="section-card">
            <h2 className="section-heading">{position} Backtest — {selectedBtSeason}</h2>
            <div style={{ height: Math.max(400, Math.min(btEnriched.length, 30) * 28), maxHeight: 700, overflowY: 'auto' }}>
              <ResponsiveContainer width="100%" height={Math.max(400, Math.min(btEnriched.length, 30) * 28)}>
                <BarChart
                  data={btEnriched.slice(0, 30).map((x) => ({
                    name: `${x.r.name ?? 'Unknown'} (W${x.week})`,
                    predicted: x.predicted,
                    actual: x.actual,
                  }))}
                  layout="vertical"
                  margin={{ top: 8, right: 50, left: 200, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal vertical={false} />
                  <XAxis type="number" stroke="#94a3b8" />
                  <YAxis type="category" dataKey="name" stroke="#94a3b8" width={190} interval={0} tick={{ fontSize: 11, fill: '#cbd5e1' }} />
                  <Tooltip
                    contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }}
                    formatter={(value: number, name: string) => [value.toFixed(1), name === 'predicted' ? 'Predicted' : 'Actual']}
                  />
                  <Bar dataKey="predicted" fill="var(--color-accent-cyan)" fillOpacity={0.7} radius={[0, 3, 3, 0]} name="predicted" />
                  <Bar dataKey="actual" fill="var(--color-accent-emerald)" fillOpacity={0.7} radius={[0, 3, 3, 0]} name="actual" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="section-card rankings__table-card">
            <div className="rankings__table-wrap">
              <table className="rankings__table">
                <thead>
                  <tr>
                    <th className="rankings__th rankings__th--rank">Rank</th>
                    <th className="rankings__th rankings__th--tier">Tier</th>
                    <th className="rankings__th rankings__th--name">Player</th>
                    <th className="rankings__th rankings__th--team">Team</th>
                    <th className="rankings__th" style={{ textAlign: 'right' }}>Week</th>
                    <th className="rankings__th rankings__th--pts">Predicted</th>
                    <th className="rankings__th" style={{ textAlign: 'right' }}>Actual</th>
                    <th className="rankings__th" style={{ textAlign: 'right' }}>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {btEnriched.slice(0, 200).map((x, i) => (
                    <tr key={i} className="rankings__tr">
                      <td className="rankings__td rankings__td--rank">
                        <span className="rankings__rank-num">{x.rank}</span>
                      </td>
                      <td className="rankings__td rankings__td--tier">
                        <span className="tier-badge" style={{ color: x.tier.color, background: x.tier.bg }}>
                          {x.tier.label}
                        </span>
                      </td>
                      <td className="rankings__td rankings__td--name">
                        <span className="rankings__player-name">{x.r.name ?? 'Unknown'}</span>
                        <span className="rankings__player-pos">{x.r.position}</span>
                      </td>
                      <td className="rankings__td rankings__td--team">{x.r.team ?? '—'}</td>
                      <td className="rankings__td" style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{x.week}</td>
                      <td className="rankings__td rankings__td--pts">
                        <span className="rankings__pts-value">{x.predicted.toFixed(1)}</span>
                      </td>
                      <td className="rankings__td" style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-emerald)' }}>
                        {x.actual.toFixed(1)}
                      </td>
                      <td className="rankings__td" style={{
                        textAlign: 'right', fontFamily: 'var(--font-mono)',
                        color: x.absError < 3 ? 'var(--color-accent-emerald)' : x.absError < 7 ? 'var(--color-accent-amber)' : '#f87171',
                      }}>
                        {x.error > 0 ? '+' : ''}{x.error.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      ) : enriched.length === 0 ? (
        <div className="section-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p style={{ color: 'var(--color-text-muted)' }}>
            No prediction data. Run <code>python scripts/generate_app_data.py</code> to generate predictions.
          </p>
        </div>
      ) : viewMode === 'chart' ? (
        /* Chart View */
        <div className="section-card">
          <h2 className="section-heading">{position} Rankings — {horizonLabel}</h2>
          <div style={{ height: Math.max(400, chartData.length * 28), maxHeight: 700, overflowY: 'auto' }}>
            <ResponsiveContainer width="100%" height={Math.max(400, chartData.length * 28)}>
              <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 50, left: 200, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal vertical={false} />
                <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                <YAxis type="category" dataKey="name" stroke="#94a3b8" width={190} interval={0} tick={{ fontSize: 11, fill: '#cbd5e1' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }}
                  formatter={(value: number) => [value.toFixed(1), isFPPosition ? 'Fantasy Pts' : 'Util Score']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  <LabelList dataKey="value" position="right" formatter={(v: number) => v.toFixed(1)} fill="#cbd5e1" />
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.tier.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        /* Table View */
        <div className="section-card rankings__table-card">
          <div className="rankings__table-wrap">
            <table className="rankings__table">
              <thead>
                <tr>
                  <th className="rankings__th rankings__th--rank">Rank</th>
                  <th className="rankings__th rankings__th--tier">Tier</th>
                  <th className="rankings__th rankings__th--name">Player</th>
                  <th className="rankings__th rankings__th--team">Team</th>
                  <th className="rankings__th rankings__th--matchup">Matchup</th>
                  <th className="rankings__th rankings__th--pts">{isFPPosition ? 'Proj Pts' : 'Util Score'}</th>
                  {enriched.some((x) => x.floor != null) && (
                    <th className="rankings__th rankings__th--range">Floor / Ceiling</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {enriched.slice(0, 150).map((x) => (
                  <tr key={x.rank} className="rankings__tr">
                    <td className="rankings__td rankings__td--rank">
                      <span className="rankings__rank-num">{x.rank}</span>
                    </td>
                    <td className="rankings__td rankings__td--tier">
                      <span className="tier-badge" style={{ color: x.tier.color, background: x.tier.bg }}>
                        {x.tier.label}
                      </span>
                    </td>
                    <td className="rankings__td rankings__td--name">
                      <span className="rankings__player-name">{x.r.name ?? 'Unknown'}</span>
                      <span className="rankings__player-pos">{x.r.position}</span>
                    </td>
                    <td className="rankings__td rankings__td--team">{x.r.team ?? '—'}</td>
                    <td className="rankings__td rankings__td--matchup">{x.matchup}</td>
                    <td className="rankings__td rankings__td--pts">
                      <span className="rankings__pts-value">{x.chartVal!.toFixed(1)}</span>
                    </td>
                    {enriched.some((y) => y.floor != null) && (
                      <td className="rankings__td rankings__td--range">
                        {x.floor != null && x.ceiling != null ? (
                          <span className="rankings__range">
                            <span className="rankings__floor">{x.floor.toFixed(1)}</span>
                            <span className="rankings__range-sep">–</span>
                            <span className="rankings__ceiling">{x.ceiling.toFixed(1)}</span>
                          </span>
                        ) : '—'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tier Legend */}
      <div className="rankings__tier-legend">
        <span className="rankings__tier-legend-title">Tier Guide:</span>
        {[
          { label: 'Must Start', color: '#10b981' },
          { label: 'Start', color: '#00f5ff' },
          { label: 'Flex', color: '#fbbf24' },
          { label: 'Bench', color: '#fb923c' },
          { label: 'Sit', color: '#94a3b8' },
        ].map((t) => (
          <span key={t.label} className="rankings__tier-legend-item" style={{ color: t.color }}>
            ● {t.label}
          </span>
        ))}
      </div>
    </div>
  )
}
