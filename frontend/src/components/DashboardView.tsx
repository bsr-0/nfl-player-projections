import { useEffect, useState, useMemo } from 'react'
import { api, type PredictionRow } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function useWindowWidth() {
  const [width, setWidth] = useState(
    typeof window !== 'undefined' ? window.innerWidth : 1024
  )
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return width
}

const POSITION_COLORS: Record<string, string> = {
  QB: '#00f5ff',
  RB: '#a78bfa',
  WR: '#10b981',
  TE: '#fbbf24',
  K: '#f472b6',
  DST: '#fb923c',
}

const POSITION_LABELS: Record<string, string> = {
  QB: 'Quarterbacks',
  RB: 'Running Backs',
  WR: 'Wide Receivers',
  TE: 'Tight Ends',
  K: 'Kickers',
  DST: 'Defense/ST',
}

function getTier(rank: number, total: number): { label: string; color: string; bg: string } {
  const pct = rank / total
  if (pct <= 0.1) return { label: 'Must Start', color: '#10b981', bg: 'rgba(16,185,129,0.15)' }
  if (pct <= 0.3) return { label: 'Start', color: '#00f5ff', bg: 'rgba(0,245,255,0.1)' }
  if (pct <= 0.55) return { label: 'Flex', color: '#fbbf24', bg: 'rgba(251,191,36,0.1)' }
  return { label: 'Sit', color: '#94a3b8', bg: 'rgba(148,163,184,0.08)' }
}

function getProjectedPoints(r: PredictionRow): number | null {
  const v = r.projection_18w ?? r.projected_points ?? r.predicted_points ?? r.projection_1w
  return v != null && !Number.isNaN(Number(v)) ? Number(v) : null
}

interface DashboardViewProps {
  allData: Record<string, PredictionRow[]>
  weekLabel: string
  loading: boolean
  scheduleAvailable?: boolean
}

export function DashboardView({ allData, weekLabel, loading, scheduleAvailable }: DashboardViewProps) {
  const windowWidth = useWindowWidth()
  const isMobile = windowWidth < 768
  const [heroData, setHeroData] = useState<{ record_count: number; correlation?: number } | null>(null)
  const [pipelineData, setPipelineData] = useState<{ row_count: number; seasons: number[]; season_range: string; n_features: number; health: string } | null>(null)

  useEffect(() => {
    api.hero().then(setHeroData).catch(() => {})
    api.dataPipeline().then(setPipelineData).catch(() => {})
  }, [])

  const topPicks = useMemo(() => {
    const picks: { pos: string; players: { name: string; team: string; pts: number; tier: ReturnType<typeof getTier>; matchup: string }[] }[] = []
    for (const pos of ['QB', 'RB', 'WR', 'TE', 'K', 'DST']) {
      const rows = allData[pos] ?? []
      const sorted = rows
        .map((r) => ({ r, pts: getProjectedPoints(r) }))
        .filter((x) => x.pts != null)
        .sort((a, b) => b.pts! - a.pts!)
      const total = sorted.length
      const players = sorted.slice(0, 5).map((x, i) => {
        const opp = x.r.upcoming_opponent ?? ''
        const ha = x.r.upcoming_home_away ?? ''
        let matchup = scheduleAvailable === false ? 'TBD' : '—'
        if (opp) matchup = ha === 'home' ? `vs ${opp}` : ha === 'away' ? `@ ${opp}` : opp
        return {
          name: String(x.r.name ?? 'Unknown'),
          team: String(x.r.team ?? '—'),
          pts: x.pts!,
          tier: getTier(i + 1, total),
          matchup,
        }
      })
      picks.push({ pos, players })
    }
    return picks
  }, [allData, scheduleAvailable])

  const overallTop10 = useMemo(() => {
    const all: { r: PredictionRow; pts: number }[] = []
    for (const pos of ['QB', 'RB', 'WR', 'TE', 'K', 'DST']) {
      (allData[pos] ?? []).forEach((r) => {
        const pts = getProjectedPoints(r)
        if (pts != null) all.push({ r, pts })
      })
    }
    return all.sort((a, b) => b.pts - a.pts).slice(0, 10)
  }, [allData])

  const hasData = Object.keys(allData).length > 0

  if (loading && !hasData) {
    return (
      <div className="dashboard">
        <div className="dashboard__hero section-card">
          <div className="skeleton" style={{ height: 120 }} />
        </div>
        <div className="dashboard__grid">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="section-card"><div className="skeleton" style={{ height: 200 }} /></div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard">
      {/* Hero Stats Row */}
      <div className="dashboard__hero-row">
        <div className="stat-card stat-card--accent">
          <div className="stat-card__value">{heroData?.record_count?.toLocaleString() ?? '—'}</div>
          <div className="stat-card__label">Player-Game Records</div>
        </div>
        <div className="stat-card stat-card--accent">
          <div className="stat-card__value">{heroData?.correlation != null ? `${heroData.correlation}%` : '—'}</div>
          <div className="stat-card__label">Model Correlation</div>
        </div>
        <div className="stat-card">
          <div className="stat-card__value">{pipelineData?.n_features ?? '—'}</div>
          <div className="stat-card__label">ML Features</div>
        </div>
        <div className="stat-card">
          <div className="stat-card__value">{pipelineData?.season_range ?? '—'}</div>
          <div className="stat-card__label">Seasons Covered</div>
        </div>
      </div>

      {/* Schedule pending notice */}
      {scheduleAvailable === false && (
        <div className="section-card" style={{ padding: '0.75rem 1rem', borderColor: 'rgba(251,191,36,0.3)', background: 'rgba(251,191,36,0.05)' }}>
          <p style={{ color: '#fbbf24', fontSize: 'var(--text-small)', margin: 0 }}>
            The upcoming schedule has not been released yet. Projections are based on historical performance and trends. Matchup-specific adjustments will be applied once the schedule is announced.
          </p>
        </div>
      )}

      {/* Overall Top 10 Chart */}
      {overallTop10.length > 0 && (
        <div className="section-card dashboard__top10">
          <h2 className="section-heading">
            Top 10 Overall — Full Season Projections
            {weekLabel && <span className="section-heading__sub">{weekLabel}</span>}
          </h2>
          <div style={{ height: isMobile ? 380 : 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={overallTop10.map((x) => {
                  const fullName = String(x.r.name)
                  const pos = String(x.r.position)
                  let label: string
                  if (isMobile) {
                    // Abbreviate: "Justin Jefferson" -> "J. Jefferson"
                    const parts = fullName.split(' ')
                    label = parts.length > 1
                      ? `${parts[0][0]}. ${parts.slice(1).join(' ')}`
                      : fullName
                    label = `${label} (${pos})`
                    if (label.length > 18) label = label.slice(0, 17) + '\u2026'
                  } else {
                    label = `${fullName} (${pos})`
                  }
                  return { name: label, value: x.pts, position: pos }
                })}
                layout="vertical"
                margin={isMobile
                  ? { top: 8, right: 16, left: 8, bottom: 8 }
                  : { top: 8, right: 50, left: 160, bottom: 8 }
                }
              >
                <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: isMobile ? 10 : 12 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#94a3b8"
                  width={isMobile ? 110 : 150}
                  tick={{ fontSize: isMobile ? 10 : 12, fill: '#cbd5e1' }}
                  interval={0}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }}
                  formatter={(value: number) => [value.toFixed(1), 'Season Projected Pts']}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {overallTop10.map((x, i) => (
                    <Cell key={i} fill={POSITION_COLORS[String(x.r.position)] ?? '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="legend-row">
            {['QB', 'RB', 'WR', 'TE', 'K', 'DST'].map((p) => (
              <span key={p} className="legend-item">
                <span className="legend-dot" style={{ background: POSITION_COLORS[p] }} />
                {p}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Position Quick Rankings */}
      <div className="dashboard__grid">
        {topPicks.map(({ pos, players }) => (
          <div key={pos} className="section-card position-card">
            <h3 className="position-card__title" style={{ color: POSITION_COLORS[pos] }}>
              {POSITION_LABELS[pos] ?? pos}
            </h3>
            {players.length === 0 ? (
              <p className="position-card__empty">No data available</p>
            ) : (
              <div className="position-card__list">
                {players.map((p, i) => (
                  <div key={i} className="position-card__row">
                    <span className="position-card__rank">#{i + 1}</span>
                    <div className="position-card__info">
                      <span className="position-card__name">{p.name}</span>
                      <span className="position-card__team">{p.team} · {p.matchup}</span>
                    </div>
                    <div className="position-card__right">
                      <span className="position-card__pts" title="Full season projected points">{p.pts.toFixed(1)}</span>
                      <span className="tier-badge" style={{ color: p.tier.color, background: p.tier.bg }}>{p.tier.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Trust Indicators */}
      <div className="section-card trust-section">
        <h2 className="section-heading">Why Trust These Projections?</h2>
        <div className="trust-grid">
          <div className="trust-item">
            <div className="trust-item__icon">🔒</div>
            <div className="trust-item__title">No Data Leakage</div>
            <div className="trust-item__desc">Strict time-series splits ensure no future information contaminates predictions</div>
          </div>
          <div className="trust-item">
            <div className="trust-item__icon">✅</div>
            <div className="trust-item__title">Cross-Validated</div>
            <div className="trust-item__desc">Multi-fold validation across seasons with ensemble of XGBoost, LightGBM, and Ridge</div>
          </div>
          <div className="trust-item">
            <div className="trust-item__icon">📅</div>
            <div className="trust-item__title">Matchup-Aware</div>
            <div className="trust-item__desc">Schedule data integrated — projections factor in opponent strength and home/away</div>
          </div>
          <div className="trust-item">
            <div className="trust-item__icon">📈</div>
            <div className="trust-item__title">Opportunity-Based</div>
            <div className="trust-item__desc">Utilization scoring (snaps, targets, red zone) is more predictive than raw fantasy points</div>
          </div>
        </div>
      </div>
    </div>
  )
}
