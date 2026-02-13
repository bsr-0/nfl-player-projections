import { useMemo, useState, useRef, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList, CartesianGrid } from 'recharts'
import type { PredictionRow } from '../api'

const ROW_HEIGHT = 28
const CHART_MAX_HEIGHT = 600
const COLORS = [
  'var(--color-accent-cyan)',
  'var(--color-accent-purple)',
  'var(--color-accent-emerald)',
  'var(--color-accent-amber)',
  'var(--color-accent-rose)',
  'var(--color-accent-sky)',
  'var(--color-accent-lime)',
  'var(--color-accent-orange)',
]

const POSITION_COLORS: Record<string, string> = {
  QB: 'var(--color-accent-cyan)',
  RB: 'var(--color-accent-purple)',
  WR: 'var(--color-accent-emerald)',
  TE: 'var(--color-accent-amber)',
}

function formatMatchup(row: PredictionRow): string {
  const opp = row.upcoming_opponent ?? ''
  const ha = row.upcoming_home_away ?? ''
  if (!opp || opp === '') return '—'
  if (ha === 'home') return `vs ${opp}`
  if (ha === 'away') return `@ ${opp}`
  return ha === 'unknown' ? opp : `${ha} ${opp}`
}

export type HorizonChartOption = 1 | 4 | 18

interface PredictionsChartProps {
  rows: PredictionRow[]
  weekLabel: string
  horizonLabel: string
  horizonWeeksLabel?: string
  scheduleNote?: string
  scheduleAvailable?: boolean
  scheduleByHorizon?: Record<string, boolean>
  horizon?: HorizonChartOption
  positionFilter?: string
  loading?: boolean
  error?: string | null
}

export function PredictionsChart({
  rows,
  weekLabel,
  horizonLabel,
  horizonWeeksLabel,
  scheduleNote,
  scheduleAvailable,
  scheduleByHorizon: _scheduleByHorizon,
  horizon: _horizon = 1,
  positionFilter,
  loading,
  error,
}: PredictionsChartProps) {
  const isQB = positionFilter === 'QB'
  const [showScheduleDetails, setShowScheduleDetails] = useState(false)
  const chartRef = useRef<HTMLDivElement>(null)
  const scrollToChart = useCallback(() => {
    chartRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const chartData = useMemo(() => {
    const projCol = (r: PredictionRow) =>
      r.projected_points ?? r.predicted_points ?? r.projection_1w ?? r.projection_4w ?? r.projection_18w
    const utilCol = (r: PredictionRow) =>
      r.utilization_score != null && !Number.isNaN(Number(r.utilization_score))
        ? Number(r.utilization_score)
        : (projCol(r) != null && !Number.isNaN(Number(projCol(r))) ? Number(projCol(r)) : null)
    const withValues = rows.map((r) => {
      const chartVal = isQB
        ? (projCol(r) != null && !Number.isNaN(Number(projCol(r))) ? Number(projCol(r)) : null)
        : utilCol(r)
      return { originalRow: r, chartValue: chartVal }
    })
    const filtered = withValues.filter((r) => r.chartValue != null)
    const sorted = [...filtered].sort((a, b) => Number(b.chartValue) - Number(a.chartValue))
    return sorted.map(({ originalRow: r, chartValue }) => {
      const teamStr = (r.team ?? '').toString().trim()
      const nameLabel = `${r.name ?? 'Unknown'} — ${teamStr || '—'}`
      return {
        name: nameLabel,
        value: Number(chartValue),
        fullRow: r,
        matchup: formatMatchup(r),
      }
    })
  }, [rows, positionFilter, isQB])

  if (error) {
    return (
      <div className="predictions-chart predictions-chart--error">
        <p>{error}</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="predictions-chart section-card">
        <div className="predictions-chart__section">
          <div className="skeleton predictions-chart__skeleton-line" style={{ width: '60%', height: 14 }} />
          <div className="skeleton predictions-chart__skeleton-line" style={{ width: '40%', height: 14 }} />
          <div className="skeleton predictions-chart__skeleton-pill" style={{ width: 140, height: 28, borderRadius: 9999 }} />
        </div>
        <div className="predictions-chart__section">
          <div className="skeleton predictions-chart__skeleton-line" style={{ width: '80%', height: 14 }} />
          <div className="skeleton predictions-chart__skeleton-line" style={{ width: '55%', height: 14 }} />
        </div>
        <div className="skeleton predictions-chart__skeleton" style={{ height: 320, marginTop: 8 }} />
      </div>
    )
  }

  if (chartData.length === 0) {
    return (
      <div className="predictions-chart predictions-chart--empty">
        <p>No prediction data for this selection. Run <code>python scripts/generate_app_data.py</code> to generate predictions.</p>
      </div>
    )
  }

  const weekTitle = weekLabel
    ? (weekLabel.startsWith('Season ') ? weekLabel.replace(/^Season (\d+), Week (\d+)$/, '$1, Week $2 Predictions') : `${weekLabel} Predictions`)
    : 'Predictions'

  return (
    <div className="predictions-chart section-card">
      <details className="predictions-chart__info" open>
        <summary className="predictions-chart__info-summary">
          <span className="predictions-chart__info-summary-text">{weekTitle}</span>
          <span
            className={`pill ${scheduleAvailable ? 'pill--success' : 'pill--warning'}`}
            aria-label={scheduleAvailable ? 'Schedule used for predictions' : 'Schedule not used for predictions'}
          >
            {scheduleAvailable ? 'Schedule used' : 'Schedule not used'}
          </span>
        </summary>
        <div className="predictions-chart__section">
          <h2 className="predictions-chart__section-heading">Week & schedule</h2>
          {horizonWeeksLabel && (
            <p className="predictions-chart__week">{horizonWeeksLabel}</p>
          )}
          <div className="predictions-chart__schedule-row">
            {scheduleNote && (
              <button
                type="button"
                className="predictions-chart__details-btn"
                onClick={() => setShowScheduleDetails((v) => !v)}
                aria-expanded={showScheduleDetails}
                aria-label={showScheduleDetails ? 'Hide schedule details' : 'Show schedule details'}
              >
                {showScheduleDetails ? 'Hide details' : 'Details'}
              </button>
            )}
          </div>
          {showScheduleDetails && scheduleNote && (
            <p className="predictions-chart__week predictions-chart__week--warning">{scheduleNote}</p>
          )}
        </div>
        <div className="predictions-chart__section">
          <h2 className="predictions-chart__section-heading">Data source & metric</h2>
          <p className="predictions-chart__week">
            Team assignments: nfl-data-py (current and next season when available).
            <span className="predictions-chart__help" title="Team and roster data source; used for matchups and next-season display." aria-label="Help"> (?)</span>
          </p>
          <p className="predictions-chart__week predictions-chart__week--primary">
            {isQB
              ? 'Dependent variable: Fantasy points (QB)'
              : `Dependent variable: Utilization score (${positionFilter ?? 'RB/WR/TE'})`}
          </p>
        </div>
      </details>
      <div className="predictions-chart__section">
        <h2 className="predictions-chart__title">
          <button
            type="button"
            className="predictions-chart__title-link"
            onClick={scrollToChart}
            aria-label={`${horizonLabel}, ${chartData.length} players — jump to chart`}
          >
            {horizonLabel} — {chartData.length} players
          </button>
        </h2>
        {chartData.length > 0 && (
          <p className="predictions-chart__top-preview">
            Top: {chartData.slice(0, 5).map((d) => d.fullRow.name ?? 'Unknown').join(', ')}
          </p>
        )}
      </div>
      <div ref={chartRef} className="predictions-chart__chart" style={{ maxHeight: CHART_MAX_HEIGHT, overflowY: 'auto' }}>
        <ResponsiveContainer width="100%" height={Math.max(CHART_MAX_HEIGHT, chartData.length * ROW_HEIGHT)}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 8, right: 24, left: 260, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              stroke="var(--color-text-muted)"
              tick={{ fill: 'var(--color-text-muted)' }}
              domain={[0, 'auto']}
              label={{
                value: isQB ? 'Fantasy points' : 'Utilization score',
                position: 'insideBottom',
                offset: -4,
                fill: 'var(--color-text-muted)',
                fontSize: 12,
              }}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="var(--color-text-muted)"
              width={240}
              interval={0}
              tickCount={chartData.length}
              tick={{ fontSize: 11, fill: 'var(--color-text-secondary)' }}
            />
            <Tooltip
              contentStyle={{
                background: 'var(--color-card)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 8,
                color: 'var(--color-text-secondary)',
              }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const pl = payload[0].payload as {
                  fullRow: PredictionRow
                  value: number
                  matchup: string
                }
                const r = pl.fullRow
                const nextTeam = r.team_next_season ?? ''
                const uncertainty = r.prediction_std != null ? `Uncertainty: ±${Number(r.prediction_std).toFixed(1)} pts` : ''
                return (
                  <div className="predictions-chart__tooltip">
                    <div className="predictions-chart__tooltip-name">{r.name} ({r.position})</div>
                    {r.team && <div>{r.team}</div>}
                    {nextTeam && <div>Team (next season): {nextTeam}</div>}
                    {isQB ? (
                      <div><strong>Fantasy points: {pl.value.toFixed(1)}</strong></div>
                    ) : (
                      <>
                        <div><strong>Utilization score: {pl.value.toFixed(1)}</strong></div>
                        {r.projected_points != null && <div style={{ color: 'var(--color-text-muted)' }}>Projected points: {Number(r.projected_points).toFixed(1)}</div>}
                      </>
                    )}
                    {scheduleAvailable && pl.matchup && pl.matchup !== '—' && <div>Matchup: {pl.matchup}</div>}
                    {uncertainty && <div>{uncertainty}</div>}
                  </div>
                )
              }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} name={isQB ? 'Fantasy points' : 'Utilization score'}>
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => Number(v).toFixed(1)}
                fill="var(--color-text-secondary)"
              />
              {chartData.map((_, i) => {
                const fill = (positionFilter && POSITION_COLORS[positionFilter]) ?? COLORS[i % COLORS.length]
                return <Cell key={i} fill={fill} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
