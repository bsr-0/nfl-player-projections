import { useEffect, useState } from 'react'
import { api, type PredictionRow } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const PROJ_COLS = [
  { key: 'projection_1w', label: '1 week' },
  { key: 'projection_4w', label: '4 weeks' },
  { key: 'projection_18w', label: '18 weeks' },
  { key: 'predicted_points', label: '1 week' },
] as const

export function LivePredictions() {
  const [data, setData] = useState<{ rows: PredictionRow[]; week_label: string; schedule_note?: string } | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.predictions().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const rows = data?.rows ?? []
  const weekLabel = data?.week_label ?? ''
  const latestSeason = rows.length && 'season' in rows[0] ? Math.max(...rows.map((r) => Number(r.season) || 0)) : null
  const dfLive = latestSeason != null ? rows.filter((r) => r.season === latestSeason) : rows

  if (rows.length === 0 && !loading) return null

  return (
    <div className="section-card">
      <h2>Live performance and upcoming weeks</h2>
      <p>
        Current projections from the trained ensemble (1 week, 4 weeks, 18 weeks). Use for lineup decisions; run <code>scripts/generate_app_data.py</code> to refresh.
      </p>
      {weekLabel && <p><strong>Predictions for:</strong> {weekLabel}</p>}
      {data?.schedule_note && <p style={{ fontSize: 'var(--text-small)', color: 'var(--color-accent-amber)' }}>{data.schedule_note}</p>}
      {loading ? (
        <div className="skeleton" style={{ height: 400 }} />
      ) : (
        <>
          {PROJ_COLS.map(({ key, label }) => {
            if (!dfLive.some((r) => (r as Record<string, unknown>)[key] != null)) return null
            const sorted = [...dfLive]
              .map((r) => ({ ...r, val: Number((r as Record<string, unknown>)[key]) }))
              .filter((r) => !Number.isNaN(r.val))
              .sort((a, b) => b.val - a.val)
              .slice(0, 35)
            if (sorted.length < 3) return null
            const chartData = sorted.map((r) => ({
              name: `${r.name ?? ''} (${r.position ?? ''})`,
              value: r.val,
              position: r.position,
            }))
            const colors = ['var(--color-accent-cyan)', 'var(--color-accent-purple)', 'var(--color-accent-emerald)', 'var(--color-accent-amber)']
            return (
              <div key={key} style={{ marginTop: '1.5rem' }}>
                <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)' }}>{label} projections</h3>
                <div style={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 120, bottom: 5 }}>
                      <XAxis type="number" stroke="var(--color-text-muted)" />
                      <YAxis type="category" dataKey="name" stroke="var(--color-text-muted)" width={110} tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                        labelStyle={{ color: 'var(--color-text-primary)' }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {chartData.map((_, i) => (
                          <Cell key={i} fill={colors[i % colors.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}
