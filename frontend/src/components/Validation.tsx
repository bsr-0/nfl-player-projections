import { useEffect, useState } from 'react'
import { api, type AdvancedResults } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export function Validation() {
  const [data, setData] = useState<AdvancedResults | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.advancedResults().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const byPosition = data?.backtest_results ?? (data as unknown as Record<string, { r2?: number }>)?.['by_position']
  const trainSeasons = data?.train_seasons ?? []
  const testSeason = data?.test_season

  if (!byPosition && !trainSeasons.length && testSeason == null && !loading) return null

  const positions = Object.keys(byPosition || {})
  const r2Data = positions.map((p) => ({
    position: p,
    r2: Math.round(((byPosition as Record<string, { r2?: number }>)[p]?.r2 ?? 0) * 1000) / 1000,
  }))

  return (
    <div className="section-card">
      <h2>Validation gauntlet</h2>
      <p>
        Strict time-series split: training on past seasons, validation for tuning, and a single held-out test season for reporting. No future data is ever used.
      </p>
      <p>
        <strong>Time horizon:</strong> Test metrics below are <strong>1-week ahead</strong>.
      </p>
      {loading ? (
        <div className="skeleton" style={{ height: 200 }} />
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ padding: '1rem', background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Training</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>{trainSeasons.length ? trainSeasons.join(', ') : 'Pipeline'}</div>
            </div>
            <div style={{ padding: '1rem', background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Validation</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>Time-ordered split</div>
            </div>
            <div style={{ padding: '1rem', background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Test season</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>{testSeason ?? '—'}</div>
            </div>
          </div>
          {r2Data.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Test R² by position (1-week ahead)
              </h3>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={r2Data} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                    <XAxis dataKey="position" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number) => [(value * 100).toFixed(2) + '%', 'R²']}
                    />
                    <Bar dataKey="r2" fill="var(--color-accent-purple)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
