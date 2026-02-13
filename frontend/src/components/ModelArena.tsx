import { useEffect, useState } from 'react'
import { api, type AdvancedResults, type PositionResult } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export function ModelArena() {
  const [data, setData] = useState<AdvancedResults | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.advancedResults().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const byPosition = (data?.backtest_results ?? (data as unknown as Record<string, unknown>)?.['by_position']) as Record<string, PositionResult> | undefined
  if (!byPosition && !loading) return null

  const positions = Object.keys(byPosition || {}).filter((p) => byPosition![p] && typeof byPosition![p] === 'object')
  const firstPos = positions[0]
  const first = firstPos ? byPosition![firstPos] : null
  const comparison = first?.comparison || []
  const bestModel = first?.best_model

  return (
    <div className="section-card">
      <h2>Model arena</h2>
      <p>
        Per-position multi-week ensemble: XGBoost, LightGBM, and Ridge are trained on utilization (or FP for QB when chosen). Stacking meta-learner combines them. Best model per position below (test R²).
      </p>
      <p>
        <strong>Time horizon:</strong> All metrics below are for the <strong>1-week ahead</strong> horizon.
      </p>
      {loading ? (
        <div className="skeleton" style={{ height: 320 }} />
      ) : (
        <>
          {positions.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Test R² by position (1-week ahead)
              </h3>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={positions.map((p) => ({
                      position: p,
                      r2: Math.round(((byPosition as Record<string, PositionResult>)[p]?.r2 ?? 0) * 1000) / 1000,
                    }))}
                    margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                  >
                    <XAxis dataKey="position" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number) => [(value * 100).toFixed(2) + '%', 'R²']}
                    />
                    <Bar dataKey="r2" fill="var(--color-accent-cyan)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          {comparison.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Model comparison ({firstPos}) (1-week ahead)
              </h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
                {comparison.map((c: { model?: string; r2?: number; time_seconds?: number }) => (
                  <div
                    key={c.model}
                    style={{
                      padding: '1rem',
                      background: c.model === bestModel ? 'rgba(0,245,255,0.1)' : 'var(--color-card)',
                      border: `1px solid ${c.model === bestModel ? 'var(--color-accent-cyan)' : 'var(--color-card-border)'}`,
                      borderRadius: 12,
                      minWidth: 140,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>{c.model}</span>
                      {c.model === bestModel && (
                        <span style={{ fontSize: 'var(--text-small)', background: 'var(--color-accent-amber)', color: 'var(--color-bg)', padding: '0.15rem 0.4rem', borderRadius: 4 }}>
                          Production
                        </span>
                      )}
                    </div>
                    <div className="data-label" style={{ marginTop: '0.5rem' }}>R² = {((c.r2 ?? 0) * 100).toFixed(2)}%</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
