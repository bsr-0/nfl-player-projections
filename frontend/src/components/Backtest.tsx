import { useEffect, useState } from 'react'
import { api, type BacktestResponse } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

export function Backtest() {
  const [data, setData] = useState<BacktestResponse | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.backtest().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const latest = data?.latest
  const byPosition = latest?.by_position ?? {}
  const byWeek = latest?.by_week ?? {}
  const metrics = latest?.metrics ?? {}
  const baseline = latest?.baseline_comparison

  const weekData = Object.entries(byWeek)
    .map(([week, v]) => ({ week, r2: (v as { r2?: number }).r2 ?? 0 }))
    .sort((a, b) => Number(a.week) - Number(b.week))

  const posData = Object.entries(byPosition).map(([position, v]) => ({
    position,
    r2: (v as { r2?: number }).r2 ?? 0,
  }))

  const dirAcc = (metrics as { directional_accuracy_pct?: number }).directional_accuracy_pct ?? 0
  const within5 = (metrics as { within_5_pts_pct?: number }).within_5_pts_pct ?? 0

  if (!latest && !loading) return null

  return (
    <div className="section-card">
      <h2>Backtesting narrative</h2>
      <p>
        How the model would have performed on a held-out season: predictions vs actuals, by position and by week. Directional accuracy and within-X-pts rates build trust.
      </p>
      <p>
        <strong>Time horizon:</strong> All backtest metrics are <strong>1-week ahead</strong>.
      </p>
      {loading ? (
        <div className="skeleton" style={{ height: 300 }} />
      ) : (
        <>
          {posData.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Backtest R² by position (1-week ahead)
              </h3>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={posData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
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
          {weekData.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                R² by week (1-week backtest)
              </h3>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weekData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                    <XAxis dataKey="week" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number) => [(value * 100).toFixed(2) + '%', 'R²']}
                    />
                    <Line type="monotone" dataKey="r2" stroke="var(--color-accent-emerald)" strokeWidth={2} dot={{ fill: 'var(--color-accent-emerald)' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          {(dirAcc > 0 || within5 > 0) && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Backtest overall (1-week)
              </h3>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      { name: 'Directional accuracy', value: dirAcc },
                      { name: 'Within 5 pts', value: within5 },
                    ]}
                    margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                  >
                    <XAxis dataKey="name" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" domain={[0, 105]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
                    />
                    <Bar dataKey="value" fill="var(--color-accent-purple)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          {baseline && !('error' in baseline) && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Model vs baseline (why trust this)
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                <div>
                  <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>R² (model vs baseline)</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>
                    Model: {(baseline.model?.r2 ?? 0).toFixed(3)} · Baseline: {(baseline.baseline?.r2 ?? 0).toFixed(3)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>MAE</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>
                    Model: {(baseline.model?.mae ?? 0).toFixed(2)} · Baseline: {(baseline.baseline?.mae ?? 0).toFixed(2)}
                  </div>
                </div>
              </div>
              {baseline.improvement && (
                <p style={{ marginTop: '0.75rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                  Improvement: {baseline.improvement.rmse_pct ?? 0}% RMSE reduction, R² gain {baseline.improvement.r2_gain ?? 0}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
