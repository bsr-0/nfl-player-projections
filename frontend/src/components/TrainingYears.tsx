import { useEffect, useState } from 'react'
import { api } from '../api'

interface Row {
  train_years?: number
  train_range?: string
  position: string
  test_correlation: number
  n_train?: number
}

export function TrainingYears() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.trainingYears().then(setData).catch(() => setData([])).finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="section-card">
        <h2>Training years analysis</h2>
        <div className="skeleton" style={{ height: 200 }} />
      </div>
    )
  }

  if (data.length === 0) return null

  const byPosition = data.reduce((acc, row) => {
    const pos = row.position || 'Other'
    if (!acc[pos]) acc[pos] = []
    acc[pos].push(row)
    return acc
  }, {} as Record<string, Row[]>)

  return (
    <div className="section-card">
      <h2>Training years analysis</h2>
      <p>Optimal training window per position. Correlation from held-out test; n_train is sample size.</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
        {Object.entries(byPosition).map(([position, rows]) => {
          const best = rows.reduce((a, b) => (a.test_correlation >= (b.test_correlation ?? 0) ? a : b), rows[0])
          const pct = Math.round((best.test_correlation ?? 0) * 1000) / 10
          const nTrain = best.n_train ?? 0
          return (
            <div
              key={position}
              style={{
                padding: '1rem',
                background: 'var(--color-card)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 12,
                textAlign: 'center',
              }}
            >
              <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                {position}
              </div>
              <div
                role="img"
                aria-label={`${pct}% correlation`}
                style={{
                  width: 80,
                  height: 80,
                  margin: '0 auto',
                  borderRadius: '50%',
                  background: `conic-gradient(var(--color-accent-cyan) 0% ${pct}%, var(--color-card-border) ${pct}% 100%)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <div style={{ width: 56, height: 56, borderRadius: '50%', background: 'var(--color-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>
                  {pct}%
                </div>
              </div>
              <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
                n = {nTrain.toLocaleString()}
              </div>
              {best.train_range && (
                <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)', marginTop: '0.25rem' }} title="Training window">
                  {best.train_range}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <p style={{ marginTop: '1rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
        Example: 4 years (2020–2023) yielded best RB correlation ~50% in this analysis.
      </p>
    </div>
  )
}
