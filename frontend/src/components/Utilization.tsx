import { useEffect, useState } from 'react'
import { api } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const POSITION_ORDER = ['QB', 'RB', 'WR', 'TE']

export function Utilization() {
  const [weights, setWeights] = useState<Record<string, Record<string, number>>>({})
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.utilizationWeights().then(setWeights).catch(() => setWeights({})).finally(() => setLoading(false))
  }, [])

  const positions = POSITION_ORDER.filter((p) => weights[p] && Object.keys(weights[p]).length > 0)
  if (positions.length === 0 && !loading) return null

  return (
    <div className="section-card">
      <h2>Variable engineering – utilization score</h2>
      <p>
        Opportunity is a better leading indicator than raw fantasy points. The utilization score combines snap share, target share, red zone usage, and other components into a 0–100 scale per position. Weights are data-driven where available.
      </p>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
        Formula (per position): score = Σ (weight_i × component_i) with components normalized.
      </p>
      <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}>
        <strong style={{ color: 'var(--color-accent-emerald)' }}>Why this matters:</strong> Predicting opportunity (snaps, targets, red zone role) is more stable than predicting raw fantasy points, which are noisy week-to-week.
      </div>
      <div style={{ marginTop: '1rem' }} aria-label="Flow: Raw stats to shares to weighted sum to 0-100 score">
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
          <span>Raw stats</span>
          <span>→</span>
          <span>Shares</span>
          <span>→</span>
          <span>Weighted sum</span>
          <span>→</span>
          <span style={{ color: 'var(--color-accent-cyan)' }}>0–100 score</span>
        </div>
      </div>
      {loading ? (
        <div className="skeleton" style={{ height: 260, marginTop: '1rem' }} />
      ) : (
        positions.map((pos) => {
          const w = weights[pos] || {}
          const entries = Object.entries(w).map(([name, value]) => ({ name, value: Math.round(value * 100) })).sort((a, b) => b.value - a.value)
          const colors = ['var(--color-accent-cyan)', 'var(--color-accent-purple)', 'var(--color-accent-emerald)', 'var(--color-accent-amber)']
          return (
            <div key={pos} style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)' }}>{pos} component weights</h3>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={entries} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                    <XAxis type="number" domain={[0, 100]} stroke="var(--color-text-muted)" tickFormatter={(v) => `${v}%`} />
                    <YAxis type="category" dataKey="name" stroke="var(--color-text-muted)" width={75} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number) => [`${value}%`, 'Weight']}
                      labelStyle={{ color: 'var(--color-text-primary)' }}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {entries.map((_, i) => (
                        <Cell key={i} fill={colors[i % colors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )
        })
      )}
    </div>
  )
}
