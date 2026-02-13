import { useEffect, useState, useMemo } from 'react'
import { api, type PredictionRow } from '../api'

export function AnalyzeByPlayer() {
  const [data, setData] = useState<{ rows: PredictionRow[]; week_label: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedName, setSelectedName] = useState('')
  useEffect(() => {
    api.predictions().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const rows = data?.rows ?? []
  const weekLabel = data?.week_label ?? ''
  const proj1w = rows[0] && 'projection_1w' in rows[0] ? 'projection_1w' : 'predicted_points'

  const names = useMemo(() => {
    const set = new Map<string, PredictionRow>()
    rows.forEach((r) => {
      const name = String(r.name ?? '').trim()
      if (name && !set.has(name)) set.set(name, r)
    })
    return Array.from(set.entries()).sort((a, b) => (a[0].toLowerCase() > b[0].toLowerCase() ? 1 : -1))
  }, [rows])

  const selected = useMemo(() => {
    if (!selectedName.trim()) return null
    const q = selectedName.trim().toLowerCase()
    return names.find(([n]) => n.toLowerCase() === q || n.toLowerCase().includes(q))?.[1] ?? null
  }, [names, selectedName])

  const rankInPosition = useMemo(() => {
    if (!selected?.position) return null
    const samePos = rows
      .filter((r) => r.position === selected.position && r[proj1w as keyof PredictionRow] != null)
      .sort((a, b) => Number(b[proj1w as keyof PredictionRow]) - Number(a[proj1w as keyof PredictionRow]))
    const idx = samePos.findIndex((r) => r.name === selected.name)
    if (idx < 0) return null
    return { rank: idx + 1, total: samePos.length }
  }, [rows, selected, proj1w])

  const positionAvg = useMemo(() => {
    if (!selected?.position) return null
    const samePos = rows.filter((r) => r.position === selected.position && r[proj1w as keyof PredictionRow] != null)
    if (samePos.length === 0) return null
    const sum = samePos.reduce((a, r) => a + Number(r[proj1w as keyof PredictionRow]), 0)
    return sum / samePos.length
  }, [rows, selected, proj1w])

  if (rows.length === 0 && !loading) return null

  return (
    <div className="section-card">
      <h2>Analyze by player</h2>
      <p>
        Select a player to see their 1-week, 4-week, and 18-week projections, rank within position, and uncertainty.
      </p>
      {weekLabel && <p><strong>Predictions for:</strong> {weekLabel}</p>}
      {loading ? (
        <div className="skeleton" style={{ height: 200 }} />
      ) : (
        <>
          <label htmlFor="player-select" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--color-text-muted)' }}>
            Select player
          </label>
          <input
            id="player-select"
            type="text"
            list="player-list"
            placeholder="Type or choose player name..."
            value={selectedName}
            onChange={(e) => setSelectedName(e.target.value)}
            style={{
              width: '100%',
              maxWidth: 360,
              padding: '0.5rem 0.75rem',
              background: 'var(--color-card)',
              border: '1px solid var(--color-card-border)',
              borderRadius: 8,
              color: 'var(--color-text-primary)',
              fontSize: 'var(--text-body)',
            }}
          />
          <datalist id="player-list">
            {names.slice(0, 500).map(([name]) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          {selected && (
            <div
              style={{
                marginTop: '1.5rem',
                padding: '1.5rem',
                background: 'var(--color-bg)',
                border: '1px solid var(--color-accent-cyan)',
                borderRadius: 12,
                boxShadow: 'var(--glow-cyan)',
              }}
            >
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginTop: 0 }}>
                {selected.name}
              </h3>
              <p style={{ color: 'var(--color-text-muted)' }}>
                {selected.position} · {selected.team ?? '—'}
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
                <div>
                  <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>1-week projected pts</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-cyan)', fontSize: '1.25rem' }}>
                    {Number(selected[proj1w as keyof PredictionRow]).toFixed(1)}
                  </div>
                </div>
                {(selected.projection_4w != null || selected.projection_18w != null) && (
                  <>
                    {selected.projection_4w != null && (
                      <div>
                        <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>4-week projected pts</div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-purple)', fontSize: '1.25rem' }}>
                          {Number(selected.projection_4w).toFixed(1)}
                        </div>
                      </div>
                    )}
                    {selected.projection_18w != null && (
                      <div>
                        <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>18-week projected pts</div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-accent-emerald)', fontSize: '1.25rem' }}>
                          {Number(selected.projection_18w).toFixed(1)}
                        </div>
                      </div>
                    )}
                  </>
                )}
                {selected.prediction_std != null && (
                  <div>
                    <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>Uncertainty (±pts)</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-text-data)', fontSize: '1.25rem' }}>
                      ±{Number(selected.prediction_std).toFixed(1)}
                    </div>
                  </div>
                )}
              </div>
              {rankInPosition && (
                <p style={{ marginTop: '1rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                  Rank in {selected.position}: {rankInPosition.rank} of {rankInPosition.total}
                </p>
              )}
              {positionAvg != null && (
                <p style={{ marginTop: '0.25rem', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                  {selected.position} average 1-week projection: {positionAvg.toFixed(1)} pts
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
