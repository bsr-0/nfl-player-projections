import { useEffect, useState, useMemo } from 'react'
import { api, type PredictionRow } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export function UpcomingTable() {
  const [data, setData] = useState<{ rows: PredictionRow[]; week_label: string; schedule_note?: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchName, setSearchName] = useState('')
  const [positionFilter, setPositionFilter] = useState<string[]>([])
  useEffect(() => {
    api.predictions().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const rows = data?.rows ?? []
  const weekLabel = data?.week_label ?? ''
  const proj1w = 'projection_1w' in (rows[0] || {}) ? 'projection_1w' : 'predicted_points'

  const positions = useMemo(() => {
    const allowed = new Set(['QB', 'RB', 'WR', 'TE'])
    const set = new Set<string>()
    rows.forEach((r) => {
      const pos = r.position ? String(r.position) : ''
      if (allowed.has(pos)) set.add(pos)
    })
    return Array.from(set).sort()
  }, [rows])

  const selectedPositions = positionFilter.length === 0 ? positions : positionFilter

  const filtered = useMemo(() => {
    let out = rows.filter((r) => r[proj1w as keyof PredictionRow] != null)
    if (selectedPositions.length > 0) {
      out = out.filter((r) => r.position && selectedPositions.includes(String(r.position)))
    }
    if (searchName.trim()) {
      const q = searchName.trim().toLowerCase()
      out = out.filter((r) => String(r.name ?? '').toLowerCase().includes(q))
    }
    return out.sort((a, b) => Number(b[proj1w as keyof PredictionRow]) - Number(a[proj1w as keyof PredictionRow]))
  }, [rows, selectedPositions, searchName, proj1w])

  const top25 = filtered.slice(0, 25)
  const chartData = top25.map((r) => ({
    name: `${r.name ?? ''} (${r.position ?? ''})`,
    value: Number(r[proj1w as keyof PredictionRow]),
    position: r.position,
  }))
  const colors = ['var(--color-accent-cyan)', 'var(--color-accent-purple)', 'var(--color-accent-emerald)']

  const downloadCsv = () => {
    const headers = ['name', 'position', 'team', '1-week projected pts', 'Uncertainty (±pts)'].filter((h) => h !== 'Uncertainty (±pts)' || rows.some((r) => r.prediction_std != null))
    const csvRows = filtered.slice(0, 200).map((r) => {
      const row = [
        r.name ?? '',
        r.position ?? '',
        r.team ?? '',
        String(r[proj1w as keyof PredictionRow] ?? ''),
      ]
      if (r.prediction_std != null) row.push(String(r.prediction_std))
      return row
    })
    const csv = [headers.join(','), ...csvRows.map((row) => row.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'upcoming_week_1w_predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (rows.length === 0 && !loading) return null

  return (
    <div className="section-card">
      <h2>Upcoming week: player-level predictions</h2>
      <p>
        Predicted fantasy points for each player for the next game week (1-week horizon). Use this table for lineup decisions and to validate the model once the week is played.
      </p>
      {weekLabel && <p><strong>Predictions for:</strong> {weekLabel}</p>}
      {data?.schedule_note && <p style={{ fontSize: 'var(--text-small)', color: 'var(--color-accent-amber)' }}>{data.schedule_note}</p>}
      {loading ? (
        <div className="skeleton" style={{ height: 420 }} />
      ) : (
        <>
          {chartData.length >= 3 && (
            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)' }}>Top 25 — 1-week projected points</h3>
              <div style={{ height: 420 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 120, bottom: 5 }}>
                    <XAxis type="number" stroke="var(--color-text-muted)" />
                    <YAxis type="category" dataKey="name" stroke="var(--color-text-muted)" width={110} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
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
          )}
          <div style={{ marginTop: '1.5rem' }}>
            <label htmlFor="upcoming-search" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--color-text-muted)' }}>
              Search by player name
            </label>
            <input
              id="upcoming-search"
              type="text"
              placeholder="Type name to filter..."
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
              style={{
                width: '100%',
                maxWidth: 320,
                padding: '0.5rem 0.75rem',
                background: 'var(--color-card)',
                border: '1px solid var(--color-card-border)',
                borderRadius: 8,
                color: 'var(--color-text-primary)',
                fontSize: 'var(--text-body)',
              }}
            />
          </div>
          {positions.length > 0 && (
            <div style={{ marginTop: '0.75rem', marginBottom: '1rem' }}>
              <span style={{ marginRight: '0.5rem', color: 'var(--color-text-muted)' }}>Filter by position:</span>
              {positions.map((p) => (
                <label key={p} style={{ marginRight: '1rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selectedPositions.includes(p)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setPositionFilter((prev) => {
                          const next = prev.includes(p) ? prev : [...prev, p]
                          return next.length === positions.length ? [] : next
                        })
                      } else {
                        setPositionFilter((prev) => {
                          const next = prev.length === 0 ? positions.filter((x) => x !== p) : prev.filter((x) => x !== p)
                          return next.length === 0 ? [] : next
                        })
                      }
                    }}
                  />{' '}
                  {p}
                </label>
              ))}
            </div>
          )}
          <p style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>All players — 1-week projected points (sortable; scroll for more)</p>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto', border: '1px solid var(--color-card-border)', borderRadius: 8 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-small)' }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--color-card)', zIndex: 1 }}>
                <tr>
                  <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left', color: 'var(--color-text-muted)' }}>Name</th>
                  <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left', color: 'var(--color-text-muted)' }}>Position</th>
                  <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left', color: 'var(--color-text-muted)' }}>Team</th>
                  {rows.some((r) => r.team_next_season != null && r.team_next_season !== '') && (
                    <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left', color: 'var(--color-text-muted)' }}>Team (next season)</th>
                  )}
                  <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right', color: 'var(--color-text-muted)' }}>1-week projected pts</th>
                  {rows.some((r) => r.prediction_std != null) && (
                    <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right', color: 'var(--color-text-muted)' }}>Uncertainty (±pts)</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 200).map((r, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--color-card-border)' }}>
                    <td style={{ padding: '0.5rem 0.75rem', color: 'var(--color-text-primary)' }}>{String(r.name ?? '')}</td>
                    <td style={{ padding: '0.5rem 0.75rem' }}>{String(r.position ?? '')}</td>
                    <td style={{ padding: '0.5rem 0.75rem' }}>{String(r.team ?? '')}</td>
                    {rows.some((x) => x.team_next_season != null && x.team_next_season !== '') && (
                      <td style={{ padding: '0.5rem 0.75rem' }}>{String(r.team_next_season ?? '—')}</td>
                    )}
                    <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                      {Number(r[proj1w as keyof PredictionRow]).toFixed(1)}
                    </td>
                    {rows.some((x) => x.prediction_std != null) && (
                      <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {r.prediction_std != null ? Number(r.prediction_std).toFixed(1) : '—'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button
            type="button"
            onClick={downloadCsv}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              background: 'var(--color-accent-cyan)',
              color: 'var(--color-bg)',
              border: 'none',
              borderRadius: 8,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Download as CSV
          </button>
        </>
      )}
    </div>
  )
}
