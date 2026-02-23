import { useEffect, useState, useMemo } from 'react'
import { api } from '../api'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

const NUMERIC_BLACKLIST = ['id', 'target_1w', 'target_util_1w', 'target_4w', 'target_18w'].flatMap((s) => [s, ...['target_']].map((p) => p + s))

function getNumericColumns(sample: Record<string, unknown>[]): string[] {
  if (!sample.length) return []
  const first = sample[0]
  const out: string[] = []
  for (const k of Object.keys(first || {})) {
    if (NUMERIC_BLACKLIST.some((b) => k.startsWith('target_') || k === b)) continue
    const v = (first as Record<string, unknown>)[k]
    if (typeof v === 'number' && !Number.isNaN(v)) out.push(k)
  }
  return out.slice(0, 20)
}

function computeCorrelationMatrix(sample: Record<string, unknown>[], cols: string[]): number[][] {
  const n = cols.length
  const M: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        M[i][j] = 1
        continue
      }
      const xi = sample.map((r) => Number((r[cols[i]] as number) ?? NaN)).filter((v) => !Number.isNaN(v))
      const xj = sample.map((r) => Number((r[cols[j]] as number) ?? NaN)).filter((v) => !Number.isNaN(v))
      const len = Math.min(xi.length, xj.length)
      if (len < 2) continue
      const meanI = xi.slice(0, len).reduce((a, b) => a + b, 0) / len
      const meanJ = xj.slice(0, len).reduce((a, b) => a + b, 0) / len
      let num = 0,
        dI = 0,
        dJ = 0
      for (let k = 0; k < len; k++) {
        const a = xi[k] - meanI
        const b = xj[k] - meanJ
        num += a * b
        dI += a * a
        dJ += b * b
      }
      const den = Math.sqrt(dI * dJ) || 1
      M[i][j] = M[j][i] = num / den
    }
  }
  return M
}

export function EDA() {
  const [data, setData] = useState<{ sample: Record<string, unknown>[]; stats: { row_count: number; seasons: number[]; n_features: number } } | null>(null)
  const [loading, setLoading] = useState(true)
  const [positionFilter, setPositionFilter] = useState<string>('')
  useEffect(() => {
    api.eda(4000).then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const sample = useMemo(() => {
    if (!data?.sample.length) return []
    if (!positionFilter || !('position' in data.sample[0])) return data.sample
    return data.sample.filter((r) => String((r as Record<string, unknown>).position || '').toUpperCase() === positionFilter.toUpperCase())
  }, [data?.sample, positionFilter])

  const numericCols = useMemo(() => getNumericColumns(sample), [sample])
  const corrMatrix = useMemo(() => computeCorrelationMatrix(sample, numericCols), [sample, numericCols])
  const positions = useMemo(() => {
    if (!data?.sample.length || !('position' in data.sample[0])) return []
    const allowed = new Set(['QB', 'RB', 'WR', 'TE'])
    const set = new Set<string>()
    data.sample.forEach((r) => {
      const pos = String((r as Record<string, unknown>).position || '')
      if (allowed.has(pos)) set.add(pos)
    })
    return Array.from(set).filter(Boolean).sort()
  }, [data?.sample])

  const fantasyData = useMemo(() => {
    if (!sample.length || !('fantasy_points' in sample[0])) return []
    const arr = sample
      .map((r) => Number((r as Record<string, unknown>).fantasy_points as number))
      .filter((v) => !Number.isNaN(v))
    if (arr.length < 10) return []
    const sorted = [...arr].sort((a, b) => a - b)
    const p25 = sorted[Math.floor(sorted.length * 0.25)]
    const p50 = sorted[Math.floor(sorted.length * 0.5)]
    const p75 = sorted[Math.floor(sorted.length * 0.75)]
    const p95 = sorted[Math.floor(sorted.length * 0.95)]
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length
    const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length
    const std = Math.sqrt(variance)
    const hist: Record<number, number> = {}
    const step = (Math.max(...arr) - Math.min(...arr)) / 30 || 1
    arr.forEach((v) => {
      const bucket = Math.floor(v / step) * step
      hist[bucket] = (hist[bucket] || 0) + 1
    })
    return {
      buckets: Object.entries(hist).map(([k, v]) => ({ bin: Number(k), count: v })),
      stats: { mean, median: p50, std, p25, p50, p75, p95 },
    }
  }, [sample])

  if (loading) {
    return (
      <div className="section-card">
        <h2>Exploratory analysis</h2>
        <div className="skeleton" style={{ height: 320 }} />
      </div>
    )
  }

  if (!data?.sample.length) return null

  return (
    <div className="section-card">
      <h2>Exploratory analysis</h2>
      <p>Distributions and correlations across key variables (sample).</p>
      {positions.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="eda-position" style={{ marginRight: '0.5rem', color: 'var(--color-text-muted)' }}>
            Filter by position:
          </label>
          <select
            id="eda-position"
            value={positionFilter}
            onChange={(e) => setPositionFilter(e.target.value)}
            style={{
              background: 'var(--color-card)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-card-border)',
              borderRadius: 6,
              padding: '0.35rem 0.75rem',
            }}
          >
            <option value="">All</option>
            {positions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
      )}
      {numericCols.length >= 2 && corrMatrix.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Correlation heatmap (top variables)
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table
              role="grid"
              aria-label="Correlation matrix"
              style={{
                borderCollapse: 'collapse',
                fontSize: 'var(--text-small)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              <thead>
                <tr>
                  <th style={{ padding: 4, textAlign: 'left', color: 'var(--color-text-muted)' }} />
                  {numericCols.slice(0, 12).map((c) => (
                    <th key={c} style={{ padding: 4, maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', color: 'var(--color-text-muted)' }} title={c}>
                      {c.slice(0, 8)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {numericCols.slice(0, 12).map((row, i) => (
                  <tr key={row}>
                    <td style={{ padding: 4, color: 'var(--color-text-muted)' }} title={row}>
                      {row.slice(0, 8)}
                    </td>
                    {numericCols.slice(0, 12).map((col, j) => {
                      const v = corrMatrix[i]?.[j] ?? 0
                      const intensity = Math.abs(v)
                      const r = Math.round(10 + intensity * 245)
                      const g = Math.round(245 - intensity * 150)
                      const b = 255
                      return (
                        <td
                          key={col}
                          style={{
                            padding: 4,
                            background: `rgb(${r},${g},${b})`,
                            color: intensity > 0.5 ? '#fff' : 'var(--color-bg)',
                            textAlign: 'center',
                            borderRadius: 2,
                          }}
                          title={`${row} vs ${col}: ${v.toFixed(3)}`}
                        >
                          {v.toFixed(2)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {fantasyData && typeof fantasyData === 'object' && 'buckets' in fantasyData && Array.isArray((fantasyData as { buckets: unknown[] }).buckets) && (fantasyData as { buckets: unknown[] }).buckets.length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
            Fantasy points distribution
          </h3>
          <p style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
            Mean: {(fantasyData as { stats: { mean: number; median: number; std: number; p25: number; p50: number; p75: number; p95: number } }).stats.mean.toFixed(1)} · Median: {(fantasyData as { stats: { median: number } }).stats.median.toFixed(1)} · Std: {(fantasyData as { stats: { std: number } }).stats.std.toFixed(1)} · 25th: {(fantasyData as { stats: { p25: number } }).stats.p25.toFixed(0)} · 50th: {(fantasyData as { stats: { p50: number } }).stats.p50.toFixed(0)} · 75th: {(fantasyData as { stats: { p75: number } }).stats.p75.toFixed(0)} · 95th: {(fantasyData as { stats: { p95: number } }).stats.p95.toFixed(0)}
          </p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={(fantasyData as { buckets: { bin: number; count: number }[] }).buckets} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                <XAxis dataKey="bin" stroke="var(--color-text-muted)" fontSize={12} />
                <YAxis stroke="var(--color-text-muted)" fontSize={12} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                  labelStyle={{ color: 'var(--color-text-primary)' }}
                />
                <Bar dataKey="count" fill="var(--color-accent-cyan)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
