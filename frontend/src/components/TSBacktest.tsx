import { useEffect, useState, useMemo } from 'react'
import {
  api,
  type TSBacktestResult,
  type TSBacktestPredictionRow,
  type TSBacktestMetrics,
} from '../api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Line, ScatterChart, Scatter, CartesianGrid, Legend,
  ComposedChart,
} from 'recharts'

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE']

export function TSBacktest() {
  const [availableSeasons, setAvailableSeasons] = useState<number[]>([])
  const [selectedSeason, setSelectedSeason] = useState<number | null>(null)
  const [result, setResult] = useState<TSBacktestResult | null>(null)
  const [predictions, setPredictions] = useState<TSBacktestPredictionRow[]>([])
  const [position, setPosition] = useState('ALL')
  const [loading, setLoading] = useState(true)
  const [predLoading, setPredLoading] = useState(false)

  // Fetch available seasons on mount
  useEffect(() => {
    api.tsBacktest()
      .then((d) => {
        setAvailableSeasons(d.available_seasons ?? [])
        if (d.latest?.season) setSelectedSeason(d.latest.season)
        if (d.latest) setResult(d.latest)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  // Fetch season-specific result when selection changes
  useEffect(() => {
    if (selectedSeason == null) return
    setLoading(true)
    api.tsBacktestSeason(selectedSeason)
      .then(setResult)
      .catch(() => setResult(null))
      .finally(() => setLoading(false))
  }, [selectedSeason])

  // Fetch per-player predictions
  useEffect(() => {
    if (selectedSeason == null) return
    setPredLoading(true)
    api.tsBacktestPredictions(selectedSeason, position === 'ALL' ? undefined : position)
      .then((d) => setPredictions(d.rows ?? []))
      .catch(() => setPredictions([]))
      .finally(() => setPredLoading(false))
  }, [selectedSeason, position])

  // Weekly RMSE/MAE trend data
  const weeklyData = useMemo(() => {
    if (!result?.by_week) return []
    return Object.entries(result.by_week)
      .map(([week, m]) => ({
        week: Number(week),
        mae: (m as TSBacktestMetrics).mae ?? 0,
        rmse: (m as TSBacktestMetrics).rmse ?? 0,
        r2: (m as TSBacktestMetrics).r2 ?? 0,
        n: (m as TSBacktestMetrics).n ?? 0,
      }))
      .sort((a, b) => a.week - b.week)
  }, [result])

  // Position bar data
  const posData = useMemo(() => {
    if (!result?.by_position) return []
    return Object.entries(result.by_position).map(([pos, m]) => ({
      position: pos,
      mae: (m as TSBacktestMetrics).mae ?? 0,
      rmse: (m as TSBacktestMetrics).rmse ?? 0,
      r2: (m as TSBacktestMetrics).r2 ?? 0,
    }))
  }, [result])

  // Scatter data: predicted vs actual (sampled for performance)
  const scatterData = useMemo(() => {
    if (!predictions.length) return []
    const filtered = predictions.filter(
      (r) => r.predicted != null && r.actual != null
    )
    // Sample if > 2000 points
    if (filtered.length > 2000) {
      const step = Math.ceil(filtered.length / 2000)
      return filtered.filter((_, i) => i % step === 0).map((r) => ({
        predicted: r.predicted!,
        actual: r.actual!,
        name: r.name ?? '',
        week: r.week ?? 0,
      }))
    }
    return filtered.map((r) => ({
      predicted: r.predicted!,
      actual: r.actual!,
      name: r.name ?? '',
      week: r.week ?? 0,
    }))
  }, [predictions])

  // Weekly predicted vs actual overlay (average per week)
  const weeklyOverlay = useMemo(() => {
    if (!predictions.length) return []
    const byWeek: Record<number, { predSum: number; actSum: number; n: number }> = {}
    for (const r of predictions) {
      if (r.predicted == null || r.actual == null || r.week == null) continue
      if (!byWeek[r.week]) byWeek[r.week] = { predSum: 0, actSum: 0, n: 0 }
      byWeek[r.week].predSum += r.predicted
      byWeek[r.week].actSum += r.actual
      byWeek[r.week].n += 1
    }
    return Object.entries(byWeek)
      .map(([w, v]) => ({
        week: Number(w),
        avgPredicted: v.n > 0 ? v.predSum / v.n : 0,
        avgActual: v.n > 0 ? v.actSum / v.n : 0,
        n: v.n,
      }))
      .sort((a, b) => a.week - b.week)
  }, [predictions])

  const m = result?.metrics
  const diag = result?.diagnostics

  if (!loading && availableSeasons.length === 0) {
    return (
      <div className="section-card">
        <h2>Time-Series Backtest</h2>
        <p style={{ color: 'var(--color-text-muted)' }}>
          No expanding-window backtest results available yet. Run the backtester:
        </p>
        <pre style={{ background: 'var(--color-bg)', padding: '0.75rem', borderRadius: 8, fontSize: 'var(--text-small)' }}>
          python scripts/run_ts_backtest.py --season 2024 --model gbm
        </pre>
      </div>
    )
  }

  return (
    <div className="section-card">
      <h2>Time-Series Backtest (Expanding Window)</h2>
      <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
        Leakage-free weekly-refit backtest: the model is retrained each week using only data available at prediction time.
      </p>

      {/* Season selector dropdown */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)', fontWeight: 600 }}>
            Prediction Season:
          </label>
          <select
            value={selectedSeason ?? ''}
            onChange={(e) => setSelectedSeason(Number(e.target.value))}
            style={{
              background: 'var(--color-card)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-card-border)',
              borderRadius: 6,
              padding: '0.4rem 0.75rem',
              fontSize: 'var(--text-small)',
            }}
          >
            {availableSeasons.map((s) => (
              <option key={s} value={s}>
                {s}/{s + 1} (Historical Backtest)
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)', fontWeight: 600 }}>
            Position:
          </label>
          <select
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            style={{
              background: 'var(--color-card)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-card-border)',
              borderRadius: 6,
              padding: '0.4rem 0.75rem',
              fontSize: 'var(--text-small)',
            }}
          >
            {POSITIONS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="skeleton" style={{ height: 300 }} />
      ) : result ? (
        <>
          {/* Diagnostics badges */}
          {diag && (
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              {Object.entries(diag).map(([k, v]) => (
                <span
                  key={k}
                  style={{
                    fontSize: '0.7rem',
                    padding: '0.2rem 0.5rem',
                    borderRadius: 4,
                    background: v ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)',
                    color: v ? 'var(--color-accent-emerald)' : '#f87171',
                    border: `1px solid ${v ? 'rgba(74,222,128,0.3)' : 'rgba(248,113,113,0.3)'}`,
                  }}
                >
                  {v ? '✓' : '✗'} {k.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}

          {/* Overall metrics cards */}
          {m && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem', marginBottom: '1.5rem' }}>
              {[
                { label: 'MAE', value: m.mae?.toFixed(2), color: 'var(--color-accent-cyan)' },
                { label: 'RMSE', value: m.rmse?.toFixed(2), color: 'var(--color-accent-purple)' },
                { label: 'R²', value: m.r2?.toFixed(3), color: 'var(--color-accent-emerald)' },
                { label: 'Predictions', value: result.n_predictions?.toLocaleString(), color: 'var(--color-accent-amber)' },
              ].map((card) => (
                <div
                  key={card.label}
                  style={{
                    background: 'var(--color-bg)',
                    border: '1px solid var(--color-card-border)',
                    borderRadius: 8,
                    padding: '0.75rem',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>{card.label}</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: card.color }}>
                    {card.value ?? '–'}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Predicted vs Actual overlay (dual-mode chart) */}
          {weeklyOverlay.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Predicted vs Actual (weekly avg) — {position === 'ALL' ? 'All Positions' : position}
              </h3>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={weeklyOverlay} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                    <XAxis dataKey="week" stroke="var(--color-text-muted)" label={{ value: 'Week', position: 'bottom', fill: 'var(--color-text-muted)' }} />
                    <YAxis stroke="var(--color-text-muted)" label={{ value: 'Fantasy Pts (avg)', angle: -90, position: 'insideLeft', fill: 'var(--color-text-muted)' }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number, name: string) => [value.toFixed(2), name === 'avgPredicted' ? 'Predicted' : 'Actual']}
                      labelFormatter={(l) => `Week ${l}`}
                    />
                    <Legend formatter={(v) => (v === 'avgPredicted' ? 'Predicted' : 'Actual')} />
                    <Line type="monotone" dataKey="avgActual" stroke="var(--color-accent-emerald)" strokeWidth={2.5} dot={{ fill: 'var(--color-accent-emerald)', r: 4 }} name="avgActual" />
                    <Line type="monotone" dataKey="avgPredicted" stroke="var(--color-accent-cyan)" strokeWidth={2.5} strokeDasharray="6 3" dot={{ fill: 'var(--color-accent-cyan)', r: 4 }} name="avgPredicted" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Scatter: Predicted vs Actual */}
          {scatterData.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Prediction Scatter ({scatterData.length.toLocaleString()} points)
              </h3>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                    <XAxis
                      dataKey="predicted" name="Predicted" type="number"
                      stroke="var(--color-text-muted)"
                      label={{ value: 'Predicted', position: 'bottom', fill: 'var(--color-text-muted)' }}
                    />
                    <YAxis
                      dataKey="actual" name="Actual" type="number"
                      stroke="var(--color-text-muted)"
                      label={{ value: 'Actual', angle: -90, position: 'insideLeft', fill: 'var(--color-text-muted)' }}
                    />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number, name: string) => [value.toFixed(1), name]}
                      labelFormatter={() => ''}
                    />
                    <Scatter data={scatterData} fill="var(--color-accent-cyan)" fillOpacity={0.5} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              {predLoading && <p style={{ textAlign: 'center', color: 'var(--color-text-muted)' }}>Loading predictions...</p>}
            </div>
          )}

          {/* Weekly RMSE trend */}
          {weeklyData.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Error by Week (early vs late season comparison)
              </h3>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={weeklyData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                    <XAxis dataKey="week" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number, name: string) => [value.toFixed(2), name.toUpperCase()]}
                      labelFormatter={(l) => `Week ${l}`}
                    />
                    <Legend />
                    <Bar dataKey="mae" fill="var(--color-accent-cyan)" fillOpacity={0.6} radius={[3, 3, 0, 0]} name="mae" />
                    <Line type="monotone" dataKey="rmse" stroke="var(--color-accent-purple)" strokeWidth={2} dot={{ r: 3 }} name="rmse" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Position RMSE comparison */}
          {posData.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: 'var(--text-h3)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
                Error by Position
              </h3>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={posData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-card-border)" />
                    <XAxis dataKey="position" stroke="var(--color-text-muted)" />
                    <YAxis stroke="var(--color-text-muted)" />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-card-border)', borderRadius: 8 }}
                      formatter={(value: number, name: string) => [value.toFixed(2), name.toUpperCase()]}
                    />
                    <Legend />
                    <Bar dataKey="mae" fill="var(--color-accent-cyan)" radius={[3, 3, 0, 0]} name="mae" />
                    <Bar dataKey="rmse" fill="var(--color-accent-purple)" radius={[3, 3, 0, 0]} name="rmse" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      ) : (
        <p style={{ color: 'var(--color-text-muted)' }}>No results for selected season.</p>
      )}
    </div>
  )
}
