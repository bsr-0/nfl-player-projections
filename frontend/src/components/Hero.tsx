import { useEffect, useState } from 'react'
import { api } from '../api'

function useCountUp(end: number, durationMs = 2000, enabled = true) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (!enabled || end <= 0) {
      setValue(end)
      return
    }
    let startTime: number
    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp
      const elapsed = timestamp - startTime
      const progress = Math.min(elapsed / durationMs, 1)
      setValue(Math.floor(progress * end))
      if (progress < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [end, durationMs, enabled])
  return value
}

export function Hero() {
  const [data, setData] = useState<{ record_count: number; correlation?: number } | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.hero().then(setData).catch(() => setData({ record_count: 0 })).finally(() => setLoading(false))
  }, [])
  const recordCount = data?.record_count ?? 0
  const displayCount = useCountUp(recordCount, 2000, !loading && recordCount > 0)
  const correlation = data?.correlation

  return (
    <header className="section-card narr-hero" style={{ textAlign: 'center', padding: '3rem 1rem 2rem' }}>
      <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: 'var(--text-h1)', color: 'var(--color-text-primary)', marginBottom: '0.5rem' }}>
        Predict NFL Fantasy Performance from Opportunity
      </h1>
      <p style={{ color: 'var(--color-text-secondary)', fontSize: '1.1rem' }}>
        From raw chaos to validated insights: utilization-based modeling with rigorous backtesting.
      </p>
      <p style={{ marginTop: '0.75rem' }}>
        <a href="docs/" className="cta-link" style={{ marginRight: '1rem' }}>Methodology (docs)</a>
        <a href="README.md" className="cta-link">README</a>
      </p>
      {loading ? (
        <div className="skeleton" style={{ height: 48, width: 200, margin: '1rem auto' }} />
      ) : (
        <p
          className="narr-counter"
          style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--color-accent-cyan)', margin: '1rem 0' }}
          aria-live="polite"
        >
          {displayCount.toLocaleString()} player-game records
        </p>
      )}
      {correlation != null && (
        <p style={{ fontSize: '1.1rem', color: 'var(--color-accent-emerald)', fontWeight: 600 }}>
          {correlation}% correlation achieved (1-week test)
        </p>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '0.75rem', marginTop: '1.5rem' }}>
        <span className="pill" aria-label="No data leakage">
          <LockIcon /> No data leakage
        </span>
        <span className="pill" aria-label="Cross-validated">
          <CheckIcon /> Cross-validated
        </span>
        <span className="pill" aria-label="Time-series split">
          <CalendarIcon /> Time-series split
        </span>
      </div>
      <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '0.9rem', marginTop: '1rem' }}>
        <a href="#data-pipeline" className="cta-link">
          See the methodology
          <span style={{ display: 'inline-block', marginLeft: '0.25rem', transition: 'transform 0.2s' }} aria-hidden>→</span>
        </a>
      </p>
    </header>
  )
}

function LockIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}
function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  )
}
