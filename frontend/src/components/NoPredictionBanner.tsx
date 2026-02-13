import { useEffect, useState } from 'react'
import { api } from '../api'

export function NoPredictionBanner() {
  const [hasPredictions, setHasPredictions] = useState<boolean | null>(null)
  useEffect(() => {
    api.predictions().then((d) => setHasPredictions(d.rows.length > 0)).catch(() => setHasPredictions(false))
  }, [])

  if (hasPredictions !== false) return null

  return (
    <div
      className="section-card"
      style={{ borderColor: 'var(--color-accent-amber)' }}
      role="alert"
    >
      <h2 style={{ color: 'var(--color-accent-amber)' }}>No prediction data</h2>
      <p>
        Run the following in your terminal to generate predictions:
      </p>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-small)', background: 'var(--color-bg)', padding: '0.75rem', borderRadius: 8 }}>
        python scripts/generate_app_data.py
      </p>
      <p>
        Optionally add <code>--parquet</code>. Then refresh this page.
      </p>
    </div>
  )
}
