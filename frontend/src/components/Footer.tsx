import { useEffect, useState } from 'react'
import { api, type AdvancedResults } from '../api'

export function Footer() {
  const [data, setData] = useState<AdvancedResults | null>(null)
  useEffect(() => {
    api.advancedResults().then(setData).catch(() => setData(null))
  }, [])

  const trainSeasons = data?.train_seasons ?? []
  const testSeason = data?.test_season
  const timestamp = data?.timestamp
  const trainRange = trainSeasons.length ? `${Math.min(...trainSeasons)}–${Math.max(...trainSeasons)}` : '—'

  return (
    <footer
      className="section-card"
      style={{
        marginTop: '3rem',
        paddingTop: '1rem',
        borderTop: '1px solid var(--color-card-border)',
        color: 'var(--color-text-muted)',
        fontSize: 'var(--text-small)',
      }}
      role="contentinfo"
    >
      <p>
        Training data: {trainRange}. Test season: {testSeason ?? '—'}.
      </p>
      <p>
        <a href="docs/" className="cta-link">Methodology (docs)</a>
        {' · '}
        <a href="README.md" className="cta-link">README</a>
        {' · '}
        <a href="docs/BACKTESTING.md" className="cta-link">Backtesting</a>
      </p>
      {timestamp && (
        <p style={{ marginTop: '0.5rem' }}>
          Last updated: {timestamp}. Model accuracy varies by position and game script volatility.
        </p>
      )}
    </footer>
  )
}
