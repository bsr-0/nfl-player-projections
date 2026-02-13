import { useEffect, useState } from 'react'
import { api } from '../api'

interface PipelineData {
  row_count: number
  seasons: number[]
  season_range: string
  n_features: number
  health: string
}

const STAGES = [
  { id: 'api', label: 'nfl-data-py API', key: 'row_count', suffix: ' records' },
  { id: 'db', label: 'Local DB', key: 'season_range', suffix: '' },
  { id: 'fe', label: 'Feature engineering', key: 'n_features', suffix: ' features' },
  { id: 'qv', label: 'Quality validation', key: 'health', suffix: '' },
] as const

export function DataPipeline() {
  const [data, setData] = useState<PipelineData | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.dataPipeline().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [])

  const getHoverText = (key: string) => {
    if (!data) return ''
    if (key === 'row_count') return `${data.row_count.toLocaleString()} records`
    if (key === 'season_range') return data.season_range || `${data.seasons?.join(', ')}`
    if (key === 'n_features') return `${data.n_features} features`
    if (key === 'health') return data.health
    return ''
  }

  return (
    <div className="section-card">
      <h2>Data collection and compilation</h2>
      <p>
        Player-level stats are loaded from nfl-data-py into a local database. Features and targets are built with strict time ordering: no future information is used.
      </p>
      {loading ? (
        <div className="skeleton" style={{ height: 120, marginTop: '1rem' }} />
      ) : (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '1rem',
            marginTop: '1.5rem',
          }}
          role="img"
          aria-label="Pipeline: nfl-data-py API to Local DB to Feature engineering to Quality validation"
        >
          {STAGES.map((stage, i) => (
            <div
              key={stage.id}
              className="pipeline-stage"
              style={{
                flex: '1',
                minWidth: 140,
                padding: '1rem',
                background: 'var(--color-card)',
                borderRadius: 8,
                border: '1px solid var(--color-card-border)',
                textAlign: 'center',
                position: 'relative',
              }}
              title={getHoverText(stage.key)}
            >
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-small)', color: 'var(--color-text-muted)' }}>
                {stage.label}
              </div>
              {data && (
                <div style={{ marginTop: '0.5rem', fontWeight: 600, color: 'var(--color-accent-cyan)' }}>
                  {stage.key === 'row_count' && data.row_count.toLocaleString()}
                  {stage.key === 'season_range' && (data.season_range || '—')}
                  {stage.key === 'n_features' && `${data.n_features}`}
                  {stage.key === 'health' && data.health}
                </div>
              )}
              {i < STAGES.length - 1 && (
                <span style={{ position: 'absolute', right: -12, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-card-border)' }} aria-hidden>→</span>
              )}
            </div>
          ))}
        </div>
      )}
      {data && (
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap' }}>
          <div className="narr-card" style={{ flex: 1, minWidth: 100, padding: '0.75rem', borderRadius: 8, background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', textAlign: 'center' }}>
            <div className="value" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>{data.row_count.toLocaleString()}</div>
            <div className="label" style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Records</div>
          </div>
          <div className="narr-card" style={{ flex: 1, minWidth: 100, padding: '0.75rem', borderRadius: 8, background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', textAlign: 'center' }}>
            <div className="value" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>{data.season_range || '—'}</div>
            <div className="label" style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Seasons</div>
          </div>
          <div className="narr-card" style={{ flex: 1, minWidth: 100, padding: '0.75rem', borderRadius: 8, background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', textAlign: 'center' }}>
            <div className="value" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-accent-cyan)' }}>{data.n_features}</div>
            <div className="label" style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Features</div>
          </div>
          <div className="narr-card" style={{ flex: 1, minWidth: 100, padding: '0.75rem', borderRadius: 8, background: 'var(--color-bg)', border: '1px solid var(--color-card-border)', textAlign: 'center' }}>
            <div className="value" style={{ fontSize: '1.25rem', fontWeight: 700, color: data.health === 'OK' ? 'var(--color-accent-emerald)' : 'var(--color-accent-amber)' }}>{data.health}</div>
            <div className="label" style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Data health</div>
          </div>
        </div>
      )}
    </div>
  )
}
