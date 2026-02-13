import { useEffect, useState, useCallback } from 'react'
import './App.css'
import { api, type PredictionsResponse } from './api'
import { PredictionsChart } from './components/PredictionsChart'

const POSITIONS = ['QB', 'RB', 'WR', 'TE'] as const
const POSITION_OPTIONS = [...POSITIONS]

type HorizonValue = 1 | 4 | 18
const HORIZONS: { value: HorizonValue; label: string }[] = [
  { value: 1, label: '1 week' },
  { value: 4, label: '4 weeks' },
  { value: 18, label: '18 weeks' },
]

function App() {
  const [position, setPosition] = useState<string>('QB')
  const [horizon, setHorizon] = useState<HorizonValue>(1)
  const [data, setData] = useState<PredictionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPredictions = useCallback(() => {
    setLoading(true)
    setError(null)
    api
      .predictions(position, undefined, horizon)
      .then(setData)
      .catch((err) => {
        setData(null)
        setError(err instanceof Error ? err.message : 'Failed to load predictions')
      })
      .finally(() => setLoading(false))
  }, [position, horizon])

  useEffect(() => {
    fetchPredictions()
  }, [fetchPredictions])

  const horizonLabel = HORIZONS.find((h) => h.value === horizon)?.label ?? `${horizon} weeks`

  return (
    <main className="app app--predictions">
      <header className="app-header">
        <h1 className="app-header__title">NFL Fantasy Predictor</h1>
        <p className="app-header__subtitle">QB: fantasy points; RB/WR/TE: utilization score — by time horizon (matchup-aware)</p>
      </header>

      <section className="filters section-card">
        <div className="filters__row">
          <label className="filters__label" htmlFor="position">
            Position
          </label>
          <select
            id="position"
            className="filters__select"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            aria-label="Filter by position"
          >
            {POSITION_OPTIONS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <label className="filters__label" htmlFor="horizon">
            Time horizon
          </label>
          <select
            id="horizon"
            className="filters__select"
            value={String(horizon)}
            onChange={(e) => setHorizon(Number(e.target.value) as HorizonValue)}
            aria-label="Filter by time horizon"
          >
            {HORIZONS.map((h) => (
              <option key={String(h.value)} value={String(h.value)}>
                {h.label}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="chart-section">
        <PredictionsChart
          rows={data?.rows ?? []}
          weekLabel={data?.week_label ?? ''}
          horizonLabel={horizonLabel}
          horizonWeeksLabel={data?.horizon_weeks_label ?? ''}
          scheduleNote={data?.schedule_note ?? ''}
          scheduleAvailable={data?.schedule_available}
          scheduleByHorizon={data?.schedule_by_horizon}
          horizon={horizon}
          positionFilter={position}
          loading={loading}
          error={error}
        />
      </section>
    </main>
  )
}

export default App
