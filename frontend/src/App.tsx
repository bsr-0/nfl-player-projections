import { useEffect, useState } from 'react'
import './App.css'
import { api, type PredictionRow } from './api'
import { DashboardView } from './components/DashboardView'
import { Footer } from './components/Footer'

function App() {
  const [allData, setAllData] = useState<Record<string, PredictionRow[]>>({})
  const [loading, setLoading] = useState(true)
  const [weekLabel, setWeekLabel] = useState('')
  const [scheduleAvailable, setScheduleAvailable] = useState<boolean>(true)

  // Fetch all positions with 18-week (full season) horizon
  useEffect(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    Promise.all(positions.map((p) => api.predictions(p, undefined, 18)))
      .then((results) => {
        const map: Record<string, PredictionRow[]> = {}
        results.forEach((d, i) => {
          map[positions[i]] = d.rows
        })
        setAllData(map)
        // Use metadata from first successful response
        const first = results.find((d) => d.week_label)
        if (first) {
          setWeekLabel(first.week_label ?? '')
          setScheduleAvailable(first.schedule_available ?? true)
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__logo">🏈</span>
          <div>
            <h1 className="app-header__title">NFL Fantasy Predictor</h1>
            <p className="app-header__subtitle">ML-powered projections for smarter lineup decisions</p>
          </div>
        </div>
        {weekLabel && <div className="app-header__week-badge">{weekLabel}</div>}
      </header>

      <main className="app-main">
        <DashboardView
          allData={allData}
          weekLabel={weekLabel}
          loading={loading}
          scheduleAvailable={scheduleAvailable}
        />
      </main>

      <Footer />
    </div>
  )
}

export default App
