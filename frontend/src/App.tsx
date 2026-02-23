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
  const [defaultHorizon, setDefaultHorizon] = useState<number | null>(null)

  // Fetch all positions with dynamic default horizon (full season in offseason, remaining weeks mid-season)
  useEffect(() => {
    api.predictions(undefined, undefined, 'all')
      .then((d) => {
        const map: Record<string, PredictionRow[]> = {}
        const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        positions.forEach((p) => {
          map[p] = (d.rows || []).filter((r) => String(r.position || '').toUpperCase() === p)
        })
        setAllData(map)
        setWeekLabel(d.default_horizon_label ?? d.week_label ?? '')
        setScheduleAvailable(d.schedule_available ?? true)
        setDefaultHorizon(d.default_horizon ?? null)
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
          defaultHorizon={defaultHorizon ?? 18}
        />
      </main>

      <Footer />
    </div>
  )
}

export default App
