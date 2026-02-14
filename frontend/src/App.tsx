import { useEffect, useState, useCallback } from 'react'
import './App.css'
import { api, type PredictionsResponse, type PredictionRow } from './api'
import { DashboardView } from './components/DashboardView'
import { RankingsView } from './components/RankingsView'
import { PlayerLookup } from './components/PlayerLookup'
import { ModelInsights } from './components/ModelInsights'
import { DraftAssistant } from './components/DraftAssistant'
import { Footer } from './components/Footer'

type TabId = 'dashboard' | 'rankings' | 'player' | 'draft' | 'model'

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'rankings', label: 'Rankings', icon: '🏆' },
  { id: 'draft', label: 'Draft Assistant', icon: '📋' },
  { id: 'player', label: 'Player Lookup', icon: '🔍' },
  { id: 'model', label: 'Model Insights', icon: '🧠' },
]

type HorizonValue = 1 | 4 | 18

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard')
  const [position, setPosition] = useState<string>('QB')
  const [horizon, setHorizon] = useState<HorizonValue>(1)
  const [data, setData] = useState<PredictionsResponse | null>(null)
  const [allData, setAllData] = useState<Record<string, PredictionRow[]>>({})
  const [draftData, setDraftData] = useState<Record<string, PredictionRow[]>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [weekLabel, setWeekLabel] = useState('')
  const [qbTarget, setQbTarget] = useState<'util' | 'fp'>('fp')

  const fetchPredictions = useCallback(() => {
    setLoading(true)
    setError(null)
    api
      .predictions(position, undefined, horizon)
      .then((d) => {
        setData(d)
        setWeekLabel(d.week_label ?? '')
      })
      .catch((err) => {
        setData(null)
        setError(err instanceof Error ? err.message : 'Failed to load predictions')
      })
      .finally(() => setLoading(false))
  }, [position, horizon])

  useEffect(() => {
    fetchPredictions()
  }, [fetchPredictions])

  useEffect(() => {
    api.modelConfig().then((cfg) => {
      if (cfg?.qb_target === 'util' || cfg?.qb_target === 'fp') {
        setQbTarget(cfg.qb_target)
      }
    }).catch(() => {})
  }, [])

  // Fetch all positions for dashboard overview (1-week)
  useEffect(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    Promise.all(positions.map((p) => api.predictions(p, undefined, 1).then((d) => ({ pos: p, rows: d.rows }))))
      .then((results) => {
        const map: Record<string, PredictionRow[]> = {}
        results.forEach(({ pos, rows }) => { map[pos] = rows })
        setAllData(map)
      })
      .catch(() => {})
  }, [])

  // Fetch all positions with 18-week horizon for draft assistant
  useEffect(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    Promise.all(positions.map((p) => api.predictions(p, undefined, 18).then((d) => ({ pos: p, rows: d.rows }))))
      .then((results) => {
        const map: Record<string, PredictionRow[]> = {}
        results.forEach(({ pos, rows }) => { map[pos] = rows })
        setDraftData(map)
      })
      .catch(() => {})
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

      <nav className="tab-nav" role="tablist" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`tab-nav__btn ${activeTab === tab.id ? 'tab-nav__btn--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-nav__icon">{tab.icon}</span>
            <span className="tab-nav__label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <main className="app-main" role="tabpanel">
        {activeTab === 'dashboard' && (
          <DashboardView
            allData={allData}
            weekLabel={weekLabel}
            loading={loading && Object.keys(allData).length === 0}
          />
        )}
        {activeTab === 'rankings' && (
          <RankingsView
            data={data}
            position={position}
            horizon={horizon}
            loading={loading}
            error={error}
            onPositionChange={setPosition}
            onHorizonChange={setHorizon}
          />
        )}
        {activeTab === 'draft' && (
          <DraftAssistant allData={draftData} weekLabel={weekLabel} qbTarget={qbTarget} />
        )}
        {activeTab === 'player' && (
          <PlayerLookup allData={allData} weekLabel={weekLabel} />
        )}
        {activeTab === 'model' && <ModelInsights />}
      </main>

      <Footer />
    </div>
  )
}

export default App
