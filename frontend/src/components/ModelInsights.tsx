import { useState } from 'react'
import { DataPipeline } from './DataPipeline'
import { Utilization } from './Utilization'
import { Validation } from './Validation'
import { ModelArena } from './ModelArena'
import { Backtest } from './Backtest'
import { TrainingYears } from './TrainingYears'
import { EDA } from './EDA'

type SubTab = 'pipeline' | 'utilization' | 'validation' | 'arena' | 'backtest' | 'training' | 'eda'

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: 'pipeline', label: 'Data Pipeline' },
  { id: 'utilization', label: 'Utilization Scoring' },
  { id: 'validation', label: 'Validation' },
  { id: 'arena', label: 'Model Arena' },
  { id: 'backtest', label: 'Backtesting' },
  { id: 'training', label: 'Training Years' },
  { id: 'eda', label: 'Exploratory Analysis' },
]

export function ModelInsights() {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('pipeline')

  return (
    <div className="model-insights">
      <div className="section-card" style={{ marginBottom: '1.5rem' }}>
        <h2 className="section-heading">Model Insights & Methodology</h2>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
          Explore how our ML pipeline works — from raw data collection through feature engineering, model training, and rigorous backtesting. Full transparency into how projections are generated.
        </p>
        <div className="model-insights__tabs">
          {SUB_TABS.map((tab) => (
            <button
              key={tab.id}
              className={`model-insights__tab ${activeSubTab === tab.id ? 'model-insights__tab--active' : ''}`}
              onClick={() => setActiveSubTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeSubTab === 'pipeline' && <DataPipeline />}
      {activeSubTab === 'utilization' && <Utilization />}
      {activeSubTab === 'validation' && <Validation />}
      {activeSubTab === 'arena' && <ModelArena />}
      {activeSubTab === 'backtest' && <Backtest />}
      {activeSubTab === 'training' && <TrainingYears />}
      {activeSubTab === 'eda' && <EDA />}
    </div>
  )
}
