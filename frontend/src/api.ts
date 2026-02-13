const API_BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`)
  return res.json()
}

export const api = {
  hero: () => get<{ record_count: number; correlation?: number }>('/hero'),
  dataPipeline: () =>
    get<{ row_count: number; seasons: number[]; season_range: string; n_features: number; health: string }>('/data-pipeline'),
  trainingYears: () =>
    get<Array<{ train_years?: number; train_range?: string; position: string; test_correlation: number; n_train?: number }>>('/training-years'),
  eda: (maxRows = 5000) =>
    get<{ sample: Record<string, unknown>[]; stats: { row_count: number; seasons: number[]; n_features: number } }>(
      `/eda?max_rows=${maxRows}`
    ),
  utilizationWeights: () =>
    get<Record<string, Record<string, number>>>('/utilization-weights'),
  advancedResults: () => get<AdvancedResults>('/advanced-results'),
  backtest: () => get<BacktestResponse>('/backtest'),
  predictions: (position?: string, name?: string, horizon?: 1 | 4 | 18 | 'all') => {
    const params = new URLSearchParams()
    if (position) params.set('position', position)
    if (name) params.set('name', name)
    if (horizon !== undefined) params.set('horizon', String(horizon))
    return get<PredictionsResponse>(`/predictions?${params}`)
  },
}

export interface AdvancedResults {
  timestamp?: string
  train_seasons?: number[]
  test_season?: number
  backtest_results?: Record<string, PositionResult>
  baseline_comparison?: BaselineComparison
}

export interface PositionResult {
  position: string
  best_model?: string
  r2?: number
  rmse?: number
  mae?: number
  time_seconds?: number
  comparison?: Array<{ model: string; r2?: number; rmse?: number; mae?: number }>
}

export interface BaselineComparison {
  model: { r2?: number; rmse?: number; mae?: number }
  baseline: { r2?: number; rmse?: number; mae?: number }
  improvement?: { rmse_pct?: number; r2_gain?: number }
}

export interface BacktestResponse {
  files: Array<{ season?: number; backtest_date?: string }>
  latest: {
    by_position: Record<string, { r2?: number; correlation?: number; [k: string]: unknown }>
    by_week: Record<string, { r2?: number; correlation?: number; [k: string]: unknown }>
    metrics: Record<string, unknown>
    baseline_comparison?: BaselineComparison
  }
  train_seasons?: number[]
  test_season?: number
}

export interface PredictionsResponse {
  rows: PredictionRow[]
  week_label: string
  upcoming_week_label?: string | null
  schedule_available?: boolean
  schedule_note?: string
  schedule_by_horizon?: Record<string, boolean>
  horizon_weeks?: number[][]
  horizon_weeks_label?: string
}

export interface PredictionRow {
  name?: string
  position?: string
  team?: string
  team_next_season?: string
  projection_1w?: number
  projection_4w?: number
  projection_18w?: number
  predicted_points?: number
  projected_points?: number
  prediction_std?: number
  utilization_score?: number
  upcoming_opponent?: string
  upcoming_home_away?: string
  season?: number
  week?: number
  [k: string]: unknown
}
