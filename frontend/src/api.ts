const API_BASE = import.meta.env.VITE_API_BASE || '/api'
const API_MODE = import.meta.env.VITE_API_MODE || 'live'
const IS_STATIC = API_MODE === 'static'

function mapStaticPath(path: string): string {
  if (path.startsWith('/hero')) return '/hero.json'
  if (path.startsWith('/data-pipeline')) return '/data-pipeline.json'
  if (path.startsWith('/training-years')) return '/training-years.json'
  if (path.startsWith('/eda')) return '/eda.json'
  if (path.startsWith('/utilization-weights')) return '/utilization-weights.json'
  if (path.startsWith('/model-config')) return '/model-config.json'
  if (path.startsWith('/advanced-results')) return '/advanced-results.json'
  if (path.startsWith('/backtest/')) {
    const season = path.split('/').pop()
    return `/backtest-${season}.json`
  }
  if (path.startsWith('/backtest')) return '/backtest.json'
  if (path.startsWith('/predictions')) return '/predictions.json'
  if (path.startsWith('/ts-backtest/')) {
    const parts = path.split('/')
    const season = parts[2]
    const tail = parts[3]
    if (tail === 'predictions') return `/ts-backtest-${season}-predictions.json`
    return `/ts-backtest-${season}.json`
  }
  if (path.startsWith('/ts-backtest')) return '/ts-backtest.json'
  return path
}

async function get<T>(path: string): Promise<T> {
  const finalPath = IS_STATIC ? mapStaticPath(path) : path
  const res = await fetch(`${API_BASE}${finalPath}`)
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`)
  return res.json()
}

export const api = {
  hero: () => get<{ record_count: number; correlation?: number }>('/hero'),
  dataPipeline: () =>
    get<{ row_count: number; seasons: number[]; season_range: string; n_features: number; health: string }>('/data-pipeline'),
  trainingYears: () =>
    get<Array<{ train_years?: number; train_range?: string; position: string; test_correlation: number; n_train?: number }>>('/training-years'),
  eda: (maxRows = 5000) => {
    if (IS_STATIC) {
      return get<{ sample: Record<string, unknown>[]; stats: { row_count: number; seasons: number[]; n_features: number } }>(
        '/eda'
      )
    }
    return get<{ sample: Record<string, unknown>[]; stats: { row_count: number; seasons: number[]; n_features: number } }>(
      `/eda?max_rows=${maxRows}`
    )
  },
  utilizationWeights: () =>
    get<Record<string, Record<string, number>>>('/utilization-weights'),
  modelConfig: () =>
    get<{ qb_target: 'util' | 'fp' }>('/model-config'),
  advancedResults: () => get<AdvancedResults>('/advanced-results'),
  backtest: () => get<BacktestResponse>('/backtest'),
  backtestSeason: (season: number) => get<BacktestSeasonResponse>(`/backtest/${season}`),
  predictions: async (position?: string, name?: string, horizon?: 1 | 4 | 18 | 'all') => {
    if (!IS_STATIC) {
      const params = new URLSearchParams()
      if (position) params.set('position', position)
      if (name) params.set('name', name)
      if (horizon !== undefined) params.set('horizon', String(horizon))
      return get<PredictionsResponse>(`/predictions?${params}`)
    }

    const payload = await get<PredictionsResponse>('/predictions')
    let rows = payload.rows || []
    if (position) {
      rows = rows.filter((r) => String(r.position || '').toUpperCase() === position.toUpperCase())
    }
    if (name && name.trim()) {
      const q = name.trim().toLowerCase()
      rows = rows.filter((r) => String(r.name || '').toLowerCase().includes(q))
    }

    let projectedCol: keyof PredictionRow | undefined
    if (horizon === 1 || horizon === 4 || horizon === 18) {
      const col = `projection_${horizon}w` as keyof PredictionRow
      const hasCol = rows.some((r) => r[col] !== undefined)
      if (horizon === 1 && !hasCol) {
        projectedCol = 'predicted_points'
      } else if (hasCol) {
        projectedCol = col
      }
    }

    if (projectedCol !== undefined) {
      rows = rows.map((r) => ({
        ...r,
        projected_points: r[projectedCol] as number | undefined,
      }))
    }

    return { ...payload, rows }
  },
  tsBacktest: () => get<TSBacktestListResponse>('/ts-backtest'),
  tsBacktestSeason: (season: number) => get<TSBacktestResult>(`/ts-backtest/${season}`),
  tsBacktestPredictions: async (season: number, position?: string, week?: number) => {
    if (!IS_STATIC) {
      const params = new URLSearchParams()
      if (position) params.set('position', position)
      if (week !== undefined) params.set('week', String(week))
      return get<TSBacktestPredictionsResponse>(`/ts-backtest/${season}/predictions?${params}`)
    }

    const payload = await get<TSBacktestPredictionsResponse>(`/ts-backtest/${season}/predictions`)
    let rows = payload.rows || []
    if (position) rows = rows.filter((r) => String(r.position || '').toUpperCase() === position.toUpperCase())
    if (week !== undefined) rows = rows.filter((r) => r.week === week)
    return { ...payload, rows }
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
    season?: number
    backtest_date?: string
  }
  train_seasons?: number[]
  test_season?: number
}

export interface PredictionsResponse {
  rows: PredictionRow[]
  qb_target?: 'util' | 'fp'
  week_label: string
  upcoming_week_label?: string | null
  schedule_available?: boolean
  schedule_note?: string
  schedule_by_horizon?: Record<string, boolean>
  horizon_weeks?: number[][]
  horizon_weeks_label?: string
}

export interface TSBacktestMetrics {
  mae?: number
  rmse?: number
  r2?: number
  mape?: number
  n?: number
}

export interface TSBacktestResult {
  season?: number
  backtest_type?: string
  backtest_date?: string
  n_predictions?: number
  positions?: string[]
  metrics?: TSBacktestMetrics
  by_week?: Record<string, TSBacktestMetrics>
  by_position?: Record<string, TSBacktestMetrics>
  diagnostics?: Record<string, boolean>
}

export interface TSBacktestListResponse {
  results: TSBacktestResult[]
  latest: TSBacktestResult | null
  available_seasons: number[]
}

export interface TSBacktestPredictionRow {
  season?: number
  week?: number
  player_id?: string
  name?: string
  position?: string
  team?: string
  predicted?: number
  actual?: number
  prediction_timestamp?: string
}

export interface TSBacktestPredictionsResponse {
  rows: TSBacktestPredictionRow[]
  season: number
}

export interface BacktestSeasonResponse {
  season?: number
  backtest_date?: string
  n_predictions?: number
  metrics?: Record<string, number>
  by_position?: Record<string, Record<string, number>>
  by_week?: Record<string, Record<string, number>>
  top_performers?: Record<string, {
    top_10_actual?: Array<{
      name: string
      actual_rank: number
      pred_rank: number
      fantasy_points: number
      predicted_points: number
    }>
    avg_pred_rank_of_top_10?: number
    top_10_in_our_top_20?: number
  }>
  biggest_misses?: Array<{
    name: string
    position: string
    team: string
    week: number
    fantasy_points: number
    predicted_points: number
    error: number
  }>
  ranking_accuracy?: Record<string, {
    top_5_hit_rate?: number
    top_10_hit_rate?: number
    top_20_hit_rate?: number
  }>
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
