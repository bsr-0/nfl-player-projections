# NFL Predictor – API

FastAPI backend that serves all data for the SPA: hero stats, data pipeline, training years, EDA sample, utilization weights, advanced results, backtest, predictions.

## Endpoints

- `GET /api/hero` – record count, optional correlation
- `GET /api/data-pipeline` – row_count, seasons, n_features, health
- `GET /api/training-years` – training years analysis (CSV as JSON)
- `GET /api/eda?max_rows=5000` – EDA sample + stats
- `GET /api/utilization-weights` – per-position weights
- `GET /api/advanced-results` – full advanced_model_results.json
- `GET /api/backtest` – list + latest backtest (by_position, by_week, metrics, baseline_comparison)
- `GET /api/backtest/{season}` – backtest for a specific season
- `GET /api/predictions?position=&name=&horizon=` – predictions table (from parquet); optional position, name (substring), and horizon (1, 4, or 18 weeks; adds `projected_points` per row)
- `GET /api/health` – health check

## Run

From the **project root** (so `config` and `src` are importable):

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Using `python -m uvicorn` avoids needing `uvicorn` on your PATH.

CORS is enabled for all origins. If `frontend/dist` exists, the app also serves the built SPA at `/`.
