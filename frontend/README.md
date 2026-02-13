# NFL Predictor – Frontend (SPA)

Premium dark-theme single-page app for the NFL Fantasy Predictor (methodology, EDA, model arena, validation, backtest, live predictions, by-player analysis).

## Prerequisites

You need **Node.js** (and **npm**) on your PATH. If you see `command not found: npm`:

- **macOS (Homebrew):** `brew install node`
- **Otherwise:** install from [nodejs.org](https://nodejs.org/) (LTS), or use [nvm](https://github.com/nvm-sh/nvm) and run `nvm install --lts`

Then restart your terminal so `npm` is available.

## Stack

- **React 18** + TypeScript
- **Vite** for build and dev server
- **Recharts** for bar/line charts
- **D3** available for custom visualizations

## Run locally

1. Install dependencies (from repo root or `frontend/`):

   ```bash
   cd frontend && npm install
   ```

2. Start the API (from repo root):

   ```bash
   python -m uvicorn api.main:app --reload --port 8000
   ```

3. Start the frontend dev server (from `frontend/`):

   ```bash
   npm run dev
   ```

   Vite proxies `/api` to `http://localhost:8000`, so the app will load data from the FastAPI backend.

4. Open `http://localhost:5173` (or the URL Vite prints).

## Build for production

```bash
npm run build
```

Output is in `frontend/dist/`. The FastAPI app serves this when you run:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

from the project root (and `frontend/dist` exists).

## Theme

- Base: `#0a0e1f` (deep space blue)
- Cards: `#1a1f3a` with subtle gradient and backdrop blur
- Accents: cyan `#00f5ff`, purple `#a78bfa`, emerald `#10b981`
- Typography: Space Grotesk (headings), Inter (body), JetBrains Mono (data)
