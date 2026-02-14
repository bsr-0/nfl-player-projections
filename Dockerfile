# Multi-stage Docker build for NFL Predictor API + Frontend

# ── Stage 1: Build frontend ──
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python API ──
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy frontend build from stage 1
COPY --from=frontend-build /build/dist /app/frontend/dist

# Create data directories
RUN mkdir -p data/models data/outputs data/backtest_results

# Default port (overridden by $PORT on PaaS platforms)
ENV PORT=8501
EXPOSE ${PORT}

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/api/health || exit 1

# Run FastAPI application (serves API + static frontend)
# Use shell form so $PORT is expanded at runtime
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
