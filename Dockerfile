# Docker build for NFL Predictor API + static frontend

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
