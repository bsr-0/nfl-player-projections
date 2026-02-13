# Multi-stage Docker build for NFL Predictor API + Frontend

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/models data/outputs

# Expose FastAPI port
EXPOSE 8501

# Health check against FastAPI health endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/api/health || exit 1

# Run FastAPI application (serves API + static frontend)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8501"]
