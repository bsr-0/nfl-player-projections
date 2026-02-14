#!/usr/bin/env bash
# Production build script for NFL Predictor.
# Builds the frontend and prepares the app for deployment.
# Works on any PaaS (Render, Railway, Fly.io) or locally.
#
# Usage:
#   chmod +x build.sh
#   ./build.sh

set -e

echo "=== NFL Predictor: Production Build ==="

# 1. Install Python dependencies
if [ -f requirements-web.txt ]; then
    echo ">> Installing Python dependencies (web-only)..."
    pip install --no-cache-dir -r requirements-web.txt
else
    echo ">> Installing Python dependencies (full)..."
    pip install --no-cache-dir -r requirements.txt
fi

# 2. Build frontend (if Node.js is available)
if command -v node &> /dev/null; then
    echo ">> Building frontend..."
    cd frontend
    npm ci
    npm run build
    cd ..
    echo ">> Frontend built to frontend/dist/"
else
    echo ">> Node.js not found; skipping frontend build."
    echo "   Pre-build the frontend locally: cd frontend && npm run build"
    if [ -d frontend/dist ]; then
        echo "   (frontend/dist already exists — using existing build)"
    fi
fi

echo "=== Build complete ==="
