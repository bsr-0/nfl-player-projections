#!/bin/bash

# You're already in the scripts directory, so use relative paths

# Step 1: Fetch current season data
echo "Step 1: Fetching 2025-26 season data..."
python realtime_integration.py

# Step 2: Launch dashboard
echo "Step 2: Launching dashboard..."
streamlit run analytics_dashboard.py
