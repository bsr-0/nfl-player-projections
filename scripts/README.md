# NFL Utilization Dashboard - Quick Start

## ⚠️ You're in the SCRIPTS directory!

Current location: `/Users/benrosen/CascadeProjects/nfl-predictor/scripts`

## ✅ CORRECT Commands (from scripts directory):

```bash
# Option 1: Run individual commands
python realtime_integration.py
streamlit run analytics_dashboard.py

# Option 2: Use launcher (runs both)
python launch.py

# Option 3: Use shell script
./run_dashboard.sh
```

## ❌ WRONG Commands (these won't work):

```bash
# Don't use these - you're already IN scripts!
python scripts/realtime_integration.py  ❌
streamlit run scripts/analytics_dashboard.py  ❌
```

## If you want to run from project root:

```bash
# First go back to project root
cd ..

# Then run:
python scripts/realtime_integration.py
streamlit run scripts/analytics_dashboard.py
```

## Files in this directory:
- ✅ analytics_dashboard.py (Main dashboard)
- ✅ realtime_integration.py (Data fetcher)  
- ✅ launch.py (Python launcher - runs both)
- ✅ run_dashboard.sh (Shell launcher)
- ✅ README.md (This file)
