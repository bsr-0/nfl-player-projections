#!/usr/bin/env python3
"""
Simple launcher for NFL Utilization Dashboard
Run from scripts directory: python launch.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("NFL UTILIZATION DASHBOARD LAUNCHER")
    print("="*70)
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Fetch data
    print("\n📥 Step 1: Fetching current season data...")
    print("-"*70)
    
    realtime_script = scripts_dir / "realtime_integration.py"
    result = subprocess.run([sys.executable, str(realtime_script)])
    
    if result.returncode != 0:
        print("\n⚠️  Data fetch had errors but continuing...")
    
    # Step 2: Launch dashboard
    print("\n"+"="*70)
    print("🚀 Step 2: Launching dashboard...")
    print("="*70)
    print("\nDashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    dashboard_script = scripts_dir / "analytics_dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_script)])

if __name__ == "__main__":
    main()
