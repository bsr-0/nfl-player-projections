# Visual Test Checklist

Run the Streamlit app and verify each component works correctly in both light and dark modes.

## How to Run

```bash
cd /Users/benrosen/nfl_predictor_claude
streamlit run app.py
```

## Light Mode Testing

Toggle Streamlit to light mode: Settings > Theme > Light

| Component | Status | Notes |
|-----------|--------|-------|
| **Sidebar** | | |
| - Navigation radio buttons visible | [ ] | |
| - Data status text readable | [ ] | |
| - Refresh button visible | [ ] | |
| **Dashboard** | | |
| - Hero title gradient visible | [ ] | |
| - Subtitle text readable | [ ] | |
| - Search bar functional | [ ] | |
| **Start/Sit Tab** | | |
| - Player cards have visible borders | [ ] | |
| - Player names readable | [ ] | |
| - START/SIT badges have correct colors | [ ] | |
| - Rank numbers visible | [ ] | |
| - Bar chart renders correctly | [ ] | |
| **Comparison View** | | |
| - Both player cards visible | [ ] | |
| - Winner highlight shows | [ ] | |
| - Stats text readable | [ ] | |
| - Comparison chart renders | [ ] | |
| **Trending Tab** | | |
| - Buy Low cards (green border) visible | [ ] | |
| - Sell High cards (red border) visible | [ ] | |
| - Player names readable | [ ] | |
| - Scatter plot renders | [ ] | |
| **My Roster Tab** | | |
| - Trade analyzer dropdowns work | [ ] | |
| - Trade value chart renders | [ ] | |
| **Deep Dive Tab** | | |
| - Expanders toggle correctly | [ ] | |
| - Model metrics readable | [ ] | |
| - Feature importance chart renders | [ ] | |

## Dark Mode Testing

Toggle Streamlit to dark mode: Settings > Theme > Dark

| Component | Status | Notes |
|-----------|--------|-------|
| **Sidebar** | | |
| - Navigation radio buttons visible | [ ] | |
| - Data status text readable | [ ] | |
| - Refresh button visible | [ ] | |
| **Dashboard** | | |
| - Hero title gradient visible | [ ] | |
| - Subtitle text readable | [ ] | |
| - Search bar functional | [ ] | |
| **Start/Sit Tab** | | |
| - Player cards have visible borders | [ ] | |
| - Player names readable (light text) | [ ] | |
| - START/SIT badges have correct colors | [ ] | |
| - Rank numbers visible | [ ] | |
| - Bar chart has transparent bg | [ ] | |
| **Comparison View** | | |
| - Both player cards visible | [ ] | |
| - Winner highlight shows | [ ] | |
| - Stats text readable (light text) | [ ] | |
| - Comparison chart renders | [ ] | |
| **Trending Tab** | | |
| - Buy Low cards (green border) visible | [ ] | |
| - Sell High cards (red border) visible | [ ] | |
| - Player names readable (light text) | [ ] | |
| - Scatter plot has dark bg | [ ] | |
| **My Roster Tab** | | |
| - Trade analyzer dropdowns work | [ ] | |
| - Trade value chart renders | [ ] | |
| **Deep Dive Tab** | | |
| - Expanders toggle correctly | [ ] | |
| - Model metrics readable | [ ] | |
| - Feature importance chart renders | [ ] | |

## Common Issues to Check

1. **Text invisible**: Text same color as background
2. **Charts not rendering**: Empty data or wrong column names
3. **Cards overlapping**: CSS layout issues
4. **Buttons not working**: State management issues
5. **Filters returning empty**: Data not loaded

## Known Limitations

- Network features (injury data, Vegas lines) require internet connection
- First load may be slow if cache is missing
- Model predictions require trained models in `models/` directory

## Predictions Chart (run_app.py — FastAPI + React)

Run: `python run_app.py --with-predictions` then open http://localhost:8501

| Check | Status | Notes |
|-------|--------|-------|
| **Y-axis names** | | |
| - First, second, and last bar show full player name (and team) on the left | [ ] | Scroll if needed; every name must be visible |
| **Schedule vs tooltip** | | |
| - When "Schedule not available for some weeks" is shown, hover a player; tooltip must NOT show "Matchup:" or opposing team | [ ] | |
| - When "Schedule used for predictions: Yes" is shown, hover a player; tooltip may show Matchup | [ ] | |
| **Filters** | | |
| - Position "All" and Time horizon "All" return data; chart shows 1w/4w/18w bars or all positions | [ ] | |

## Reporting Issues

If you find visual bugs, note:
1. Theme (light/dark)
2. Component name
3. Expected vs actual behavior
4. Browser and screen size
