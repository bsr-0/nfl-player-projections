# Calibrated simulation backtest (unrestricted)

Scored 12456 player-games, 555 games, 44 weeks, seasons [2024, 2025, 2026]; 1000 draws per player. Lower is better except coverage (target 0.50 / 0.80). Every calibrated support is centred on the served prediction, so MAE differences from production_marginal are Monte Carlo noise in the draw mean.

## Modes

| mode | MAE | CRPS | cov50 | cov80 | energy | variogram | scaled variogram | stack CRPS | stack cov80 | team CRPS | game CRPS | below support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_marginal | 4.395 | 4.395 | 0.000 | 0.000 | 28.062 | 1.797 | 1.0998 | 12.930 | 0.000 | 19.042 | 29.741 | 0.0000 |
| calibrated_independent_legacy | 4.397 | 3.183 | 0.498 | 0.797 | 20.012 | 1.520 | 0.4201 | 9.405 | 0.635 | 13.574 | 21.294 | 0.1695 |
| calibrated_independent | 4.395 | 3.048 | 0.489 | 0.787 | 19.943 | 1.416 | 0.3128 | 9.241 | 0.687 | 13.585 | 21.334 | 0.1336 |
| calibrated_team_factor | 4.395 | 3.048 | 0.489 | 0.787 | 19.951 | 1.417 | 0.3120 | 9.220 | 0.700 | 13.503 | 21.017 | 0.1336 |
| calibrated_role_factor | 4.395 | 3.048 | 0.489 | 0.787 | 19.940 | 1.416 | 0.3122 | 9.148 | 0.744 | 13.484 | 20.964 | 0.1336 |
| calibrated_role_correlation | 4.395 | 3.048 | 0.489 | 0.787 | 19.978 | 1.419 | 0.3128 | 9.152 | 0.736 | 13.487 | 21.002 | 0.1336 |

## Pre-registered primary comparisons

Week-block bootstrap, Holm-corrected at alpha=0.05. Negative delta favours the candidate.

| candidate | baseline | table | metric | delta | 95% CI | p | Holm |
|---|---|---|---|---|---|---|---|
| calibrated_independent | calibrated_independent_legacy | marginal | crps | -0.1345 | [-0.1496, -0.1169] | 0.0020 | significant |
| calibrated_team_factor | calibrated_independent | joint | variogram_score_p05_scaled | -0.0008 | [-0.0017, 0.0001] | 0.0979 | - |
| calibrated_team_factor | calibrated_independent | sum:stack | crps | -0.0215 | [-0.0307, -0.0117] | 0.0020 | significant |
| calibrated_role_factor | calibrated_independent | joint | variogram_score_p05_scaled | -0.0006 | [-0.0015, 0.0002] | 0.1858 | - |
| calibrated_role_factor | calibrated_independent | sum:stack | crps | -0.0935 | [-0.1247, -0.0582] | 0.0020 | significant |
| calibrated_role_factor | calibrated_team_factor | joint | variogram_score_p05_scaled | 0.0002 | [-0.0000, 0.0004] | 0.0599 | - |
| calibrated_role_factor | calibrated_team_factor | sum:stack | crps | -0.0720 | [-0.1020, -0.0435] | 0.0020 | significant |

## Gate

```
{
  "calibrated_independent": {
    "eligible_for_later_comparison": true,
    "coverage_80_in_band": true,
    "improves_on_legacy_holm": true
  },
  "calibrated_team_factor": {
    "eligible_for_later_comparison": true,
    "improves_joint_primary_holm": true,
    "worsens_joint_primary_holm": false
  },
  "calibrated_role_factor": {
    "eligible_for_later_comparison": true,
    "improves_joint_primary_holm": true,
    "worsens_joint_primary_holm": false
  },
  "role_structure_over_team_factor": true,
  "calibrated_role_correlation": "comparison_only; not a promotion candidate",
  "promotion": "none; this report is a research gate only"
}
```

## Marginal family selected per week

- legacy:50: 3 weeks
- residual:500: 41 weeks

## Dependence calibration (held-out normal scores vs model-implied, largest classes)

| roles | relation | n | held-out | team factor | role factor |
|---|---|---|---|---|---|
| RB1-WRx | opponent | 1804 | 0.030 | 0.015 | 0.020 |
| RB1-WRx | same_team | 1804 | -0.014 | 0.028 | -0.000 |
| TE1-WRx | same_team | 1804 | 0.037 | 0.028 | 0.036 |
| WR1-WRx | opponent | 1804 | -0.007 | 0.015 | 0.012 |
| WR1-WRx | same_team | 1804 | -0.015 | 0.028 | 0.014 |
| WR2-WRx | same_team | 1804 | 0.026 | 0.028 | 0.024 |
| WR3-WRx | same_team | 1804 | 0.018 | 0.028 | 0.038 |
| TE1-WRx | opponent | 1803 | 0.028 | 0.015 | 0.005 |
| WR2-WRx | opponent | 1802 | 0.020 | 0.015 | 0.012 |
| QB1-WRx | same_team | 1792 | 0.118 | 0.028 | 0.103 |
| WR3-WRx | opponent | 1792 | 0.018 | 0.015 | 0.004 |
| QB1-WRx | opponent | 1786 | -0.015 | 0.015 | 0.025 |
| RB2-WRx | opponent | 1778 | 0.004 | 0.015 | 0.002 |
| RB2-WRx | same_team | 1770 | -0.022 | 0.028 | -0.012 |
| TE2-WRx | same_team | 1765 | 0.034 | 0.028 | 0.029 |

## Caveat

Scored rows are the player-games present in the OOF panel. Players ruled inactive before kickoff are not in it, so these results describe calibration conditional on the player appearing, not production calibration when availability is itself uncertain.
