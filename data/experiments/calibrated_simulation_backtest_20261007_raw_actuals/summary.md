# Calibrated simulation backtest (unrestricted)

Scored 12456 player-games, 555 games, 44 weeks, seasons [2024, 2025, 2026]; 1000 draws per player. Lower is better except coverage (target 0.50 / 0.80). Every calibrated support is centred on the served prediction, so MAE differences from production_marginal are Monte Carlo noise in the draw mean.

## Modes

| mode | MAE | CRPS | cov50 | cov80 | energy | variogram | scaled variogram | stack CRPS | stack cov80 | team CRPS | game CRPS | below support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_marginal | 4.441 | 4.441 | 0.000 | 0.000 | 28.631 | 1.833 | 1.1060 | 13.238 | 0.000 | 19.406 | 30.384 | 0.0000 |
| calibrated_independent_legacy | 4.443 | 3.226 | 0.499 | 0.798 | 20.462 | 1.554 | 0.4241 | 9.669 | 0.635 | 13.844 | 21.778 | 0.0978 |
| calibrated_independent | 4.441 | 3.085 | 0.489 | 0.785 | 20.385 | 1.443 | 0.3114 | 9.484 | 0.694 | 13.857 | 21.840 | 0.0164 |
| calibrated_team_factor | 4.441 | 3.085 | 0.489 | 0.785 | 20.389 | 1.443 | 0.3108 | 9.464 | 0.705 | 13.775 | 21.519 | 0.0164 |
| calibrated_role_factor | 4.441 | 3.085 | 0.489 | 0.785 | 20.378 | 1.443 | 0.3110 | 9.393 | 0.750 | 13.755 | 21.466 | 0.0164 |
| calibrated_role_correlation | 4.441 | 3.085 | 0.489 | 0.785 | 20.397 | 1.444 | 0.3108 | 9.393 | 0.748 | 13.763 | 21.501 | 0.0164 |

## Pre-registered primary comparisons

Week-block bootstrap, Holm-corrected at alpha=0.05. Negative delta favours the candidate.

| candidate | baseline | table | metric | delta | 95% CI | p | Holm |
|---|---|---|---|---|---|---|---|
| calibrated_independent | calibrated_independent_legacy | marginal | crps | -0.1405 | [-0.1533, -0.1252] | 0.0020 | significant |
| calibrated_team_factor | calibrated_independent | joint | variogram_score_p05_scaled | -0.0006 | [-0.0014, 0.0001] | 0.1139 | - |
| calibrated_team_factor | calibrated_independent | sum:stack | crps | -0.0198 | [-0.0297, -0.0103] | 0.0020 | significant |
| calibrated_role_factor | calibrated_independent | joint | variogram_score_p05_scaled | -0.0005 | [-0.0013, 0.0003] | 0.2418 | - |
| calibrated_role_factor | calibrated_independent | sum:stack | crps | -0.0910 | [-0.1236, -0.0560] | 0.0020 | significant |
| calibrated_role_factor | calibrated_team_factor | joint | variogram_score_p05_scaled | 0.0002 | [-0.0000, 0.0004] | 0.1379 | - |
| calibrated_role_factor | calibrated_team_factor | sum:stack | crps | -0.0712 | [-0.1010, -0.0430] | 0.0020 | significant |

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

- legacy:50: 2 weeks
- residual:500: 42 weeks

## Dependence calibration (held-out normal scores vs model-implied, largest classes)

| roles | relation | n | held-out | team factor | role factor |
|---|---|---|---|---|---|
| RB1-WRx | opponent | 1804 | 0.037 | 0.016 | 0.019 |
| RB1-WRx | same_team | 1804 | -0.015 | 0.028 | 0.000 |
| TE1-WRx | same_team | 1804 | 0.043 | 0.028 | 0.037 |
| WR1-WRx | opponent | 1804 | -0.006 | 0.016 | 0.012 |
| WR1-WRx | same_team | 1804 | -0.011 | 0.028 | 0.016 |
| WR2-WRx | same_team | 1804 | 0.026 | 0.028 | 0.023 |
| WR3-WRx | same_team | 1804 | 0.019 | 0.028 | 0.039 |
| TE1-WRx | opponent | 1803 | 0.031 | 0.016 | 0.004 |
| WR2-WRx | opponent | 1802 | 0.013 | 0.016 | 0.012 |
| QB1-WRx | same_team | 1792 | 0.118 | 0.028 | 0.104 |
| WR3-WRx | opponent | 1792 | 0.020 | 0.016 | 0.005 |
| QB1-WRx | opponent | 1786 | -0.015 | 0.016 | 0.024 |
| RB2-WRx | opponent | 1778 | 0.002 | 0.016 | 0.001 |
| RB2-WRx | same_team | 1770 | -0.023 | 0.028 | -0.010 |
| TE2-WRx | same_team | 1765 | 0.041 | 0.028 | 0.028 |

## Caveat

Scored rows are the player-games present in the OOF panel. Players ruled inactive before kickoff are not in it, so these results describe calibration conditional on the player appearing, not production calibration when availability is itself uncertain.
