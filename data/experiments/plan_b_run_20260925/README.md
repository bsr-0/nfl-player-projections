# Plan B first-cut share evaluation — 2026-09-25

**Result:** The player random-intercept mixed-effects baseline scored worse
than rolling-3 on all four predeclared targets. It also scored worse than the
fixed-effects ridge control on all four. These are share MAEs on capped-roster
rows, not PPR MAEs or a jointly constrained team allocation.

| Target | Held-out rows | Rolling-3 MAE | Fixed ridge MAE | Mixed-effects MAE | Mixed − rolling-3, paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Targets | 21,211 | 0.044571 | 0.046057 | 0.046695 | +0.002124 [0.001666, 0.002597] |
| Rushing attempts | 24,475 | 0.035549 | 0.037827 | 0.038508 | +0.002959 [0.002618, 0.003295] |
| Receiving yards | 21,211 | 0.057050 | 0.058329 | 0.059278 | +0.002229 [0.001665, 0.002798] |
| Rushing yards | 24,475 | 0.043298 | 0.045500 | 0.046301 | +0.003003 [0.002558, 0.003429] |

Each target uses expanding historical training for the 2023, 2024, and 2025
held-out seasons. The same player-week keys and raw share labels are used by
all three arms. Intervals resample whole calendar weeks within season, using
paired draws and player-week weighting. Positive differences mean mixed effects
is worse. Every interval above excludes zero in that direction. The mixed
model's paired intervals against ridge are also entirely positive; see the
per-target `report.json` files and `summary.json`.

All 12 seasonal fits converged with L-BFGS. No fold used the explicit mean
fallback, and no prediction used the unseen-slot mean fallback. Every fit
emitted covariance singularity/boundary warnings during optimization. A
separate refit of the 2023 rushing-attempts training fold ended with positive
player random-effect covariance (0.000409), so the warnings alone do not prove
the final fitted random effects were zero. The saved fit diagnostics retain all
warnings for review.

The mixed model clipped 208 target predictions (1.0%), 422 receiving-yard
predictions (2.0%), 3,980 rushing-attempt predictions (16.3%), and 3,659
rushing-yard predictions (15.0%) to [0, 1]. Rolling-3 null-to-zero prediction
rows were counted, not silently discarded. The mixed model's represented-team
share-mass MAE was also worse than rolling-3 for all four targets. The main
rushing weakness is on RB rows; the QB rushing subset had a small lower mixed
MAE, but it does not reverse the pooled or seasonal results.

Roster caps omitted 53,201 eligible target/receiving rows and 62,220 rushing
rows across the 2013–2025 source panel. This evaluation is limited to the
represented population. It cannot establish a full-population or PPR result,
and it does not evaluate the proposed full joint team architecture.

`verify.py` independently checked the saved input and prediction SHA-256
hashes, exact held-out keys, label equality, fold boundaries, finite and
bounded predictions, and recomputed pooled MAEs and paired point estimates
from the saved row-level `predictions.csv` files. It wrote `summary.json`.

**Decision:** Do not promote this mixed-effects baseline. If Plan B research
continues, first investigate RB rushing errors, sparse zero shares, and the
large clipping rates. A genuinely joint allocation would be a separate model
and needs its own identical-row, predeclared evaluation before any serving or
Plan A comparison claim.
