# PPR comparison preparation

The comparator is implemented and verified. The production-versus-Plan-A
comparison remains pending a compatible production prediction export.

`inputs/preparation_audit.json` records successful verification of all 16 saved
allocation/team-total input hashes, raw truth lineage, and 40,559 Plan A rows.
The database was opened read-only. Original research artifacts were preserved.

`rolling3_verification/report.json` reproduces the corrected Plan A comparison:

| Held-out season | Rows | Plan A MAE | Rolling-3 MAE |
|---|---:|---:|---:|
| 2023 | 13,404 | 2.496936 | 2.496936 |
| 2024 | 13,466 | 2.349361 | 2.505267 |
| 2025 | 13,689 | 2.265527 | 2.472045 |
| Pooled | 40,559 | 2.369837 | 2.491301 |

Pooled candidate-minus-baseline MAE is **-0.121464**. The paired calendar-week
block bootstrap interval is **[-0.154947, -0.093643]**, using 2,000 draws within
seasons and seed 42. This interval differs from the earlier individual-row
bootstrap by design. No rows were excluded. MAEs were independently recomputed
from `rolling3_verification/paired_rows.csv` before publication.

The A/B jobs were still running when this preparation completed. The saved
2025 production JSON has no complete row-level prediction export. The new
OOF capture code currently retains forecast-origin week/team with a shifted
`target_1w` actual, and does not establish an eight-component prediction target.
It must not be passed directly to this comparator by renaming fields. Required
target-game mapping, scoring compatibility, manifest fields and commands are
documented in `docs/PPR_HEAD_TO_HEAD.md` at the repository root.

No automatic follow-up was created: its scheduling request was declined.
Once the producer supplies a compatible completed export, run the comparator
with that arm as baseline and `inputs/plan_a.manifest.json` as candidate. Use a
predeclared population file if the producer covers fewer games. The comparison
must fail if requested keys, raw labels, scoring weights or provenance cutoffs
do not pass validation.
