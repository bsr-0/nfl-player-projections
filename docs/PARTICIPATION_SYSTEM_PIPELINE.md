# Participation System Pipeline

This is the main entrypoint for the new availability/opportunity system:

```bash
python scripts/run_participation_pipeline.py
```

It runs and records the system in this order:

1. Phase 1 builds and audits `canonical_player_weeks` from the repository
   database.
2. Phase 2 trains/evaluates participation and opportunity models from that
   canonical table and emits provenance-bearing OOF predictions plus a manifest.
3. Phase 3 consumes only that canonical OOF artifact and tests whether it adds
   value to the existing Phase 7 weekly-PPR evaluator on a matched population.

Artifacts live together under `data/experiments/participation_system/`:

| Stage | Artifact |
|---|---|
| 1 | `phase1_audit.csv` |
| 2 | `phase2/oof_predictions.csv`, `phase2/evaluation.json`, `phase2/phase2_manifest.json` |
| 3 | `phase3/row_predictions.csv`, `phase3/fold_metrics.csv`, `phase3/report.json` |
| system | `participation_system_manifest.json` |

Partial reruns are supported, but never guessed: `--stages phase2 phase3`
requires an existing canonical table; `--stages phase3` requires the named
canonical Phase 2 OOF artifact and its manifest. `--dry-run` prints the exact
commands without modifying the database or artifacts.

The pipeline is experimental and has no import into the Phase 7 serving path.
That boundary remains until the Phase 3 decision gate is met.
