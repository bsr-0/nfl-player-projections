"""Paired player-week metrics and calendar-week block uncertainty for RB split."""
from __future__ import annotations
import numpy as np
import pandas as pd

ARMS = ("served", "game_sim", "rb_split")
KEY = ["player_id", "season", "week"]


def paired_week_interval(rows: pd.DataFrame, baseline: str, *, replicates: int = 5000,
                         seed: int = 20260927) -> tuple[float, float] | None:
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates required")
    if rows.empty or not np.isfinite(rows[["actual", baseline, "rb_split"]].to_numpy(float)).all():
        raise ValueError("bootstrap requires complete finite paired outcomes")
    delta = (rows.rb_split - rows.actual).abs() - (rows[baseline] - rows.actual).abs()
    block = rows.assign(delta=delta).groupby(["season", "week"]).delta.agg(["sum", "count"])
    if len(block) < 2:
        return None
    indices = np.random.default_rng(seed).integers(0, len(block), size=(replicates, len(block)))
    draws = block["sum"].to_numpy()[indices].sum(axis=1) / block["count"].to_numpy()[indices].sum(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [.025, .975]))


def evaluate_paired_panel(panel: pd.DataFrame, *, confirmation: bool = False,
                          confirmation_evidence_valid: bool = False,
                          replicates: int = 5000, seed: int = 20260927) -> dict:
    required = set(KEY + ["position", "receiving_role", "actual", *ARMS])
    if missing := required - set(panel):
        raise ValueError(f"paired panel missing {sorted(missing)}")
    if panel.empty or panel[KEY].isna().any().any() or panel.duplicated(KEY).any():
        raise ValueError("eligible cohort must be nonempty and have unique nonnull keys")
    if not panel.position.isin(["QB", "RB", "WR", "TE"]).all():
        raise ValueError("unknown overall position")
    if not np.isfinite(panel[list(ARMS)].to_numpy(float)).all():
        raise ValueError("missing/nonfinite arm predictions; cannot silently reduce cohort")
    if not np.isfinite(panel.actual.dropna().to_numpy(float)).all():
        raise ValueError("nonfinite actual")
    if not panel.loc[panel.position.eq("RB"), "receiving_role"].isin(["low", "mixed", "high"]).all():
        raise ValueError("RB receiving slices must be predeclared low/mixed/high")
    observed = panel.loc[panel.actual.notna()].copy()
    slices = {"overall": panel, **{p: panel.loc[panel.position.eq(p)] for p in ["QB", "RB", "WR", "TE"]},
              **{f"RB_{r}": panel.loc[panel.position.eq("RB") & panel.receiving_role.eq(r)] for r in ["low", "mixed", "high"]}}
    table = []
    for label, eligible in slices.items():
        data = eligible.loc[eligible.actual.notna()]
        for arm in ARMS:
            error = data[arm] - data.actual
            table.append({"slice": label, "arm": arm, "eligible_n": len(eligible), "observed_n": len(data),
                          "missing_n": len(eligible)-len(data), "weeks": data[["season", "week"]].drop_duplicates().shape[0],
                          "mae": float(error.abs().mean()) if len(data) else None,
                          "rmse": float(np.sqrt(np.mean(error**2))) if len(data) else None,
                          "metric_scope": "full_cohort" if len(data)==len(eligible) else "observed_rows_only_not_primary"})
    comparisons = []
    for label, eligible in slices.items():
        data = eligible.loc[eligible.actual.notna()]
        for baseline in ARMS[:2]:
            base = float((data[baseline]-data.actual).abs().mean()) if len(data) else None
            new = float((data.rb_split-data.actual).abs().mean()) if len(data) else None
            ci = paired_week_interval(data, baseline, replicates=replicates, seed=seed) if len(data) else None
            comparisons.append({"slice": label, "baseline": baseline, "eligible_n": len(eligible),
                                "observed_n": len(data), "delta_mae": new-base if len(data) else None,
                                "relative_improvement_pct": 100*(base-new)/base if base else None,
                                "ci95": ci, "scope": "full_cohort" if len(data)==len(eligible) else "observed_rows_only_not_primary"})
    complete = len(panel) == len(observed)
    weeks = panel[["season", "week"]].drop_duplicates().shape[0]
    ready = confirmation and confirmation_evidence_valid and complete and weeks >= 8 and panel.position.eq("RB").any()
    gates = None
    if ready:
        rb = [r for r in comparisons if r["slice"] == "RB"]
        overall = [r for r in comparisons if r["slice"] == "overall"]
        gates = {"rb_five_percent_each": all(r["relative_improvement_pct"] is not None and r["relative_improvement_pct"] >= 5 for r in rb),
                 "rb_ci_below_zero_each": all(r["ci95"] is not None and r["ci95"][1] < 0 for r in rb),
                 "overall_no_worse_each": all(r["delta_mae"] <= 0 for r in overall)}
    return {"decision": ("PASS" if all(gates.values()) else "FAIL") if gates is not None else "INCONCLUSIVE",
            "confirmation_status": "evaluated" if ready else "BLOCKED", "acceptance_gates": gates,
            "eligible_rows": len(panel), "observed_rows": len(observed), "missing_outcomes": len(panel)-len(observed),
            "calendar_week_blocks": weeks, "primary_metrics_available": complete,
            "metrics": table, "comparisons": comparisons,
            "bootstrap": {"replicates": replicates, "seed": seed, "weighting": "equal player-week; resample whole season-week blocks",
                          "minimum_blocks_for_interval": 2, "minimum_confirmation_blocks": 8}}
