#!/usr/bin/env python3
"""Apply the selection rule's gates to 2025 (Amendment 1 of docs/PRODUCTION_SELECTION_RULE.md).

Evaluation set: the frozen matched set intersected with the 2025 pre-kickoff
lists (5,824 rows). Arms:
    actual               raw ten-component full PPR of the target game
    incumbent            live blend (scripts/export_live_blend_fold.py)
    served_unblended     the held-out served fold (candidate)
    plan_a               corrected Plan A, pre-kickoff export (candidate)
    rolling3             pre-kickoff export (G9 and the neutral tiering)
Every arm must cover every row, or nothing is evaluated. Exports are pinned by
the sha256s in their manifests and in Amendment 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.paired_ppr_comparison import KEY, file_sha256
from src.evaluation.selection_gates import Columns, evaluate

E = ROOT / "data/experiments"
INPUTS = {
    "paired": E / "full_ppr_head_to_head_20260930/comparison_2025/paired_rows.csv",
    "incumbent": E / "selection_incumbent_live_blend_2025/predictions.csv",
    "plan_a": E / "selection_candidates_2025/plan_a_prekickoff/predictions.csv",
    "rolling3": E / "selection_candidates_2025/rolling3_prekickoff/predictions.csv",
}
PINNED = {  # Amendment 1
    "incumbent": "63ad88afe4906465e627232cb02740bb23e085f8a1f556dfbef81fd4b9c31242",
    "plan_a": "b5585a9fc9c081f5722f1215af7b77aaaa3d4f56e84cc6adc2950f494042d8f1",
    "rolling3": "c1bded4f4675807365199f8b91bbf77ef0327a2369f8828d0ce7d8e146df5a6f",
}
K3 = ["player_id", "season", "week"]


def frame() -> tuple[pd.DataFrame, dict]:
    for name, digest in PINNED.items():
        if file_sha256(INPUTS[name]) != digest:
            raise SystemExit(f"{name} predictions differ from the pinned export")
    paired = pd.read_csv(INPUTS["paired"], dtype={"player_id": str})
    df = paired[KEY + ["actual_full_ppr", "served_prediction"]].rename(
        columns={"actual_full_ppr": "actual", "served_prediction": "served_unblended"})
    inc = pd.read_csv(INPUTS["incumbent"], dtype={"player_id": str})[KEY + ["predicted_ppr"]]
    df = df.merge(inc.rename(columns={"predicted_ppr": "incumbent"}), on=KEY, how="left", validate="one_to_one")
    for arm in ("plan_a", "rolling3"):
        x = pd.read_csv(INPUTS[arm], dtype={"player_id": str})[K3 + ["predicted_ppr"]]
        df = df.merge(x.rename(columns={"predicted_ppr": arm}), on=K3, how="left", validate="one_to_one")
    before = len(df)
    listed = df.plan_a.notna()
    if not listed.equals(df.rolling3.notna()):
        raise SystemExit("Plan A and rolling-3 cover different player-weeks")
    df = df[listed].reset_index(drop=True)
    arms = ["actual", "incumbent", "served_unblended", "plan_a", "rolling3"]
    if df[arms].isna().any().any():
        raise SystemExit(f"missing forecasts: {df[arms].isna().sum().to_dict()}")
    return df, {"matched_rows": before, "evaluated_rows": len(df), "dropped_not_on_list": before - len(df)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df, coverage = frame()
    if coverage["evaluated_rows"] != 5824:
        raise SystemExit(f"expected the amended 5,824-row set, got {coverage['evaluated_rows']}")
    report = evaluate(df, ["plan_a", "served_unblended"], cols=Columns(), B=a.B, seed=a.seed)
    report["coverage"] = coverage
    report["inputs_sha256"] = {k: file_sha256(v) for k, v in INPUTS.items()}
    report["script_sha256"] = file_sha256(Path(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(coverage))
    for c, r in report["candidates"].items():
        g1 = r["G1"]
        print(f"{c}: MAE {r['mae']:.3f} vs incumbent {r['incumbent_mae']:.3f} | G1 diff {g1['point_difference']:+.3f} "
              f"UB {g1['upper_bound_95']:+.3f} p {g1['p_one_sided']:.4f} holm {g1['holm_rejects']} pass {g1['passes']} | "
              f"G9 {r['G9']['passes']} | guardrail failures {r['guardrail_failures']} | stage {r['stage_passes']}")


if __name__ == "__main__":
    main()
