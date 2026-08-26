"""Config-driven missing-data policy registry for feature groups."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureGroupPolicy:
    group: str
    matchers: List[str]
    numeric_strategy: str = "median"
    fallback_value: float = 0.0
    warn_threshold: float = 0.05
    fail_threshold: float = 1.1
    # Exact column names this group matches but does NOT own. Their NaN is
    # filled by a dedicated step instead, and filling it here would consume
    # the missingness before that step ever sees it -- while also imputing
    # from the frame in hand rather than from train-fitted values.
    exclude: List[str] = field(default_factory=list)


@dataclass
class PolicyApplicationResult:
    rates: Dict[str, float]
    flagged_warn: List[str]
    flagged_fail: List[str]


class FeaturePolicyRegistry:
    """Policy lookup + imputation executor for grouped feature families."""

    def __init__(self, policies: Dict[str, FeatureGroupPolicy]):
        self.policies = policies

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "FeaturePolicyRegistry":
        path = config_path or Path(__file__).with_name("feature_policy_config.json")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        policies = {
            group: FeatureGroupPolicy(group=group, **cfg)
            for group, cfg in raw.get("groups", {}).items()
        }
        return cls(policies=policies)

    def resolve_group_for_feature(self, feature_name: str) -> Optional[str]:
        """Most SPECIFIC matching group, not merely the first one declared.

        Matchers are substrings, so they collide: 'epa' (pbp_advanced) is
        inside 'ngs_avg_sEPAration', and because pbp_advanced is declared
        before ngs, every NGS separation column -- including the
        `_roll3_mean` derivative the model actually consumes -- was silently
        governed by the pbp_advanced policy and median-filled, while its
        sibling ngs_avg_cushion went to ngs. Declaration order is not a
        specificity ordering, and nothing failed when it disagreed.

        Ranked by (matches as a PREFIX, matcher length), so an anchored
        family prefix like 'ngs_' beats an incidental mid-word hit like
        'epa', while genuine pbp columns ('pass_epa') are unaffected.
        """
        lname = feature_name.lower()
        best_score = None
        best_group = None
        for group, policy in self.policies.items():
            for m in policy.matchers:
                ml = m.lower()
                if ml not in lname:
                    continue
                score = (lname.startswith(ml), len(ml))
                if best_score is None or score > best_score:
                    best_score, best_group = score, group
        return best_group

    def apply(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[Iterable[str]] = None,
        fail_on_threshold: bool = False,
        context: str = "feature_builder",
    ) -> PolicyApplicationResult:
        candidates = list(columns) if columns is not None else list(df.columns)
        rates: Dict[str, float] = {}
        flagged_warn: List[str] = []
        flagged_fail: List[str] = []
        if df.empty:
            return PolicyApplicationResult(rates=rates, flagged_warn=flagged_warn, flagged_fail=flagged_fail)

        indicator_cols: Dict[str, pd.Series] = {}
        for col in candidates:
            if col not in df.columns:
                continue
            group = self.resolve_group_for_feature(col)
            if not group:
                continue
            policy = self.policies[group]
            if col in policy.exclude:
                continue
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue

            missing_rate = float(missing_mask.mean())
            rates[col] = missing_rate

            # "preserve": the group's missingness is STRUCTURAL -- the source
            # does not exist for those rows at all -- so any fill invents a
            # measurement. Rate and thresholds are still recorded so monitoring
            # keeps working; only the fill and its `_imputed` indicator are
            # skipped. Used by `ngs` (Next Gen Stats begin in 2016), where an
            # `exclude` list would not do: the columns are built dynamically
            # from whatever the ngs_* tables carry, and the model consumes the
            # `_roll3_mean` derivatives under different names.
            if policy.numeric_strategy == "preserve":
                if missing_rate > policy.warn_threshold:
                    flagged_warn.append(col)
                    logger.warning("[%s] structural missingness preserved: feature=%s rate=%.3f", context, col, missing_rate)
                if missing_rate > policy.fail_threshold:
                    flagged_fail.append(col)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if policy.numeric_strategy == "median":
                    med = df[col].median()
                    fill_value = float(med) if not np.isnan(med) else policy.fallback_value
                elif policy.numeric_strategy == "mean":
                    mean_val = df[col].mean()
                    fill_value = float(mean_val) if not np.isnan(mean_val) else policy.fallback_value
                else:
                    fill_value = policy.fallback_value
            else:
                mode = df[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "unknown"

            df[col] = df[col].fillna(fill_value)
            indicator_name = f"{col}_imputed"
            indicator_cols[indicator_name] = missing_mask.astype(np.int8)

            logger.info("[%s] imputation_rate feature=%s group=%s rate=%.3f strategy=%s", context, col, group, missing_rate, policy.numeric_strategy)
            if missing_rate > policy.warn_threshold:
                flagged_warn.append(col)
                logger.warning("[%s] missing rate exceeded warning threshold: feature=%s rate=%.3f warn_threshold=%.3f", context, col, missing_rate, policy.warn_threshold)
            if missing_rate > policy.fail_threshold:
                flagged_fail.append(col)

        if indicator_cols:
            indicator_df = pd.DataFrame(indicator_cols, index=df.index)
            overlap = [c for c in indicator_df.columns if c in df.columns]
            if overlap:
                for c in overlap:
                    df[c] = (df[c].astype(bool) | indicator_df[c].astype(bool)).astype(np.int8)
                indicator_df = indicator_df.drop(columns=overlap)
            if not indicator_df.empty:
                for c in indicator_df.columns:
                    df[c] = indicator_df[c]

        if flagged_fail and fail_on_threshold:
            raise ValueError(f"{context}: policy fail threshold exceeded for features: {sorted(flagged_fail)}")

        return PolicyApplicationResult(rates=rates, flagged_warn=flagged_warn, flagged_fail=flagged_fail)
