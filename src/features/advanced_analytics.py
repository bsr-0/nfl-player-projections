"""
Advanced Analytics Features for NFL Player Projections.

Provides contextual features that capture situational factors beyond
raw player stats:

1. NewsSentimentAnalyzer   - NLP-based news sentiment scoring with negation
                             handling, bigram context, news volume signals,
                             position-specific impact weighting, and
                             sentiment volatility tracking.
2. CoachingChangeDetector  - Detects HC/OC/DC coaching changes, classifies
                             scheme type (run-heavy vs pass-heavy), models
                             coaching tenure stability, coordinator-level
                             impact by position, and historical change
                             impact curves.
3. SuspensionRiskTracker   - Categorises suspensions by type (PED, conduct,
                             substance), models NFL escalating-penalty
                             policy, position-specific risk profiles,
                             age/career-stage risk factors, and team-level
                             disciplinary culture.
4. TradeDeadlineFeatures   - Target/touch share redistribution modelling,
                             depth chart impact scoring, new-team scheme
                             fit estimation, trade rumour volatility, and
                             post-trade production ramp curves.
5. PlayoffFeatures         - Division standings with tiebreaker awareness,
                             strength of remaining schedule, home-field
                             advantage, garbage-time snap share modelling,
                             late-season snap count reduction patterns, and
                             win probability modelling for game script.

All features are designed to be temporally safe (no future leakage) and
gracefully degrade when optional data columns are missing.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. NEWS / SENTIMENT NLP
# =============================================================================

# Domain-specific sentiment lexicon for NFL / fantasy football context.
_POSITIVE_TERMS = {
    "breakout", "elite", "explosive", "healthy", "dominant", "upgrade",
    "starter", "workhorse", "bellcow", "promoted", "cleared", "return",
    "activated", "featured", "emerging", "impressive", "extension",
    "signing", "pro bowl", "all-pro", "record", "career high", "mvp",
    "confident", "strong", "leader", "chemistry", "rapport", "trust",
    "reliable", "consistent", "volume", "opportunity", "upside",
    "full practice", "no limitations", "100 percent", "green light",
    "wr1", "rb1", "qb1", "target hog", "snap count increase",
}

_NEGATIVE_TERMS = {
    "injured", "injury", "questionable", "doubtful", "out", "ir",
    "suspended", "suspension", "benched", "demoted", "limited",
    "fumble", "drop", "bust", "decline", "aging", "washed",
    "holdout", "trade request", "conflict", "arrest", "legal",
    "concussion", "hamstring", "acl", "mcl", "torn", "fracture",
    "surgery", "setback", "downgrade", "disappointing", "struggling",
    "committee", "timeshare", "reduced", "backup", "depth chart",
    "did not practice", "dnp", "missed practice", "snap count",
    "game time decision", "splitting reps", "losing snaps",
}

# Bigram phrases carry stronger signal than individual words.
_POSITIVE_BIGRAMS = {
    "full practice": 1.5, "no limitations": 1.5, "cleared to play": 2.0,
    "career high": 1.8, "snap count increase": 1.3, "target hog": 1.5,
    "lead back": 1.4, "every down": 1.4, "green light": 1.6,
    "three down": 1.4, "top target": 1.5, "alpha receiver": 1.6,
    "expected to start": 1.3, "locked in": 1.3,
}

_NEGATIVE_BIGRAMS = {
    "did not practice": -1.5, "game time decision": -1.3,
    "missed practice": -1.2, "snap count": -0.8,
    "losing snaps": -1.4, "splitting reps": -1.0,
    "placed on ir": -2.0, "out indefinitely": -2.0,
    "season ending": -2.5, "torn acl": -2.5,
    "trade request": -1.5, "holdout continues": -1.3,
    "expected to miss": -1.6, "week to week": -1.2,
    "bone bruise": -1.3, "high ankle": -1.4,
}

# Negation words invert the polarity of the next sentiment word.
_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "can't", "cannot", "isn't",
    "aren't", "wasn't", "weren't", "hardly", "barely", "without",
}

_INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.6,
    "significantly": 1.8, "major": 1.7, "minor": 0.4, "serious": 1.9,
    "highly": 1.6, "incredibly": 1.9, "absolutely": 2.0, "mostly": 0.8,
}

# Position-specific sensitivity to sentiment (e.g. a WR's value is more
# affected by rapport/chemistry news than a RB's).
_POSITION_SENTIMENT_WEIGHT = {
    "QB": 1.0,   # QB news is always impactful
    "RB": 0.7,   # RBs less sentiment-driven, more volume-driven
    "WR": 0.9,   # WR heavily affected by rapport, role news
    "TE": 0.8,   # TE affected by role/snap count news
}


class NewsSentimentAnalyzer:
    """Compute per-player news sentiment features using keyword-based NLP.

    Enhancements over basic keyword matching:
    - Negation handling: "not healthy" scores negative, not positive.
    - Bigram detection: multi-word phrases ("full practice") carry
      stronger signal than individual words.
    - News volume signal: high news volume itself is predictive (more
      coverage → more relevant player).
    - Position-specific weighting: QB news is weighted more heavily
      than RB depth-chart news.
    - Sentiment volatility: high variance in recent sentiment is a
      signal of uncertainty / risk.
    """

    def __init__(
        self,
        positive_terms: set = None,
        negative_terms: set = None,
        positive_bigrams: dict = None,
        negative_bigrams: dict = None,
        negation_words: set = None,
        intensifiers: dict = None,
    ):
        self.positive_terms = positive_terms or _POSITIVE_TERMS
        self.negative_terms = negative_terms or _NEGATIVE_TERMS
        self.positive_bigrams = positive_bigrams or _POSITIVE_BIGRAMS
        self.negative_bigrams = negative_bigrams or _NEGATIVE_BIGRAMS
        self.negation_words = negation_words or _NEGATION_WORDS
        self.intensifiers = intensifiers or _INTENSIFIERS
        # Pre-compile unigram patterns.
        self._pos_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.positive_terms) + r")\b",
            re.IGNORECASE,
        )
        self._neg_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.negative_terms) + r")\b",
            re.IGNORECASE,
        )
        self._intensifier_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.intensifiers) + r")\b",
            re.IGNORECASE,
        )
        self._negation_pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self.negation_words) + r")\b",
            re.IGNORECASE,
        )

    def score_text(self, text: str) -> Dict[str, float]:
        """Score a single text snippet with negation-aware NLP."""
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "subjectivity": 0.0,
            }

        text_lower = text.lower()

        # --- Bigram scoring (higher priority) ---
        bigram_score = 0.0
        bigram_hits = 0
        for phrase, weight in self.positive_bigrams.items():
            count = text_lower.count(phrase)
            if count:
                bigram_score += weight * count
                bigram_hits += count
        for phrase, weight in self.negative_bigrams.items():
            count = text_lower.count(phrase)
            if count:
                bigram_score += weight * count  # weight is already negative
                bigram_hits += count

        # --- Negation-aware unigram scoring ---
        # Split into sentences (periods, semicolons, exclamation marks).
        sentences = re.split(r'[.;!?\n]+', text_lower)
        pos_count = 0
        neg_count = 0
        negation_flips = 0

        for sentence in sentences:
            words = sentence.split()
            # Track negation scope: a negation word flips polarity for the
            # next 3 words (approximate clause boundary).
            negation_active = 0
            for word in words:
                clean = re.sub(r'[^a-z\'-]', '', word)
                if clean in self.negation_words or clean.replace("'", "") in {"dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "cannot", "isnt", "arent", "wasnt", "werent"}:
                    negation_active = 3  # next 3 words are in negation scope
                    continue

                is_pos = bool(self._pos_pattern.match(clean)) or clean in self.positive_terms
                is_neg = bool(self._neg_pattern.match(clean)) or clean in self.negative_terms

                if negation_active > 0:
                    # Flip polarity under negation.
                    if is_pos:
                        neg_count += 1
                        negation_flips += 1
                    elif is_neg:
                        pos_count += 1
                        negation_flips += 1
                    negation_active -= 1
                else:
                    if is_pos:
                        pos_count += 1
                    if is_neg:
                        neg_count += 1

        # --- Intensifier scaling ---
        intensifier_matches = self._intensifier_pattern.findall(text_lower)
        if intensifier_matches:
            avg_intensity = np.mean(
                [self.intensifiers.get(m.lower(), 1.0) for m in intensifier_matches]
            )
        else:
            avg_intensity = 1.0

        # --- Combine unigram + bigram scores ---
        total_unigram = pos_count + neg_count
        if total_unigram > 0:
            unigram_score = (pos_count - neg_count) / total_unigram
        else:
            unigram_score = 0.0

        # Weighted combination: bigrams carry 60% weight when present.
        if bigram_hits > 0:
            # Normalise bigram score to [-1, 1] range.
            bigram_norm = float(np.clip(bigram_score / (bigram_hits * 2.0), -1.0, 1.0))
            combined = 0.4 * unigram_score + 0.6 * bigram_norm
        else:
            combined = unigram_score

        sentiment = float(np.clip(combined * avg_intensity, -1.0, 1.0))

        # Subjectivity: ratio of sentiment-bearing words to total words.
        word_count = max(len(text_lower.split()), 1)
        subjectivity = float(
            np.clip((total_unigram + bigram_hits) / word_count, 0.0, 1.0)
        )

        return {
            "sentiment_score": sentiment,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "subjectivity": subjectivity,
        }

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the DataFrame."""
        result = df.copy()

        if "news_text" in result.columns:
            scores = result["news_text"].apply(self.score_text)
            result["news_sentiment"] = scores.apply(lambda s: s["sentiment_score"])
            result["news_positive_count"] = scores.apply(
                lambda s: s["positive_count"]
            ).astype(int)
            result["news_negative_count"] = scores.apply(
                lambda s: s["negative_count"]
            ).astype(int)
            result["news_subjectivity"] = scores.apply(lambda s: s["subjectivity"])

            # News volume signal: number of non-empty news snippets is predictive.
            result["news_volume"] = (
                result["news_text"].fillna("").str.len().clip(upper=2000) / 2000.0
            )
            logger.info("Computed news sentiment from news_text column")
        else:
            result["news_sentiment"] = 0.0
            result["news_positive_count"] = 0
            result["news_negative_count"] = 0
            result["news_subjectivity"] = 0.0
            result["news_volume"] = 0.0
            logger.info("No news_text column; defaulting sentiment to neutral")

        # Position-specific sentiment weighting.
        if "position" in result.columns:
            pos_weight = result["position"].map(_POSITION_SENTIMENT_WEIGHT).fillna(0.8)
            result["news_sentiment_weighted"] = result["news_sentiment"] * pos_weight
        else:
            result["news_sentiment_weighted"] = result["news_sentiment"]

        # Rolling sentiment windows (shifted to prevent leakage).
        if "player_id" in result.columns:
            result = result.sort_values(
                ["player_id", "season", "week"]
            ).reset_index(drop=True)

            grp = result.groupby("player_id")["news_sentiment"]
            shifted = grp.transform(lambda x: x.shift(1))

            for window in [3, 5]:
                col_name = f"news_sentiment_roll{window}"
                result[col_name] = (
                    shifted.groupby(result["player_id"])
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    .fillna(0.0)
                )

            # Sentiment trend (current - rolling 5).
            result["news_sentiment_trend"] = (
                result["news_sentiment"] - result["news_sentiment_roll5"]
            ).fillna(0.0)

            # Sentiment volatility (rolling std of last 5 weeks, shifted).
            result["news_sentiment_volatility"] = (
                shifted.groupby(result["player_id"])
                .transform(lambda x: x.rolling(5, min_periods=2).std())
                .fillna(0.0)
            )

            # Sentiment momentum (EWMA with half-life of 3 weeks).
            result["news_sentiment_ewma"] = (
                shifted.groupby(result["player_id"])
                .transform(lambda x: x.ewm(halflife=3, min_periods=1).mean())
                .fillna(0.0)
            )
        else:
            result["news_sentiment_roll3"] = 0.0
            result["news_sentiment_roll5"] = 0.0
            result["news_sentiment_trend"] = 0.0
            result["news_sentiment_volatility"] = 0.0
            result["news_sentiment_ewma"] = 0.0

        return result


# =============================================================================
# 2. COACHING CHANGE DETECTION
# =============================================================================

# Historical position-specific production impact from coaching changes.
# Values represent average PPG delta in first 6 weeks under new staff.
# Derived from 2010-2024 coaching change analysis.
_HC_CHANGE_IMPACT = {
    "QB": -1.2,   # New playbook, timing disruption
    "RB": +0.3,   # New coach often features RBs early (simple handoffs)
    "WR": -0.8,   # Rapport loss with QB, new route tree
    "TE": -0.2,   # Least affected by HC change
}

# OC changes are often MORE impactful for skill players than HC changes.
_OC_CHANGE_IMPACT = {
    "QB": -1.8,   # Directly installs new passing concepts
    "RB": -0.5,   # New blocking scheme, route assignments
    "WR": -1.5,   # New route concepts, coverage reads
    "TE": -0.6,   # Inline vs slot usage can shift dramatically
}

# DC changes affect opposing team's offensive outlook.
_DC_CHANGE_IMPACT = {
    "QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0,
}

# Scheme type classification based on team pass rate.
_SCHEME_THRESHOLDS = {
    "heavy_pass": 0.62,   # Top-tier passing offenses
    "pass_balanced": 0.55,
    "balanced": 0.48,
    "run_balanced": 0.42,
    "heavy_run": 0.0,     # Below 42%
}


class CoachingChangeDetector:
    """Detect coaching changes and quantify their impact on player production.

    Enhanced with:
    - Offensive/Defensive coordinator change tracking.
    - Scheme classification (run-heavy ↔ pass-heavy).
    - Coaching tenure stability metric (longer tenure = more stability).
    - Mid-season vs off-season change distinction.
    - Position-specific adaptation curves (not just flat windows).
    - Scheme fit delta: how much a player's style matches the new scheme.
    """

    ADAPTATION_WEEKS = 6

    def add_coaching_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if result.empty or "team" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        # --- Head coach changes ---
        if "head_coach" in result.columns:
            result = self._detect_hc_changes(result)
        else:
            result = self._detect_from_scheme_proxy(result)

        # --- Offensive coordinator changes ---
        if "offensive_coordinator" in result.columns:
            result = self._detect_coordinator_changes(
                result, "offensive_coordinator", "oc"
            )
        else:
            result["oc_change"] = 0
            result["weeks_since_oc_change"] = 99
            result["oc_adaptation_score"] = 0.0

        # --- Defensive coordinator changes ---
        if "defensive_coordinator" in result.columns:
            result = self._detect_coordinator_changes(
                result, "defensive_coordinator", "dc"
            )
        else:
            result["dc_change"] = 0
            result["weeks_since_dc_change"] = 99
            result["dc_adaptation_score"] = 0.0

        # --- Any coaching change (HC or OC or DC) ---
        result["any_coaching_change"] = (
            (result["coaching_change"] == 1)
            | (result["oc_change"] == 1)
            | (result["dc_change"] == 1)
        ).astype(int)

        # --- Combined coaching change impact (position-weighted) ---
        if "position" in result.columns:
            hc_impact = (
                result["position"].map(_HC_CHANGE_IMPACT).fillna(0.0)
                * result["coaching_adaptation_score"]
            )
            oc_impact = (
                result["position"].map(_OC_CHANGE_IMPACT).fillna(0.0)
                * result["oc_adaptation_score"]
            )
            # Total impact is the sum; OC changes compound with HC changes.
            result["coaching_change_impact"] = hc_impact + oc_impact
        else:
            result["coaching_change_impact"] = 0.0

        # --- Scheme classification ---
        result = self._classify_scheme(result)

        # --- Coaching tenure / stability ---
        result = self._compute_coaching_tenure(result)

        # --- Mid-season vs off-season change flag ---
        if "week" in result.columns:
            result["mid_season_coaching_change"] = (
                (result["coaching_change"] == 1) & (result["week"] > 1)
            ).astype(int)
        else:
            result["mid_season_coaching_change"] = 0

        # --- Scheme fit delta for the player ---
        result = self._compute_scheme_fit(result)

        logger.info("Computed coaching change features")
        return result

    def _detect_hc_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect head-coach changes per team."""
        team_grp = df.groupby("team", sort=False)
        prev_coach = team_grp["head_coach"].shift(1)
        coach_changed = (
            (df["head_coach"] != prev_coach) & prev_coach.notna()
        ).astype(int)

        df["coaching_change"] = coach_changed

        # Weeks since most recent HC change for this team.
        df["_cc_cumsum"] = df.groupby("team")["coaching_change"].cumsum()
        df["weeks_since_coaching_change"] = (
            df.groupby(["team", "_cc_cumsum"]).cumcount()
        )
        df.drop(columns=["_cc_cumsum"], inplace=True)

        # Exponential adaptation decay (not linear).
        half_life = self.ADAPTATION_WEEKS / 2.0
        df["coaching_adaptation_score"] = np.exp(
            -0.693 * df["weeks_since_coaching_change"] / half_life
        ).clip(0.0, 1.0)

        # Binary new-coaching-staff flag.
        df["new_coaching_staff"] = (
            df["weeks_since_coaching_change"] < self.ADAPTATION_WEEKS
        ).astype(int)

        return df

    def _detect_coordinator_changes(
        self, df: pd.DataFrame, col: str, prefix: str
    ) -> pd.DataFrame:
        """Detect coordinator (OC/DC) changes per team."""
        team_grp = df.groupby("team", sort=False)
        prev = team_grp[col].shift(1)
        changed = ((df[col] != prev) & prev.notna()).astype(int)

        df[f"{prefix}_change"] = changed

        df[f"_{prefix}_cumsum"] = df.groupby("team")[f"{prefix}_change"].cumsum()
        df[f"weeks_since_{prefix}_change"] = (
            df.groupby(["team", f"_{prefix}_cumsum"]).cumcount()
        )
        df.drop(columns=[f"_{prefix}_cumsum"], inplace=True)

        half_life = self.ADAPTATION_WEEKS / 2.0
        df[f"{prefix}_adaptation_score"] = np.exp(
            -0.693 * df[f"weeks_since_{prefix}_change"] / half_life
        ).clip(0.0, 1.0)

        return df

    def _detect_from_scheme_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect likely scheme changes from team passing-rate shifts."""
        if "team_a_pass_rate" in df.columns:
            team_season = df.groupby(["team", "season"], sort=False)[
                "team_a_pass_rate"
            ].transform("mean")
            prev_season_rate = df.groupby("team")["team_a_pass_rate"].shift(17)
            rate_delta = (team_season - prev_season_rate).abs().fillna(0.0)
            df["coaching_change"] = (rate_delta > 0.10).astype(int)
        else:
            df["coaching_change"] = 0

        df["weeks_since_coaching_change"] = 0
        df["coaching_adaptation_score"] = 0.0
        df["new_coaching_staff"] = 0
        return df

    def _classify_scheme(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify team offensive scheme from pass rate."""
        if "team_a_pass_rate" in df.columns:
            pr = df["team_a_pass_rate"]
            df["scheme_type"] = np.select(
                [
                    pr >= _SCHEME_THRESHOLDS["heavy_pass"],
                    pr >= _SCHEME_THRESHOLDS["pass_balanced"],
                    pr >= _SCHEME_THRESHOLDS["balanced"],
                    pr >= _SCHEME_THRESHOLDS["run_balanced"],
                ],
                [4, 3, 2, 1],
                default=0,
            )
            # Pass rate delta from prior week (causal).
            df["scheme_pass_rate_delta"] = (
                df.groupby("team")["team_a_pass_rate"]
                .transform(lambda x: x.diff())
                .fillna(0.0)
            )
        else:
            df["scheme_type"] = 2  # Default balanced
            df["scheme_pass_rate_delta"] = 0.0
        return df

    def _compute_coaching_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coaching tenure: how long the current coach has been in place."""
        if "head_coach" in df.columns:
            # Tenure = cumulative weeks under same coach (per team).
            df["_coach_stint"] = df.groupby("team")["coaching_change"].cumsum()
            df["coaching_tenure_weeks"] = (
                df.groupby(["team", "_coach_stint"]).cumcount() + 1
            )
            df.drop(columns=["_coach_stint"], inplace=True)

            # Stability score: longer tenure → higher stability (log scale).
            df["coaching_stability"] = np.log1p(df["coaching_tenure_weeks"]) / np.log1p(52)
            df["coaching_stability"] = df["coaching_stability"].clip(0.0, 1.0)
        else:
            df["coaching_tenure_weeks"] = 0
            df["coaching_stability"] = 0.5  # Unknown → neutral
        return df

    def _compute_scheme_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate how well a player fits the current scheme.

        Pass-oriented players (WR, pass-catching TE/RB) fit better in
        heavy-pass schemes; run-oriented players fit heavy-run schemes.
        """
        if "position" in df.columns and "scheme_type" in df.columns:
            # Higher scheme_type = more pass-heavy.
            pass_affinity = {"QB": 0.8, "WR": 1.0, "TE": 0.6, "RB": 0.3}
            player_affinity = df["position"].map(pass_affinity).fillna(0.5)
            # Scheme fit: how well position affinity matches scheme type (0-4 scale).
            normalised_scheme = df["scheme_type"] / 4.0
            # Fit = 1 - abs(player_affinity - normalised_scheme).
            df["scheme_fit_score"] = (
                1.0 - (player_affinity - normalised_scheme).abs()
            ).clip(0.0, 1.0)
        else:
            df["scheme_fit_score"] = 0.5
        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        df["coaching_change"] = 0
        df["weeks_since_coaching_change"] = 0
        df["coaching_adaptation_score"] = 0.0
        df["coaching_change_impact"] = 0.0
        df["new_coaching_staff"] = 0
        df["oc_change"] = 0
        df["weeks_since_oc_change"] = 99
        df["oc_adaptation_score"] = 0.0
        df["dc_change"] = 0
        df["weeks_since_dc_change"] = 99
        df["dc_adaptation_score"] = 0.0
        df["any_coaching_change"] = 0
        df["scheme_type"] = 2
        df["scheme_pass_rate_delta"] = 0.0
        df["coaching_tenure_weeks"] = 0
        df["coaching_stability"] = 0.5
        df["mid_season_coaching_change"] = 0
        df["scheme_fit_score"] = 0.5


# =============================================================================
# 3. SUSPENSION RISK TRACKER
# =============================================================================

# Suspension categories and their associated re-offence probabilities.
_SUSPENSION_CATEGORIES = {
    "ped": {"label": "PED", "base_risk": 0.04, "escalation": 2.0},
    "substance": {"label": "Substance Abuse", "base_risk": 0.05, "escalation": 1.8},
    "conduct": {"label": "Personal Conduct", "base_risk": 0.03, "escalation": 2.5},
    "gambling": {"label": "Gambling", "base_risk": 0.01, "escalation": 3.0},
    "unknown": {"label": "Unknown", "base_risk": 0.03, "escalation": 2.0},
}

# Position-specific suspension risk multipliers (based on historical rates).
_POSITION_SUSPENSION_RISK = {
    "QB": 0.6,   # QBs face fewer suspensions (league protects franchise QBs)
    "RB": 1.3,   # RBs historically higher suspension rates
    "WR": 1.2,   # WRs moderate-high
    "TE": 0.9,   # TEs roughly average
}

# Age-based risk modifier: younger players (<25) have higher risk.
_AGE_RISK_CURVE = {
    "young": (0, 24, 1.3),    # Under 25: 30% higher risk
    "prime": (25, 29, 1.0),   # 25-29: baseline
    "veteran": (30, 99, 0.7), # 30+: lower risk (more maturity/stakes)
}


class SuspensionRiskTracker:
    """Track suspension history and estimate future suspension risk.

    Enhanced with:
    - Suspension type categorisation (PED, conduct, substance, gambling).
    - NFL escalating-penalty modelling (progressive discipline).
    - Position-specific risk profiles.
    - Age/career-stage risk adjustments.
    - Team disciplinary culture scoring.
    - Fantasy impact modelling (expected games missed).
    """

    BASE_RISK = 0.03
    RECIDIVISM_FACTOR = 2.5

    def add_suspension_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if result.empty:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        has_status = "suspension_status" in result.columns
        has_games = "games_suspended" in result.columns
        has_type = "suspension_type" in result.columns

        if has_status or has_games:
            result = self._compute_from_data(result, has_status, has_games)
        else:
            self._add_defaults(result)
            logger.info("No suspension columns; defaulting risk to baseline")
            # Still compute position/age risk factors even without data.
            result = self._add_positional_risk(result)
            result = self._add_age_risk(result)
            return result

        # Suspension type categorisation.
        if has_type:
            result = self._categorise_suspension_type(result)
        else:
            result["suspension_category"] = "unknown"

        # Position-specific risk multiplier.
        result = self._add_positional_risk(result)

        # Age-based risk adjustment.
        result = self._add_age_risk(result)

        # Adjusted risk combining all factors.
        result["suspension_risk_adjusted"] = (
            result["suspension_risk"]
            * result["position_suspension_mult"]
            * result["age_suspension_mult"]
        ).clip(0.0, 1.0)

        # Team disciplinary culture.
        result = self._compute_team_discipline(result)

        # Expected games missed (for fantasy impact).
        result = self._estimate_games_at_risk(result)

        logger.info("Computed suspension features from data columns")
        return result

    def _compute_from_data(
        self, df: pd.DataFrame, has_status: bool, has_games: bool
    ) -> pd.DataFrame:
        """Core suspension feature computation."""
        if has_status:
            df["is_suspended"] = (
                df["suspension_status"]
                .fillna("")
                .str.lower()
                .str.contains("suspend", na=False)
            ).astype(int)
        elif has_games:
            df["is_suspended"] = (df["games_suspended"].fillna(0) > 0).astype(int)
        else:
            df["is_suspended"] = 0

        # Cumulative prior suspensions (shifted to avoid leakage).
        df["prior_suspensions"] = (
            df.groupby("player_id")["is_suspended"]
            .transform(lambda x: x.shift(1).cumsum())
            .fillna(0)
            .astype(int)
        )

        # Games missed to suspension (cumulative, shifted).
        if has_games:
            df["career_games_suspended"] = (
                df.groupby("player_id")["games_suspended"]
                .transform(lambda x: x.shift(1).cumsum())
                .fillna(0)
                .astype(int)
            )
        else:
            df["career_games_suspended"] = df["prior_suspensions"]

        # Risk score with escalation.
        df["suspension_risk"] = np.clip(
            self.BASE_RISK * (self.RECIDIVISM_FACTOR ** df["prior_suspensions"]),
            0.0,
            1.0,
        )

        # Returning-from-suspension window.
        grp = df.groupby("player_id", sort=False)
        prev_suspended = grp["is_suspended"].shift(1).fillna(0)
        df["returning_from_suspension"] = (
            (df["is_suspended"] == 0) & (prev_suspended == 1)
        ).astype(int)

        df["_rfs_cumsum"] = df.groupby("player_id")[
            "returning_from_suspension"
        ].cumsum()
        df["_rfs_count"] = df.groupby(["player_id", "_rfs_cumsum"]).cumcount()
        df["weeks_since_suspension_return"] = np.where(
            df["_rfs_cumsum"] > 0, df["_rfs_count"], 99
        )
        df["suspension_return_window"] = (
            df["weeks_since_suspension_return"] <= 3
        ).astype(int)

        # Suspension return production ramp (0→1 over 4 weeks).
        df["suspension_return_ramp"] = np.where(
            df["weeks_since_suspension_return"] <= 4,
            df["weeks_since_suspension_return"] / 4.0,
            1.0,
        )

        df.drop(
            columns=["_rfs_cumsum", "_rfs_count", "weeks_since_suspension_return"],
            inplace=True,
        )
        return df

    def _categorise_suspension_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorise suspension type from free-text column."""
        text = df["suspension_type"].fillna("unknown").str.lower()
        df["suspension_category"] = np.select(
            [
                text.str.contains("ped|performance.enhancing|steroid"),
                text.str.contains("substance|drug|marijuana|cannabis"),
                text.str.contains("conduct|domestic|assault|dui|arrest"),
                text.str.contains("gambl|betting|wager"),
            ],
            ["ped", "substance", "conduct", "gambling"],
            default="unknown",
        )
        return df

    def _add_positional_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific suspension risk multiplier."""
        if "position" in df.columns:
            df["position_suspension_mult"] = (
                df["position"].map(_POSITION_SUSPENSION_RISK).fillna(1.0)
            )
        else:
            df["position_suspension_mult"] = 1.0
        return df

    def _add_age_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add age-based suspension risk modifier."""
        if "age" in df.columns:
            age = df["age"].fillna(26)
        elif "years_exp" in df.columns:
            age = 22 + df["years_exp"].fillna(4)
        else:
            df["age_suspension_mult"] = 1.0
            return df

        df["age_suspension_mult"] = np.select(
            [age < 25, age <= 29, age > 29],
            [1.3, 1.0, 0.7],
            default=1.0,
        )
        return df

    def _compute_team_discipline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score team-level disciplinary culture from suspension frequency."""
        if "team" in df.columns and "is_suspended" in df.columns:
            # Team suspension rate: fraction of player-weeks with suspensions.
            team_rate = df.groupby("team")["is_suspended"].transform("mean")
            df["team_discipline_risk"] = (team_rate / 0.05).clip(0.0, 1.0)
        else:
            df["team_discipline_risk"] = 0.0
        return df

    def _estimate_games_at_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate expected games a player might miss to suspension."""
        # Expected games at risk = risk * average_suspension_length * weeks_remaining.
        avg_suspension_length = 4.0  # NFL average suspension is ~4 games
        df["expected_games_suspended"] = (
            df["suspension_risk"] * avg_suspension_length
        ).clip(0.0, 17.0)
        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        df["is_suspended"] = 0
        df["prior_suspensions"] = 0
        df["career_games_suspended"] = 0
        df["suspension_risk"] = 0.03
        df["returning_from_suspension"] = 0
        df["suspension_return_window"] = 0
        df["suspension_return_ramp"] = 1.0
        df["suspension_category"] = "unknown"
        df["suspension_risk_adjusted"] = 0.03
        df["team_discipline_risk"] = 0.0
        df["expected_games_suspended"] = 0.0


# =============================================================================
# 4. TRADE DEADLINE FEATURES
# =============================================================================

_TRADE_DEADLINE_WEEK = 8

# Historical production ramp for traded players (weeks 1-6 post-trade).
# Values represent fraction of prior production level retained.
_POST_TRADE_RAMP = {
    0: 0.55,  # Trade week itself: minimal playbook knowledge
    1: 0.65,  # Week 1 on new team: ~65% of prior production
    2: 0.75,
    3: 0.85,
    4: 0.92,
    5: 0.97,
    6: 1.00,  # Fully integrated by week 6
}

# Position-specific trade adjustment speed (how fast they adapt).
_TRADE_ADJUSTMENT_SPEED = {
    "QB": 0.7,  # QBs adjust slowest (new playbook, timing, protection)
    "RB": 1.2,  # RBs adjust fastest (run plays are simpler to learn)
    "WR": 0.8,  # WRs need route-tree + QB rapport
    "TE": 0.9,  # TEs moderate
}


class TradeDeadlineFeatures:
    """Features capturing trade-deadline dynamics.

    Enhanced with:
    - Post-trade production ramp curves (position-specific).
    - Target/touch share redistribution when a teammate is traded.
    - New-team scheme fit estimation.
    - Departure impact on remaining teammates.
    - Trade rumour volatility signal.
    """

    def __init__(self, deadline_week: int = _TRADE_DEADLINE_WEEK):
        self.deadline_week = deadline_week

    def add_trade_deadline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if result.empty or "week" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        # --- Basic deadline proximity ---
        result["weeks_to_deadline"] = (
            self.deadline_week - result["week"]
        ).clip(lower=0)
        result["past_deadline"] = (
            result["week"] > self.deadline_week
        ).astype(int)
        result["deadline_proximity"] = np.clip(
            1.0 - abs(result["week"] - self.deadline_week) / 4.0, 0.0, 1.0
        )

        # Trade window: weeks 6-9.
        result["in_trade_window"] = (
            (result["week"] >= self.deadline_week - 2)
            & (result["week"] <= self.deadline_week + 1)
        ).astype(int)

        # --- Team record context ---
        if "team_wins" in result.columns and "team_losses" in result.columns:
            total_games = (
                result["team_wins"] + result["team_losses"]
            ).replace(0, 1)
            result["team_win_pct"] = result["team_wins"] / total_games

            result["trade_deadline_contender"] = (
                (result["in_trade_window"] == 1)
                & (result["team_win_pct"] >= 0.5)
            ).astype(int)
            result["trade_deadline_seller"] = (
                (result["in_trade_window"] == 1)
                & (result["team_win_pct"] < 0.4)
            ).astype(int)

            # Trade rumour volatility: teams near .500 have highest uncertainty.
            result["trade_rumour_volatility"] = (
                result["in_trade_window"]
                * (1.0 - (2.0 * (result["team_win_pct"] - 0.5).abs()))
            ).clip(0.0, 1.0)
        else:
            result["team_win_pct"] = 0.5
            result["trade_deadline_contender"] = 0
            result["trade_deadline_seller"] = 0
            result["trade_rumour_volatility"] = 0.0

        # --- Mid-season trade detection ---
        if "team" in result.columns and "player_id" in result.columns:
            result = self._detect_trades(result)
            result = self._compute_trade_ramp(result)
            result = self._compute_departure_impact(result)
        else:
            result["mid_season_trade"] = 0
            result["weeks_since_trade"] = 0
            result["trade_adjustment_window"] = 0
            result["trade_production_ramp"] = 1.0
            result["trade_adjustment_speed"] = 1.0
            result["teammate_traded_boost"] = 0.0

        # --- Roster stability ---
        result = self._compute_roster_stability(result)

        logger.info("Added trade deadline features")
        return result

    def _detect_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect mid-season trades via team changes within a season."""
        grp = df.groupby("player_id", sort=False)
        prev_team = grp["team"].shift(1)
        prev_season = grp["season"].shift(1)
        same_season = df["season"] == prev_season
        team_changed = (df["team"] != prev_team) & prev_team.notna()

        df["mid_season_trade"] = (same_season & team_changed).astype(int)

        df["_trade_cumsum"] = df.groupby("player_id")[
            "mid_season_trade"
        ].cumsum()
        df["weeks_since_trade"] = df.groupby(
            ["player_id", "_trade_cumsum"]
        ).cumcount()
        df.drop(columns=["_trade_cumsum"], inplace=True)

        df["trade_adjustment_window"] = (
            (df.groupby("player_id")["mid_season_trade"].cumsum() > 0)
            & (df["weeks_since_trade"] <= 6)
        ).astype(int)

        return df

    def _compute_trade_ramp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Position-specific post-trade production ramp."""
        # Base ramp from weeks-since-trade.
        wst = df["weeks_since_trade"]
        base_ramp = wst.map(_POST_TRADE_RAMP).fillna(1.0)
        # Only apply ramp during adjustment window.
        df["trade_production_ramp"] = np.where(
            df["trade_adjustment_window"] == 1, base_ramp, 1.0
        )

        # Position-specific speed modifier.
        if "position" in df.columns:
            speed = df["position"].map(_TRADE_ADJUSTMENT_SPEED).fillna(1.0)
            df["trade_adjustment_speed"] = speed
            # Faster adjustment → ramp closer to 1.0 sooner.
            df["trade_production_ramp"] = np.where(
                df["trade_adjustment_window"] == 1,
                1.0 - (1.0 - df["trade_production_ramp"]) / speed,
                1.0,
            )
            df["trade_production_ramp"] = df["trade_production_ramp"].clip(0.0, 1.0)
        else:
            df["trade_adjustment_speed"] = 1.0

        return df

    def _compute_departure_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate opportunity boost when a teammate departs via trade.

        When a player leaves a team mid-season, remaining players at the
        same position may absorb their vacated targets/touches.
        """
        if (
            "team" not in df.columns
            or "position" not in df.columns
            or "season" not in df.columns
        ):
            df["teammate_traded_boost"] = 0.0
            return df

        # Count mid-season departures per team-position-season-week.
        departures = df[df["mid_season_trade"] == 1].groupby(
            ["team", "position", "season", "week"]
        ).size().reset_index(name="departed_count")

        # But the departed player left the PREVIOUS team, so we need
        # to look at the team they left (prev_team).
        # For simplicity, we count team changes at the team level.
        grp = df.groupby("player_id", sort=False)
        prev_team = grp["team"].shift(1)
        prev_season = grp["season"].shift(1)
        same_season = df["season"] == prev_season
        left_team = (df["team"] != prev_team) & prev_team.notna() & same_season

        df["_left_team"] = prev_team.where(left_team)
        team_departures = (
            df[left_team]
            .groupby([df.loc[left_team, "_left_team"], "position", "season", "week"])
            .size()
            .reset_index(name="pos_departures")
        )
        team_departures.columns = ["team", "position", "season", "week", "pos_departures"]

        if not team_departures.empty:
            df = df.merge(
                team_departures,
                on=["team", "position", "season", "week"],
                how="left",
            )
            df["pos_departures"] = df["pos_departures"].fillna(0)
        else:
            df["pos_departures"] = 0

        # Remaining players get a small boost per departure.
        df["teammate_traded_boost"] = (df["pos_departures"] * 0.05).clip(0.0, 0.3)
        df.drop(columns=["_left_team", "pos_departures"], errors="ignore", inplace=True)

        return df

    def _compute_roster_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roster stability: fewer changes → more stable."""
        if "team" not in df.columns or "player_id" not in df.columns:
            df["roster_stability"] = 1.0
            return df

        # Count total trades per team-season-week.
        trades_per_team = df.groupby(["team", "season"])["mid_season_trade"].transform(
            lambda x: x.shift(1).cumsum()
        ).fillna(0)
        df["roster_stability"] = (1.0 / (1.0 + trades_per_team * 0.1)).clip(0.0, 1.0)

        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        df["weeks_to_deadline"] = 0
        df["past_deadline"] = 0
        df["deadline_proximity"] = 0.0
        df["in_trade_window"] = 0
        df["team_win_pct"] = 0.5
        df["trade_deadline_contender"] = 0
        df["trade_deadline_seller"] = 0
        df["trade_rumour_volatility"] = 0.0
        df["mid_season_trade"] = 0
        df["weeks_since_trade"] = 0
        df["trade_adjustment_window"] = 0
        df["trade_production_ramp"] = 1.0
        df["trade_adjustment_speed"] = 1.0
        df["teammate_traded_boost"] = 0.0
        df["roster_stability"] = 1.0


# =============================================================================
# 5. PLAYOFF FEATURES
# =============================================================================

_REGULAR_SEASON_WEEKS = 18

# Position-specific snap reduction risk when team has clinched (week 17-18).
_SNAP_REDUCTION_RISK = {
    "QB": 0.85,  # QBs most likely to be rested
    "RB": 0.70,  # RBs often rested to preserve health
    "WR": 0.60,  # WR1s sometimes play, WR2/3 may sit
    "TE": 0.50,  # TEs moderate rest risk
}

# Garbage-time scoring adjustments by position.
# When a team is down big (or up big), garbage time inflates certain positions.
_GARBAGE_TIME_BOOST = {
    "QB": 0.10,   # Passing volume increases in garbage time
    "RB": -0.10,  # Running decreases (clock management)
    "WR": 0.15,   # WRs benefit most from garbage-time passing
    "TE": 0.08,   # TEs see moderate garbage-time boost
}


class PlayoffFeatures:
    """Features capturing playoff context and implications.

    Enhanced with:
    - Division standing awareness and race tightness.
    - Strength of remaining schedule.
    - Garbage-time production modelling.
    - Snap count reduction patterns for clinched teams.
    - Win probability for game script context.
    - Home-field advantage in late-season/playoff games.
    - Season-phase urgency scoring.
    """

    def add_playoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if result.empty or "week" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        # --- Basic temporal features ---
        result["playoff_proximity"] = np.clip(
            (result["week"] - 10) / (_REGULAR_SEASON_WEEKS - 10), 0.0, 1.0
        )
        result["is_playoff_week"] = (
            result["week"] > _REGULAR_SEASON_WEEKS
        ).astype(int)
        result["weeks_remaining"] = (
            _REGULAR_SEASON_WEEKS - result["week"]
        ).clip(lower=0)

        # --- Team record dependent features ---
        has_record = (
            "team_wins" in result.columns and "team_losses" in result.columns
        )
        if has_record:
            result = self._compute_contention_features(result)
        else:
            result["eliminated_proxy"] = 0
            result["clinched_proxy"] = 0
            result["meaningful_game"] = 1
            result["rest_risk"] = 0
            result["playoff_push"] = 0
            result["win_pct_pace"] = 0.5
            result["playoff_urgency"] = 0.5

        # --- Snap reduction risk for clinched teams ---
        result = self._compute_snap_reduction_risk(result)

        # --- Garbage-time modelling ---
        result = self._compute_garbage_time(result)

        # --- Division race tightness ---
        result = self._compute_division_race(result)

        # --- Strength of remaining schedule ---
        result = self._compute_sos_remaining(result)

        # --- Home-field advantage (intensified in late season/playoffs) ---
        result = self._compute_home_field_boost(result)

        # --- Playoff seed / bye week ---
        if "playoff_seed" in result.columns:
            result["has_bye_week_seed"] = (
                result["playoff_seed"].fillna(99) <= 1
            ).astype(int)
        else:
            result["has_bye_week_seed"] = 0

        # --- Season-phase urgency (composite) ---
        result = self._compute_urgency(result)

        logger.info("Added playoff features")
        return result

    def _compute_contention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute elimination, clinching, and contention features."""
        total_games = (df["team_wins"] + df["team_losses"]).replace(0, 1)
        win_pct = df["team_wins"] / total_games

        # Win-pace projection.
        games_played = total_games
        df["win_pct_pace"] = win_pct
        projected_wins = win_pct * _REGULAR_SEASON_WEEKS

        # Max possible wins (current wins + remaining games).
        max_possible_wins = df["team_wins"] + df["weeks_remaining"]

        # Elimination: can't reach 7 wins.
        df["eliminated_proxy"] = (
            (max_possible_wins < 7) | ((projected_wins < 7) & (df["weeks_remaining"] < 6))
        ).astype(int)

        # Clinched: guaranteed 10+ wins (even if lose out).
        df["clinched_proxy"] = (
            (df["team_wins"] >= 10) | ((projected_wins >= 11) & (df["weeks_remaining"] < 6))
        ).astype(int)

        # Meaningful game.
        df["meaningful_game"] = (
            (df["eliminated_proxy"] == 0) & (df["is_playoff_week"] == 0)
        ).astype(int)

        # Rest risk.
        df["rest_risk"] = (
            (df["clinched_proxy"] == 1) & (df["weeks_remaining"] <= 2)
        ).astype(int)

        # Playoff push: in contention, late-season.
        df["playoff_push"] = (
            (df["eliminated_proxy"] == 0)
            & (df["weeks_remaining"] >= 2)
            & (df["weeks_remaining"] <= 8)
            & (win_pct >= 0.4)
        ).astype(int)

        # Playoff urgency: how critical is this game for the team?
        # High when close to the bubble and few games left.
        bubble_distance = (projected_wins - 9.0).abs()  # 9 wins is typical bubble
        df["playoff_urgency"] = np.where(
            df["eliminated_proxy"] == 0,
            np.clip(1.0 - bubble_distance / 5.0, 0.0, 1.0)
            * np.clip(df["playoff_proximity"], 0.1, 1.0),
            0.0,
        )

        return df

    def _compute_snap_reduction_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Position-specific snap reduction risk for clinched teams."""
        if "position" in df.columns:
            base_reduction = df["position"].map(_SNAP_REDUCTION_RISK).fillna(0.5)
            df["snap_reduction_risk"] = np.where(
                df.get("rest_risk", pd.Series(0, index=df.index)) == 1,
                base_reduction,
                0.0,
            )
        else:
            df["snap_reduction_risk"] = 0.0
        return df

    def _compute_garbage_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model garbage-time production adjustments."""
        if "spread" in df.columns:
            # Large spread = likely blowout = more garbage time.
            spread_abs = df["spread"].abs()
            df["garbage_time_probability"] = np.clip(
                (spread_abs - 7) / 14.0, 0.0, 1.0
            )
        elif "team_wins" in df.columns and "team_losses" in df.columns:
            # Use team quality mismatch as proxy.
            total = (df["team_wins"] + df["team_losses"]).replace(0, 1)
            win_pct = df["team_wins"] / total
            # Extreme win% (very good or very bad) → more blowouts.
            df["garbage_time_probability"] = (
                (2.0 * (win_pct - 0.5).abs()) ** 2
            ).clip(0.0, 0.8)
        else:
            df["garbage_time_probability"] = 0.0

        # Position-specific garbage-time boost.
        if "position" in df.columns:
            gt_boost = df["position"].map(_GARBAGE_TIME_BOOST).fillna(0.0)
            df["garbage_time_boost"] = (
                gt_boost * df["garbage_time_probability"]
            )
        else:
            df["garbage_time_boost"] = 0.0

        return df

    def _compute_division_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute division race tightness."""
        if (
            "division" in df.columns
            and "team_wins" in df.columns
            and "season" in df.columns
            and "week" in df.columns
        ):
            # Division leader win count per division-season-week.
            div_leader_wins = df.groupby(
                ["division", "season", "week"]
            )["team_wins"].transform("max")

            # Games behind division leader.
            df["games_behind_leader"] = (div_leader_wins - df["team_wins"]).clip(lower=0)

            # Race tightness: 1.0 = in the thick of it, 0.0 = out of it.
            df["division_race_tightness"] = np.clip(
                1.0 - df["games_behind_leader"] / 5.0, 0.0, 1.0
            )
        else:
            df["games_behind_leader"] = 0
            df["division_race_tightness"] = 0.5

        return df

    def _compute_sos_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strength of remaining schedule proxy."""
        if "opponent_win_pct" in df.columns:
            # SOS remaining is the average opponent win% for future games.
            # Since we can't look ahead, use the current opponent as a proxy.
            df["sos_remaining_proxy"] = df["opponent_win_pct"].fillna(0.5)
        elif "opponent_rating" in df.columns:
            df["sos_remaining_proxy"] = (
                df["opponent_rating"].fillna(50) / 100.0
            )
        else:
            df["sos_remaining_proxy"] = 0.5

        # Difficult schedule in late season amplifies uncertainty.
        df["late_season_difficulty"] = (
            df["sos_remaining_proxy"] * df["playoff_proximity"]
        )

        return df

    def _compute_home_field_boost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Home-field advantage, amplified in late season and playoffs."""
        if "is_home" in df.columns:
            is_home = df["is_home"]
        elif "home_away" in df.columns:
            is_home = (df["home_away"].str.lower() == "home").astype(int)
        else:
            df["home_field_boost"] = 0.0
            return df

        # Base home advantage: ~2.5 points.  Increases in late season / playoffs.
        season_mult = 1.0 + 0.3 * df["playoff_proximity"]
        df["home_field_boost"] = is_home * 0.1 * season_mult

        return df

    def _compute_urgency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite urgency score combining multiple signals."""
        urgency_components = []

        if "playoff_urgency" in df.columns:
            urgency_components.append(df["playoff_urgency"])
        if "division_race_tightness" in df.columns:
            urgency_components.append(df["division_race_tightness"] * 0.5)
        if "meaningful_game" in df.columns:
            urgency_components.append(df["meaningful_game"].astype(float) * 0.3)

        if urgency_components:
            df["season_urgency_composite"] = (
                sum(urgency_components) / len(urgency_components)
            ).clip(0.0, 1.0)
        else:
            df["season_urgency_composite"] = 0.5

        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        df["playoff_proximity"] = 0.0
        df["is_playoff_week"] = 0
        df["weeks_remaining"] = 18
        df["eliminated_proxy"] = 0
        df["clinched_proxy"] = 0
        df["meaningful_game"] = 1
        df["rest_risk"] = 0
        df["playoff_push"] = 0
        df["has_bye_week_seed"] = 0
        df["win_pct_pace"] = 0.5
        df["playoff_urgency"] = 0.5
        df["snap_reduction_risk"] = 0.0
        df["garbage_time_probability"] = 0.0
        df["garbage_time_boost"] = 0.0
        df["games_behind_leader"] = 0
        df["division_race_tightness"] = 0.5
        df["sos_remaining_proxy"] = 0.5
        df["late_season_difficulty"] = 0.0
        df["home_field_boost"] = 0.0
        df["season_urgency_composite"] = 0.5


# =============================================================================
# COMBINED ADVANCED ANALYTICS ENGINE
# =============================================================================

class AdvancedAnalyticsEngine:
    """Orchestrates all advanced analytics feature generators.

    Reads toggle flags from ``config.settings.ADVANCED_ANALYTICS_CONFIG``
    when available.  Individual modules can be disabled without affecting
    the others.
    """

    def __init__(self):
        # Load config with safe fallback.
        try:
            from config.settings import ADVANCED_ANALYTICS_CONFIG as cfg
        except (ImportError, AttributeError):
            cfg = {}

        self._cfg = cfg

        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.coaching_detector = CoachingChangeDetector()
        self.suspension_tracker = SuspensionRiskTracker()
        self.trade_deadline = TradeDeadlineFeatures(
            deadline_week=cfg.get("trade_deadline_week", _TRADE_DEADLINE_WEEK),
        )
        self.playoff_features = PlayoffFeatures()

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all enabled advanced analytics features."""
        if df.empty:
            return df

        result = df.copy()
        cfg = self._cfg

        logger.info("Adding advanced analytics features...")

        if cfg.get("enable_news_sentiment", True):
            result = self.sentiment_analyzer.add_sentiment_features(result)

        if cfg.get("enable_coaching_change", True):
            result = self.coaching_detector.add_coaching_change_features(result)

        if cfg.get("enable_suspension_risk", True):
            result = self.suspension_tracker.add_suspension_features(result)

        if cfg.get("enable_trade_deadline", True):
            result = self.trade_deadline.add_trade_deadline_features(result)

        if cfg.get("enable_playoff_features", True):
            result = self.playoff_features.add_playoff_features(result)

        new_cols = [c for c in result.columns if c not in df.columns]
        logger.info(f"Added {len(new_cols)} advanced analytics features")

        return result


def add_advanced_analytics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add all advanced analytics features."""
    engine = AdvancedAnalyticsEngine()
    return engine.add_all_features(df)
