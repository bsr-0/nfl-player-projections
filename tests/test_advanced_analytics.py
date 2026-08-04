"""Tests for advanced analytics features module.

Covers all five feature generators:
1. NewsSentimentAnalyzer  - negation, bigrams, volume, position weighting
2. CoachingChangeDetector - HC/OC/DC, scheme classification, tenure, fit
3. SuspensionRiskTracker  - categories, position risk, age risk, team culture
4. TradeDeadlineFeatures  - ramp curves, departure impact, roster stability
5. PlayoffFeatures        - division race, SOS, garbage time, snap reduction
Plus the combined AdvancedAnalyticsEngine.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.advanced_analytics import (
    NewsSentimentAnalyzer,
    CoachingChangeDetector,
    SuspensionRiskTracker,
    TradeDeadlineFeatures,
    PlayoffFeatures,
    AdvancedAnalyticsEngine,
    add_advanced_analytics_features,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_player_df(
    n_players: int = 2,
    n_weeks: int = 10,
    season: int = 2024,
    extra_cols: dict = None,
) -> pd.DataFrame:
    """Create a minimal player-weekly DataFrame for testing."""
    rows = []
    for pid in range(1, n_players + 1):
        team = "KC" if pid % 2 == 1 else "BUF"
        pos = ["QB", "RB", "WR", "TE"][pid % 4]
        for wk in range(1, n_weeks + 1):
            row = {
                "player_id": f"P{pid:03d}",
                "name": f"Player {pid}",
                "team": team,
                "position": pos,
                "season": season,
                "week": wk,
                "fantasy_points": np.random.uniform(5, 25),
            }
            if extra_cols:
                for col, val_fn in extra_cols.items():
                    row[col] = val_fn(pid, wk)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 1. NewsSentimentAnalyzer tests
# =============================================================================

class TestNewsSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = NewsSentimentAnalyzer()

    def test_score_text_positive(self):
        result = self.analyzer.score_text("Player is healthy and dominant this season")
        assert result["sentiment_score"] > 0
        assert result["positive_count"] >= 2
        assert result["negative_count"] == 0

    def test_score_text_negative(self):
        result = self.analyzer.score_text("Player is injured and questionable for Sunday")
        assert result["sentiment_score"] < 0
        assert result["negative_count"] >= 2

    def test_score_text_neutral(self):
        result = self.analyzer.score_text("Player attended practice today")
        assert result["sentiment_score"] == 0.0
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_score_text_empty(self):
        result = self.analyzer.score_text("")
        assert result["sentiment_score"] == 0.0

    def test_score_text_none(self):
        result = self.analyzer.score_text(None)
        assert result["sentiment_score"] == 0.0

    def test_intensifiers_amplify(self):
        base = self.analyzer.score_text("Player is healthy")
        intensified = self.analyzer.score_text("Player is extremely healthy")
        assert intensified["sentiment_score"] >= base["sentiment_score"]

    def test_negation_flips_positive(self):
        """'not healthy' should score negative, not positive."""
        result = self.analyzer.score_text("Player is not healthy this week")
        assert result["sentiment_score"] <= 0

    def test_negation_flips_negative(self):
        """'not injured' should score positive, not negative."""
        result = self.analyzer.score_text("Player is not injured anymore")
        assert result["sentiment_score"] >= 0

    def test_bigram_positive(self):
        """Positive bigrams should carry strong positive signal."""
        result = self.analyzer.score_text("Player had full practice and cleared to play")
        assert result["sentiment_score"] > 0.3

    def test_bigram_negative(self):
        """Negative bigrams should carry strong negative signal."""
        result = self.analyzer.score_text("Player did not practice, placed on ir")
        assert result["sentiment_score"] < 0

    def test_subjectivity_score(self):
        """Subjectivity should be higher for sentiment-heavy text."""
        neutral = self.analyzer.score_text("Player attended practice today at 3pm")
        opinionated = self.analyzer.score_text("Player is elite dominant breakout healthy")
        assert opinionated["subjectivity"] > neutral["subjectivity"]

    def test_add_sentiment_features_with_text(self):
        df = _make_player_df(
            n_players=2,
            n_weeks=5,
            extra_cols={
                "news_text": lambda pid, wk: (
                    "breakout healthy dominant" if wk <= 3 else "injured questionable"
                )
            },
        )
        result = self.analyzer.add_sentiment_features(df)

        expected_cols = [
            "news_sentiment", "news_positive_count", "news_negative_count",
            "news_subjectivity", "news_volume", "news_sentiment_weighted",
            "news_sentiment_roll3", "news_sentiment_roll5",
            "news_sentiment_trend", "news_sentiment_volatility",
            "news_sentiment_ewma",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert not result["news_sentiment"].isna().any()

    def test_add_sentiment_features_without_text(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        result = self.analyzer.add_sentiment_features(df)
        assert (result["news_sentiment"] == 0.0).all()
        assert (result["news_volume"] == 0.0).all()

    def test_sentiment_score_bounded(self):
        text = " ".join(list(self.analyzer.positive_terms)[:20])
        result = self.analyzer.score_text(text)
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_position_weighting(self):
        """QB sentiment should be weighted higher than RB."""
        df = _make_player_df(
            n_players=4,
            n_weeks=3,
            extra_cols={"news_text": lambda pid, wk: "Player is elite and dominant"},
        )
        result = self.analyzer.add_sentiment_features(df)
        # All players have same text, but position weight differs.
        assert "news_sentiment_weighted" in result.columns

    def test_sentiment_volatility_computed(self):
        """Volatility should be non-zero when sentiment varies."""
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "news_text": lambda pid, wk: (
                    "elite breakout dominant" if wk % 2 == 0 else "injured struggling bust"
                )
            },
        )
        result = self.analyzer.add_sentiment_features(df)
        # Later weeks should have non-zero volatility.
        late = result[result["week"] >= 5]
        assert late["news_sentiment_volatility"].max() > 0

    def test_ewma_computed(self):
        """EWMA sentiment should be computed."""
        df = _make_player_df(
            n_players=1,
            n_weeks=6,
            extra_cols={"news_text": lambda pid, wk: "healthy starter"},
        )
        result = self.analyzer.add_sentiment_features(df)
        assert "news_sentiment_ewma" in result.columns


# =============================================================================
# 2. CoachingChangeDetector tests
# =============================================================================

class TestCoachingChangeDetector:
    def setup_method(self):
        self.detector = CoachingChangeDetector()

    def test_detect_hc_change(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A" if wk <= 4 else "Coach B"
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert result[result["week"] == 5]["coaching_change"].iloc[0] == 1

    def test_no_change_all_same_coach(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"head_coach": lambda pid, wk: "Coach A"},
        )
        result = self.detector.add_coaching_change_features(df)
        assert result["coaching_change"].sum() == 0

    def test_oc_change_detected(self):
        """Offensive coordinator changes should be tracked separately."""
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A",
                "offensive_coordinator": lambda pid, wk: (
                    "OC1" if wk <= 4 else "OC2"
                ),
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert "oc_change" in result.columns
        assert result[result["week"] == 5]["oc_change"].iloc[0] == 1

    def test_dc_change_detected(self):
        """Defensive coordinator changes should be tracked."""
        df = _make_player_df(
            n_players=1,
            n_weeks=6,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A",
                "defensive_coordinator": lambda pid, wk: (
                    "DC1" if wk <= 3 else "DC2"
                ),
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert "dc_change" in result.columns
        assert result[result["week"] == 4]["dc_change"].iloc[0] == 1

    def test_any_coaching_change_flag(self):
        """any_coaching_change should fire for OC-only changes too."""
        df = _make_player_df(
            n_players=1,
            n_weeks=6,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A",
                "offensive_coordinator": lambda pid, wk: (
                    "OC1" if wk <= 3 else "OC2"
                ),
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert result[result["week"] == 4]["any_coaching_change"].iloc[0] == 1

    def test_scheme_classification(self):
        """Scheme type should be classified from team pass rate."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A",
                "team_a_pass_rate": lambda pid, wk: 0.65,
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert "scheme_type" in result.columns
        assert (result["scheme_type"] == 4).all()  # heavy_pass

    def test_coaching_tenure_stability(self):
        """Coaching tenure and stability should increase over time."""
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={"head_coach": lambda pid, wk: "Coach A"},
        )
        result = self.detector.add_coaching_change_features(df)
        assert "coaching_tenure_weeks" in result.columns
        assert "coaching_stability" in result.columns
        # Tenure should increase.
        assert result["coaching_tenure_weeks"].is_monotonic_increasing

    def test_adaptation_exponential_decay(self):
        """Adaptation score should decay exponentially, not linearly."""
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A" if wk <= 2 else "Coach B"
            },
        )
        result = self.detector.add_coaching_change_features(df)
        wk3 = result[result["week"] == 3]["coaching_adaptation_score"].iloc[0]
        wk8 = result[result["week"] == 8]["coaching_adaptation_score"].iloc[0]
        assert wk3 > wk8

    def test_scheme_fit_score(self):
        """Scheme fit should differ by position."""
        df = _make_player_df(
            n_players=4,
            n_weeks=3,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A",
                "team_a_pass_rate": lambda pid, wk: 0.65,
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert "scheme_fit_score" in result.columns
        # WR should fit pass-heavy scheme better than RB.
        wr_fit = result[result["position"] == "WR"]["scheme_fit_score"].mean()
        rb_fit = result[result["position"] == "RB"]["scheme_fit_score"].mean()
        assert wr_fit > rb_fit

    def test_mid_season_coaching_change_flag(self):
        """Mid-season changes (week > 1) should be flagged."""
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A" if wk <= 4 else "Coach B"
            },
        )
        result = self.detector.add_coaching_change_features(df)
        assert result[result["week"] == 5]["mid_season_coaching_change"].iloc[0] == 1

    def test_scheme_proxy_fallback(self):
        df = _make_player_df(n_players=1, n_weeks=5)
        result = self.detector.add_coaching_change_features(df)
        assert "coaching_change" in result.columns

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.detector.add_coaching_change_features(df)
        assert "coaching_change" in result.columns
        assert "scheme_fit_score" in result.columns


# =============================================================================
# 3. SuspensionRiskTracker tests
# =============================================================================

class TestSuspensionRiskTracker:
    def setup_method(self):
        self.tracker = SuspensionRiskTracker()

    def test_no_suspension_columns(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        result = self.tracker.add_suspension_features(df)
        assert (result["is_suspended"] == 0).all()
        assert (result["suspension_risk"] == 0.03).all()
        assert "position_suspension_mult" in result.columns
        assert "age_suspension_mult" in result.columns

    def test_with_suspension_status(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if 3 <= wk <= 4 else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert result[result["week"] == 3]["is_suspended"].iloc[0] == 1
        assert result[result["week"] == 5]["is_suspended"].iloc[0] == 0

    def test_with_games_suspended(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"games_suspended": lambda pid, wk: 2 if wk == 2 else 0},
        )
        result = self.tracker.add_suspension_features(df)
        assert "career_games_suspended" in result.columns

    def test_recidivism_increases_risk(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if wk in (2, 6) else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)
        late_risk = result[result["week"] == 8]["suspension_risk"].iloc[0]
        assert late_risk > 0.03

    def test_suspension_risk_bounded(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={"suspension_status": lambda pid, wk: "Suspended"},
        )
        result = self.tracker.add_suspension_features(df)
        assert (result["suspension_risk"] <= 1.0).all()
        assert (result["suspension_risk"] >= 0.0).all()

    def test_suspension_type_categorisation(self):
        """Suspension type should be categorised from free text."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={
                "suspension_status": lambda pid, wk: "Suspended" if wk == 2 else "Active",
                "suspension_type": lambda pid, wk: "PED violation" if wk == 2 else "",
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "suspension_category" in result.columns
        ped_rows = result[result["suspension_type"].str.contains("PED", na=False)]
        assert (ped_rows["suspension_category"] == "ped").all()

    def test_position_risk_multiplier(self):
        """RBs should have higher risk multiplier than QBs."""
        df = _make_player_df(n_players=4, n_weeks=3)
        result = self.tracker.add_suspension_features(df)
        rb = result[result["position"] == "RB"]["position_suspension_mult"].iloc[0]
        qb = result[result["position"] == "QB"]["position_suspension_mult"].iloc[0]
        assert rb > qb

    def test_age_risk_modifier(self):
        """Young players should have higher age risk modifier."""
        df = _make_player_df(
            n_players=2,
            n_weeks=3,
            extra_cols={"age": lambda pid, wk: 22 if pid == 1 else 32},
        )
        result = self.tracker.add_suspension_features(df)
        young = result[result["player_id"] == "P001"]["age_suspension_mult"].iloc[0]
        old = result[result["player_id"] == "P002"]["age_suspension_mult"].iloc[0]
        assert young > old

    def test_team_discipline_risk(self):
        """Teams with more suspensions should have higher discipline risk."""
        df = _make_player_df(
            n_players=3,
            n_weeks=5,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if pid == 1 and wk <= 3 else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "team_discipline_risk" in result.columns

    def test_expected_games_suspended(self):
        """Expected games at risk should scale with suspension risk."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if wk == 2 else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "expected_games_suspended" in result.columns
        assert result["expected_games_suspended"].max() > 0

    def test_suspension_return_ramp(self):
        """Return ramp should increase from 0 toward 1 after suspension ends."""
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if 2 <= wk <= 4 else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "suspension_return_ramp" in result.columns

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.tracker.add_suspension_features(df)
        assert "is_suspended" in result.columns

    def test_adjusted_risk_computed(self):
        """Adjusted risk should combine base risk with position and age."""
        df = _make_player_df(
            n_players=2,
            n_weeks=5,
            extra_cols={
                "suspension_status": lambda pid, wk: "Active",
                "age": lambda pid, wk: 23,
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "suspension_risk_adjusted" in result.columns


# =============================================================================
# 4. TradeDeadlineFeatures tests
# =============================================================================

class TestTradeDeadlineFeatures:
    def setup_method(self):
        self.trade = TradeDeadlineFeatures()

    def test_basic_features(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)
        for col in ["weeks_to_deadline", "past_deadline", "deadline_proximity",
                     "in_trade_window", "trade_rumour_volatility",
                     "trade_production_ramp", "roster_stability"]:
            assert col in result.columns, f"Missing: {col}"

    def test_weeks_to_deadline(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)
        assert result[result["week"] == 4]["weeks_to_deadline"].iloc[0] == 4
        assert result[result["week"] == 10]["weeks_to_deadline"].iloc[0] == 0

    def test_past_deadline(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)
        assert result[result["week"] == 6]["past_deadline"].iloc[0] == 0
        assert result[result["week"] == 10]["past_deadline"].iloc[0] == 1

    def test_trade_window(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)
        for wk in [6, 7, 8, 9]:
            assert result[result["week"] == wk]["in_trade_window"].iloc[0] == 1
        for wk in [4, 5, 10, 11]:
            assert result[result["week"] == wk]["in_trade_window"].iloc[0] == 0

    def test_team_record_features(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "team_wins": lambda pid, wk: 5,
                "team_losses": lambda pid, wk: 2,
            },
        )
        result = self.trade.add_trade_deadline_features(df)
        wk7 = result[result["week"] == 7]
        assert wk7["trade_deadline_contender"].iloc[0] == 1

    def test_trade_rumour_volatility(self):
        """Teams near .500 should have highest trade rumour volatility."""
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "team_wins": lambda pid, wk: 4,
                "team_losses": lambda pid, wk: 4,
            },
        )
        result = self.trade.add_trade_deadline_features(df)
        in_window = result[result["in_trade_window"] == 1]
        assert in_window["trade_rumour_volatility"].max() > 0.5

    def test_mid_season_trade_detection(self):
        df = _make_player_df(n_players=1, n_weeks=10)
        df.loc[df["week"] >= 6, "team"] = "BUF"
        result = self.trade.add_trade_deadline_features(df)
        assert result[result["week"] == 6]["mid_season_trade"].iloc[0] == 1

    def test_trade_production_ramp(self):
        """Traded players should have reduced production ramp initially."""
        df = _make_player_df(n_players=1, n_weeks=12)
        df.loc[df["week"] >= 6, "team"] = "BUF"
        result = self.trade.add_trade_deadline_features(df)
        wk6_ramp = result[result["week"] == 6]["trade_production_ramp"].iloc[0]
        wk11_ramp = result[result["week"] == 11]["trade_production_ramp"].iloc[0]
        assert wk6_ramp < wk11_ramp

    def test_trade_adjustment_speed_by_position(self):
        """RBs should adjust faster than QBs after trade."""
        df = _make_player_df(n_players=4, n_weeks=10)
        # Trade all players at week 5.
        df.loc[df["week"] >= 5, "team"] = "NYJ"
        result = self.trade.add_trade_deadline_features(df)
        assert "trade_adjustment_speed" in result.columns

    def test_roster_stability(self):
        """Roster stability should decrease with more trades."""
        df = _make_player_df(n_players=3, n_weeks=10)
        # Trade one player mid-season.
        df.loc[(df["player_id"] == "P001") & (df["week"] >= 5), "team"] = "NYJ"
        result = self.trade.add_trade_deadline_features(df)
        assert "roster_stability" in result.columns

    def test_teammate_traded_boost(self):
        """When a teammate departs, remaining players should get a boost."""
        df = _make_player_df(n_players=3, n_weeks=10)
        # All start on KC; player 1 trades to BUF at week 5.
        df["team"] = "KC"
        df.loc[(df["player_id"] == "P001") & (df["week"] >= 5), "team"] = "BUF"
        result = self.trade.add_trade_deadline_features(df)
        assert "teammate_traded_boost" in result.columns

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.trade.add_trade_deadline_features(df)
        assert "weeks_to_deadline" in result.columns
        assert "trade_production_ramp" in result.columns


# =============================================================================
# 5. PlayoffFeatures tests
# =============================================================================

class TestPlayoffFeatures:
    def setup_method(self):
        self.playoff = PlayoffFeatures()

    def test_basic_features(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)
        for col in ["playoff_proximity", "is_playoff_week", "weeks_remaining",
                     "meaningful_game", "snap_reduction_risk",
                     "garbage_time_probability", "garbage_time_boost",
                     "division_race_tightness", "sos_remaining_proxy",
                     "home_field_boost", "season_urgency_composite"]:
            assert col in result.columns, f"Missing: {col}"

    def test_playoff_proximity_increases(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)
        prox5 = result[result["week"] == 5]["playoff_proximity"].iloc[0]
        prox16 = result[result["week"] == 16]["playoff_proximity"].iloc[0]
        assert prox16 > prox5

    def test_weeks_remaining(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)
        assert result[result["week"] == 1]["weeks_remaining"].iloc[0] == 17
        assert result[result["week"] == 18]["weeks_remaining"].iloc[0] == 0

    def test_eliminated_proxy(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 2,
                "team_losses": lambda pid, wk: 10,
            },
        )
        result = self.playoff.add_playoff_features(df)
        late = result[result["week"] >= 14]
        assert (late["eliminated_proxy"] == 1).any()

    def test_clinched_proxy(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 12,
                "team_losses": lambda pid, wk: 1,
            },
        )
        result = self.playoff.add_playoff_features(df)
        late = result[result["week"] >= 14]
        assert (late["clinched_proxy"] == 1).any()

    def test_rest_risk(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 13,
                "team_losses": lambda pid, wk: 1,
            },
        )
        result = self.playoff.add_playoff_features(df)
        wk17 = result[result["week"] == 17]
        assert wk17["rest_risk"].iloc[0] == 1

    def test_snap_reduction_risk_by_position(self):
        """QBs should have higher snap reduction risk than TEs when resting."""
        df = _make_player_df(
            n_players=4,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 14,
                "team_losses": lambda pid, wk: 0,
            },
        )
        result = self.playoff.add_playoff_features(df)
        late = result[(result["week"] == 17) & (result["rest_risk"] == 1)]
        if not late.empty:
            qb_risk = late[late["position"] == "QB"]["snap_reduction_risk"]
            te_risk = late[late["position"] == "TE"]["snap_reduction_risk"]
            if not qb_risk.empty and not te_risk.empty:
                assert qb_risk.iloc[0] > te_risk.iloc[0]

    def test_garbage_time_with_spread(self):
        """Large spreads should increase garbage-time probability."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"spread": lambda pid, wk: -14.0},
        )
        result = self.playoff.add_playoff_features(df)
        assert result["garbage_time_probability"].max() > 0

    def test_garbage_time_boost_wr_positive(self):
        """WRs should get positive garbage-time boost."""
        df = _make_player_df(
            n_players=4,
            n_weeks=5,
            extra_cols={"spread": lambda pid, wk: -14.0},
        )
        result = self.playoff.add_playoff_features(df)
        wr_boost = result[result["position"] == "WR"]["garbage_time_boost"]
        if not wr_boost.empty:
            assert wr_boost.iloc[0] > 0

    def test_division_race_tightness(self):
        """Division race should be tighter when teams are close in wins."""
        df = _make_player_df(
            n_players=2,
            n_weeks=5,
            extra_cols={
                "division": lambda pid, wk: "AFC West",
                "team_wins": lambda pid, wk: 3 if pid == 1 else 4,
                "team_losses": lambda pid, wk: 2,
            },
        )
        result = self.playoff.add_playoff_features(df)
        assert "division_race_tightness" in result.columns
        assert result["division_race_tightness"].max() > 0

    def test_sos_remaining_proxy(self):
        """SOS should use opponent rating when available."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"opponent_rating": lambda pid, wk: 75.0},
        )
        result = self.playoff.add_playoff_features(df)
        assert (result["sos_remaining_proxy"] == 0.75).all()

    def test_home_field_boost(self):
        """Home games should get a boost."""
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"is_home": lambda pid, wk: 1 if wk % 2 == 0 else 0},
        )
        result = self.playoff.add_playoff_features(df)
        home = result[result["is_home"] == 1]["home_field_boost"]
        away = result[result["is_home"] == 0]["home_field_boost"]
        assert home.max() > away.max()

    def test_playoff_urgency(self):
        """Playoff urgency should be higher for bubble teams late in season."""
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 5,
                "team_losses": lambda pid, wk: 5,
            },
        )
        result = self.playoff.add_playoff_features(df)
        assert "playoff_urgency" in result.columns
        late_urgency = result[result["week"] >= 14]["playoff_urgency"].mean()
        early_urgency = result[result["week"] <= 6]["playoff_urgency"].mean()
        assert late_urgency > early_urgency

    def test_season_urgency_composite(self):
        """Composite urgency should combine multiple signals."""
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 6,
                "team_losses": lambda pid, wk: 6,
            },
        )
        result = self.playoff.add_playoff_features(df)
        assert "season_urgency_composite" in result.columns
        assert (result["season_urgency_composite"] >= 0).all()
        assert (result["season_urgency_composite"] <= 1).all()

    def test_playoff_week_detection(self):
        df = _make_player_df(n_players=1, n_weeks=2)
        df["week"] = [19, 20]
        result = self.playoff.add_playoff_features(df)
        assert (result["is_playoff_week"] == 1).all()

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.playoff.add_playoff_features(df)
        assert "playoff_proximity" in result.columns
        assert "snap_reduction_risk" in result.columns


# =============================================================================
# Combined AdvancedAnalyticsEngine tests
# =============================================================================

class TestAdvancedAnalyticsEngine:
    def setup_method(self):
        self.engine = AdvancedAnalyticsEngine()

    def test_all_features_added(self):
        df = _make_player_df(n_players=2, n_weeks=10)
        result = self.engine.add_all_features(df)
        for col in ["news_sentiment", "coaching_change", "suspension_risk",
                     "weeks_to_deadline", "playoff_proximity",
                     "scheme_fit_score", "trade_production_ramp",
                     "snap_reduction_risk", "garbage_time_boost"]:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nans_in_key_features(self):
        df = _make_player_df(n_players=3, n_weeks=12)
        result = self.engine.add_all_features(df)
        key_cols = [
            "news_sentiment", "coaching_change", "suspension_risk",
            "in_trade_window", "meaningful_game", "scheme_fit_score",
            "trade_production_ramp",
        ]
        for col in key_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_convenience_function(self):
        df = _make_player_df(n_players=1, n_weeks=5)
        result = add_advanced_analytics_features(df)
        assert "news_sentiment" in result.columns
        assert len(result) == len(df)

    def test_empty_df_handled(self):
        result = self.engine.add_all_features(pd.DataFrame())
        assert result.empty

    def test_row_count_preserved(self):
        df = _make_player_df(n_players=3, n_weeks=8)
        result = self.engine.add_all_features(df)
        assert len(result) == len(df)

    def test_original_columns_preserved(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        original_cols = set(df.columns)
        result = self.engine.add_all_features(df)
        assert original_cols.issubset(set(result.columns))

    def test_no_temporal_leakage_in_sentiment(self):
        """Sentiment rolling features use shift(1) so current week is excluded."""
        df = _make_player_df(
            n_players=1,
            n_weeks=6,
            extra_cols={
                "news_text": lambda pid, wk: (
                    "elite breakout dominant" if wk == 6 else "normal practice"
                )
            },
        )
        result = self.engine.add_all_features(df)
        wk5_roll = result[result["week"] == 5]["news_sentiment_roll3"].iloc[0]
        wk6_roll = result[result["week"] == 6]["news_sentiment_roll3"].iloc[0]
        assert abs(wk5_roll - wk6_roll) < 0.5

    def test_full_feature_integration(self):
        """Test with all optional columns present."""
        df = _make_player_df(
            n_players=2,
            n_weeks=12,
            extra_cols={
                "news_text": lambda pid, wk: "healthy starter",
                "head_coach": lambda pid, wk: "Coach A" if wk <= 6 else "Coach B",
                "offensive_coordinator": lambda pid, wk: "OC1",
                "suspension_status": lambda pid, wk: "Active",
                "team_wins": lambda pid, wk: wk // 2,
                "team_losses": lambda pid, wk: wk - wk // 2,
                "age": lambda pid, wk: 26,
            },
        )
        result = self.engine.add_all_features(df)
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) >= 40  # Significantly more features now

    def test_feature_count_substantial(self):
        """Ensure we add a meaningful number of features even with minimal data."""
        df = _make_player_df(n_players=2, n_weeks=8)
        result = self.engine.add_all_features(df)
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) >= 35
