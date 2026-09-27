import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_calibrated_simulation import (
    DEFAULT_CANDIDATE,
    INDEPENDENT,
    LEGACY_INDEPENDENT,
    MODES,
    PRODUCTION,
    ROLE_FACTOR,
    TEAM_FACTOR,
    LEGACY_ROLE,
    Candidate,
    _sum_groups,
    _write_run,
    candidate_grid,
    fit_candidate,
    minimum_bootstrap_for_holm,
    run,
    select_candidate,
    support_metrics,
)
from scripts.verify_calibrated_simulation import verify as verify_run
from src.models.simulation_evaluation import empirical_crps
from tests.simulation_fixtures import DEPENDENT_LOADINGS, synthetic_panel

SMALL_GRID = candidate_grid((50,), (60,))
N_BOOT = minimum_bootstrap_for_holm() + 20


def _small_run(panel, **kwargs):
    options = dict(draws=40, seed=7, candidates=SMALL_GRID, n_bootstrap=N_BOOT, selection_bootstrap=100)
    options.update(kwargs)
    return run(panel, **options)


@pytest.fixture(scope="module")
def small_panel():
    return synthetic_panel(seasons=(2022, 2023, 2024), weeks=3, n_teams=6,
                           loadings=DEPENDENT_LOADINGS, seed=5)


@pytest.fixture(scope="module")
def small_result(small_panel):
    return _small_run(small_panel)


def test_support_metrics_equal_empirical_crps_on_the_full_support(small_panel):
    history = small_panel[small_panel["season"] == 2022]
    rows = small_panel[small_panel["season"] == 2023].head(20)
    for candidate in SMALL_GRID:
        calibration = fit_candidate(candidate, history)
        crps, _ = support_metrics(calibration, rows)
        for value, row in zip(crps, rows.itertuples(index=False)):
            support = row.predicted_points + calibration.support_deviations(
                position=row.position, predicted_points=row.predicted_points, is_cold_start=False)
            assert value == pytest.approx(empirical_crps(support, row.actual_points), rel=1e-10)


def test_six_modes_share_keys_and_dependent_modes_keep_independent_marginals(small_result):
    draws = small_result.draws
    assert draws.modes == MODES
    shapes = {draws.matrices[mode].shape for mode in MODES}
    assert len(shapes) == 1
    predicted = draws.rows["predicted_points"].to_numpy(float)
    assert np.allclose(draws.matrices[PRODUCTION], predicted)
    independent = np.sort(draws.matrices[INDEPENDENT], axis=0)
    for mode in (TEAM_FACTOR, ROLE_FACTOR, LEGACY_ROLE):
        # Reordering only: every player's draws are the same multiset.
        assert np.array_equal(np.sort(draws.matrices[mode], axis=0), independent)
    # Common random numbers: when both arms select the same candidate the
    # two independent arms are identical draws.
    week_ids = draws.rows["season"].to_numpy() * 100 + draws.rows["week"].to_numpy()
    for entry in small_result.report["selection"]:
        if entry["full"]["candidate"] == entry["legacy"]["candidate"]:
            columns = week_ids == entry["season"] * 100 + entry["week"]
            assert np.array_equal(draws.matrices[INDEPENDENT][:, columns],
                                  draws.matrices[LEGACY_INDEPENDENT][:, columns])
    # Scored weeks are exactly those with an earlier panel season.
    assert small_result.report["scoring_population"]["seasons"] == [2023, 2024]


def test_changing_a_weeks_outcomes_cannot_change_any_draw_or_selection(small_panel, small_result):
    last = small_panel["season"].eq(2024) & small_panel["week"].eq(3)
    changed = small_panel.copy()
    changed.loc[last, "actual_points"] += 25.0
    changed["residual"] = changed["predicted_points"] - changed["actual_points"]
    rerun = _small_run(changed)
    for mode in MODES:
        assert np.array_equal(rerun.draws.matrices[mode], small_result.draws.matrices[mode])
    assert rerun.report["selection"] == small_result.report["selection"]


def test_first_scored_week_uses_the_default_and_logs_why(small_result):
    first = small_result.report["selection"][0]
    assert (first["season"], first["week"]) == (2023, 1)
    assert first["full"]["candidate"] == DEFAULT_CANDIDATE.id
    assert first["full"]["reason"] == "no_prior_evidence"


def _evidence(deltas_by_week, *, coverage=0.8):
    rows = []
    rng = np.random.default_rng(0)
    for week, delta in enumerate(deltas_by_week, start=1):
        base = rng.uniform(3, 5, size=30)
        for candidate, shift in (("legacy:50", 0.0), ("residual:60", delta)):
            rows.append(pd.DataFrame({"candidate": candidate, "week_id": 202300 + week,
                                      "player_id": [f"p{i}" for i in range(30)],
                                      "crps": base + shift, "coverage_80": coverage}))
    return pd.concat(rows, ignore_index=True)


def test_selection_leaves_the_default_only_on_significant_evidence():
    grid = (Candidate("legacy", 50), Candidate("residual", 60))
    consistent = select_candidate(_evidence([-0.2] * 8), target_week_id=202399, candidates=grid)
    assert consistent["candidate"] == "residual:60"
    assert consistent["reason"] == "significant_improvement_over_default"
    noisy = select_candidate(_evidence([-0.9, 0.8, -0.7, 0.75, -0.6, 0.5]), target_week_id=202399,
                             candidates=grid)
    assert noisy["candidate"] == "legacy:50"
    assert noisy["reason"] == "improvement_not_significant"
    # Only strictly earlier weeks count.
    early = select_candidate(_evidence([-0.2] * 8), target_week_id=202301, candidates=grid)
    assert early["reason"] == "no_prior_evidence"
    out_of_band = _evidence([0.1] * 4)
    out_of_band.loc[out_of_band["candidate"] == "legacy:50", "coverage_80"] = 0.6
    chosen = select_candidate(out_of_band, target_week_id=202399, candidates=grid)
    assert chosen["candidate"] == "residual:60"
    assert chosen["reason"] == "default_outside_coverage_band"


def test_confirmation_season_is_withheld_then_scored_alone(small_panel, small_result):
    development = _small_run(small_panel, confirm_season=2024)
    assert development.report["phase"] == "development"
    assert development.report["scoring_population"]["seasons"] == [2023]
    tampered = small_panel.copy()
    confirm = tampered["season"] == 2024
    tampered.loc[confirm, "actual_points"] *= 3.0
    tampered["residual"] = tampered["predicted_points"] - tampered["actual_points"]
    again = _small_run(tampered, confirm_season=2024)
    assert again.report["comparisons"] == development.report["comparisons"]
    confirmation = _small_run(small_panel, confirm_season=2024, run_confirmation=True)
    assert confirmation.report["scoring_population"]["seasons"] == [2024]
    with pytest.raises(ValueError, match="latest season"):
        _small_run(small_panel, confirm_season=2023)
    with pytest.raises(ValueError, match="requires confirm_season"):
        _small_run(small_panel, run_confirmation=True)


def test_bootstrap_too_small_for_holm_is_rejected(small_panel):
    with pytest.raises(ValueError, match="Holm"):
        _small_run(small_panel, n_bootstrap=minimum_bootstrap_for_holm() - 1)


def test_stack_is_quarterback_plus_two_highest_projected_pass_catchers():
    rows = pd.DataFrame({
        "player_id": ["qb", "wr1", "wr2", "te", "rb", "opp_qb", "opp_wr"],
        "team": ["H"] * 5 + ["A"] * 2, "home_team": "H", "away_team": "A",
        "position": ["QB", "WR", "WR", "TE", "RB", "QB", "WR"],
        "predicted_points": [20., 15., 8., 9., 14., 18., 12.],
    })
    groups = {(kind, side): members.tolist() for kind, side, members in _sum_groups(rows)}
    assert groups[("stack", "home")] == [0, 1, 3]
    assert ("stack", "away") not in groups  # only one pass catcher
    assert groups[("team_total", "home")] == [0, 1, 2, 3, 4]
    assert groups[("game_total", "both")] == list(range(7))


def test_saved_run_recomputes_verifies_and_detects_tampering(tmp_path, small_result):
    output = tmp_path / "run"
    _write_run(output, small_result, oof_run_dir=tmp_path / "oof",
               oof_verification={"panel_sha256": "test"}, seed=7, n_bootstrap=N_BOOT, config={"test": True})
    assert verify_run(output)["status"] == "verified"
    assert (output / "summary.md").read_text().startswith("# Calibrated simulation backtest")
    draws = pd.read_parquet(output / "player_draws" / f"{ROLE_FACTOR}.parquet")
    draws.loc[0, "fantasy_points"] += 1.0
    draws.to_parquet(output / "player_draws" / f"{ROLE_FACTOR}.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_run(output)


@pytest.mark.parametrize("loadings, expect_dependence", [(DEPENDENT_LOADINGS, True), (None, False)])
def test_gates_detect_real_dependence_and_reject_its_absence(loadings, expect_dependence):
    panel = synthetic_panel(seasons=(2021, 2022, 2023), weeks=10, n_teams=12, loadings=loadings, seed=21)
    result = run(panel, draws=150, seed=3, candidates=candidate_grid((50,), (100,)),
                 n_bootstrap=400, selection_bootstrap=100)
    gate = result.report["promotion_gate"]
    # Heteroscedastic marginals: the analog family should beat the pooled legacy.
    assert gate[INDEPENDENT]["improves_on_legacy_holm"]
    if expect_dependence:
        assert gate[ROLE_FACTOR]["eligible_for_later_comparison"]
        assert gate["role_structure_over_team_factor"]
    else:
        assert not gate[TEAM_FACTOR]["eligible_for_later_comparison"]
        assert not gate[ROLE_FACTOR]["eligible_for_later_comparison"]
        assert not gate["role_structure_over_team_factor"]
