from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from src.models.game_simulation import (
    GameScriptInput, PlayerSimulationInput, GameDraw, _draw_split_target_multipliers,
    _split_opportunity_multiplier, simulate_players,
)
from src.models.rb_ppr_split import fit_rb_receiving_fraction
from src.evaluation.rb_split_evaluation import evaluate_paired_panel, paired_week_interval


def history():
    return pd.DataFrame([
        dict(player_id='a', season=s, week=w, position='RB', receptions=3., receiving_yards=20.,
             receiving_tds=0., rushing_yards=50., rushing_tds=0.)
        for s, w in [(2022, 1), (2022, 2), (2023, 1), (2023, 2)]])


def game():
    return GameScriptInput('2026_2_A_B', 'A', 'B', .6, 3., 45.)


def test_fraction_has_no_current_or_future_label_dependency():
    h = history()
    m = fit_rb_receiving_fraction(h, train_through_season=2022)
    rows = pd.DataFrame([dict(player_id='a', season=2023, week=2, position='RB'),
                         dict(player_id='rookie', season=2023, week=2, position='RB')])
    expected = m.predict(rows, h)
    corrupted = h.copy()
    corrupted.loc[(corrupted.season == 2023) & (corrupted.week >= 2), 'receiving_tds'] = 100000
    assert fit_rb_receiving_fraction(corrupted, train_through_season=2022) == m
    pd.testing.assert_frame_equal(expected, m.predict(rows, corrupted))
    assert expected.history_games.tolist() == [3, 0]
    assert expected.receiving_points_fraction.tolist() == [.5, .5]
    with pytest.raises(ValueError, match='before'):
        m.predict(rows.assign(season=2022), h)


def test_component_conservation_and_script_direction():
    p = PlayerSimulationInput('a', 'A', 'RB', 18., receiving_points_fraction=10/18)
    assert _split_opportunity_multiplier(p, 1., 1.) == 1.
    assert 18 * _split_opportunity_multiplier(p, .8, 1.) == pytest.approx(16.4)
    assert 18 * _split_opportunity_multiplier(p, .8, 1.2) == pytest.approx(18.4)
    assert _split_opportunity_multiplier(replace(p, receiving_points_fraction=0), .8, 1.2) == .8
    assert _split_opportunity_multiplier(replace(p, receiving_points_fraction=1), .8, 1.2) == 1.2


def test_common_random_numbers_legacy_nonrb_activity_and_disabled_script():
    players = [PlayerSimulationInput('a', 'A', 'RB', 18.), PlayerSimulationInput('b', 'A', 'WR', 16.),
               PlayerSimulationInput('c', 'B', 'QB', 22.)]
    old = pd.DataFrame(simulate_players(game(), players, 200, seed=18))
    split = [replace(players[0], receiving_points_fraction=.5), *players[1:]]
    new = pd.DataFrame(simulate_players(game(), split, 200, seed=18))
    pd.testing.assert_frame_equal(old.loc[old.position.ne('RB')], new.loc[new.position.ne('RB')])
    zero = [replace(players[0], receiving_points_fraction=0.), *players[1:]]
    assert simulate_players(game(), players, 200) == simulate_players(game(), zero, 200)
    assert simulate_players(game(), players, 200, apply_game_script=False) == simulate_players(game(), split, 200, apply_game_script=False)
    inactive = [replace(split[0], participation_prob=0.)]
    assert all(r['fantasy_points'] == 0 for r in simulate_players(game(), inactive, 20))


def test_rb_targets_share_one_receiver_pool_and_exclude_qb():
    players = [PlayerSimulationInput('a', 'A', 'RB', 18., usage_share=.7,
                                    receiving_points_fraction=.5, receiving_usage_share=.2),
               PlayerSimulationInput('b', 'A', 'WR', 16., usage_share=.5),
               PlayerSimulationInput('c', 'A', 'QB', 22., usage_share=1.)]
    script = GameDraw(0, 20., 20., 60, 60, 30, 30, True, 0., 40.)
    for seed in range(10):
        mult = _draw_split_target_multipliers(game(), players, script, np.random.default_rng(seed))
        assert set(mult) == {0, 1}
        allocated = sum(mult[i] * game().home.plays * game().home.pass_rate * s for i,s in [(0,.2),(1,.5)])
        assert allocated <= 30 + 1e-8
    with pytest.raises(ValueError, match='exceed'):
        _draw_split_target_multipliers(game(), [replace(players[0], receiving_usage_share=.8), *players[1:]], script, np.random.default_rng(1))


@pytest.mark.parametrize('fraction', [-1., 1.1, float('nan'), float('inf')])
def test_bad_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        PlayerSimulationInput('a', 'A', 'RB', 10., receiving_points_fraction=fraction)


def panel():
    return pd.DataFrame([dict(player_id=str(i), season=2026, week=w, position='RB', receiving_role='mixed',
                              actual=10., served=12., game_sim=11., rb_split=10.5)
                         for w in range(4, 12) for i in range(w)])


def test_gates_require_complete_confirmation_and_both_baselines():
    p = panel()
    assert evaluate_paired_panel(p)['decision'] == 'INCONCLUSIVE'
    assert evaluate_paired_panel(p, confirmation=True, confirmation_evidence_valid=True)['decision'] == 'PASS'
    p.loc[0, 'actual'] = np.nan
    r = evaluate_paired_panel(p, confirmation=True, confirmation_evidence_valid=True)
    assert r['decision'] == 'INCONCLUSIVE' and r['missing_outcomes'] == 1
    p = panel().assign(game_sim=10.1)
    assert evaluate_paired_panel(p, confirmation=True, confirmation_evidence_valid=True)['decision'] == 'FAIL'
    with pytest.raises(ValueError, match='predictions'):
        evaluate_paired_panel(panel().assign(game_sim=np.nan))


def test_bootstrap_weights_player_weeks_not_week_means():
    p = panel()
    p['rb_split'] = np.where(p.week.eq(4), 13., 10.5)
    got = paired_week_interval(p, 'served', replicates=100, seed=123)
    weeks = list(p.groupby(['season','week']))
    draws = []
    for indices in np.random.default_rng(123).integers(0,len(weeks),size=(100,len(weeks))):
        rows = pd.concat([weeks[i][1] for i in indices])
        draws.append(((rows.rb_split-rows.actual).abs()-(rows.served-rows.actual).abs()).mean())
    assert got == pytest.approx(np.quantile(draws,[.025,.975]))
    assert paired_week_interval(p[p.week.eq(4)], 'served') is None


def test_overall_guardrail_catches_nonrb_regression():
    p=panel()
    others=p.assign(player_id=lambda x:'wr'+x.player_id, position='WR', receiving_role='not_rb',
                    served=10., game_sim=10., rb_split=20.)
    r=evaluate_paired_panel(pd.concat([p,others]),confirmation=True,confirmation_evidence_valid=True)
    assert r['decision']=='FAIL' and not r['acceptance_gates']['overall_no_worse_each']
