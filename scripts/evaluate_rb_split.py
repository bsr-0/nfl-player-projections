#!/usr/bin/env python3
"""Frozen, research-only RB split experiment. Never writes serving artifacts.

python scripts/evaluate_rb_split.py --output-dir data/experiments/rb_split_20260927/results
python scripts/evaluate_rb_split.py --verify data/experiments/rb_split_20260927/results
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.models.game_simulation import simulate_players
from src.models.simulation_adapter import simulation_inputs_from_predictions
from src.models.rb_ppr_split import fit_rb_receiving_fraction, component_history, KEY
from src.evaluation.rb_split_evaluation import evaluate_paired_panel

EXPERIMENT = ROOT / 'data/experiments/rb_split_20260927'
CONFIG = {'candidate': {'lookback_games': 8, 'prior_games': 4.0},
          'validation_seasons': [2023, 2024, 2025], 'final_train_through': 2025,
          'draws_per_seed': 10000, 'simulation_seeds': [42, 20260927],
          'bootstrap_replicates': 5000, 'bootstrap_seed': 20260927,
          'confirmation_seasons_weeks': {'2026': list(range(4, 19))}}
SCORING = {'passing_yards': .04, 'passing_tds': 4., 'interceptions': -2.,
           'rushing_yards': .1, 'rushing_tds': 6., 'receptions': 1.,
           'receiving_yards': .1, 'receiving_tds': 6., 'fumbles_lost': -2., 'two_point_conversions': 2.}
SOURCES = ['src/models/game_simulation.py', 'src/models/rb_ppr_split.py',
           'src/models/player_correlation.py', 'src/models/usage_allocation.py',
           'src/models/simulation_adapter.py', 'src/evaluation/rb_split_evaluation.py',
           'scripts/evaluate_rb_split.py', 'scripts/generate_simulation_data.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def load_pinned_baseline(root):
    name = '_rb_split_pinned_game_simulation'
    spec = importlib.util.spec_from_file_location(name, root/'baseline/src/models/game_simulation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_inputs(root):
    manifest = json.loads((root/'baseline_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(root/'baseline'/name) != digest:
            raise ValueError(f'pinned baseline changed: {name}')
        if name != 'src/models/game_simulation.py' and sha(ROOT/name) != digest:
            raise ValueError(f'non-hypothesis baseline dependency changed: {name}')
    for name,digest in manifest['inputs'].items():
        if sha(root/'inputs'/name) != digest:
            raise ValueError(f'frozen input changed: {name}')
    return manifest


def prepare_cohort(root, manifest):
    inputs = root/'inputs'
    players = pd.read_json(inputs/'archived_weekly_2026_wk2.json')
    games = pd.read_json(inputs/'archived_game_predictions_2026_wk2.json')
    players['season'], players['week'] = 2026, 2
    games['season'], games['week'] = 2026, 2
    if players.duplicated(KEY).any() or not players.position.isin(['QB','RB','WR','TE']).all():
        raise ValueError('archive duplicate player-weeks or unsupported positions')
    if not np.isfinite(players.predicted_points.to_numpy(float)).all():
        raise ValueError('missing/nonfinite archived forecast')
    # Remove embedded outcomes before any simulator/model input is built.
    players = players.drop(columns=['actual_points'], errors='ignore')
    schedule = pd.read_csv(inputs/'schedule.csv')
    schedule = schedule.loc[schedule.season.eq(2026)&schedule.week.eq(2)]
    game_keys = ['season','week','home_team','away_team']
    games = games.merge(schedule[game_keys+['game_time']],on=game_keys,how='left',validate='one_to_one')
    if games.game_time.isna().any():
        raise ValueError('archived game lacks unique schedule date')
    # Date-only kickoff source: exclude the whole capture day and all earlier games.
    capture_day = pd.Timestamp(manifest['archive_commit_time']).tz_convert('UTC').date()
    games['eligible'] = pd.to_datetime(games.game_time).dt.date > capture_day
    pairs = pd.concat([games.assign(team=games.home_team,opponent=games.away_team),
                       games.assign(team=games.away_team,opponent=games.home_team)])
    players = players.merge(pairs[['team','opponent','game_time','eligible']],on=['team','opponent'],how='left',validate='many_to_one')
    if players.eligible.isna().any():
        raise ValueError('player schedule/opponent mismatch')
    excluded = players.loc[~players.eligible].assign(exclusion_reason='game_on_or_before_archive_capture_date')
    cohort = players.loc[players.eligible].copy()
    return cohort, games.loc[games.eligible].copy(), excluded, len(players)


def fraction_diagnostics(history):
    diagnostics=[]
    for season in CONFIG['validation_seasons']:
        model=fit_rb_receiving_fraction(history,train_through_season=season-1,**CONFIG['candidate'])
        test=history.loc[history.season.eq(season)]
        f=model.predict(test[KEY],history)
        truth=component_history(test)[KEY+['receiving_component','rushing_component']]
        f=f.merge(truth,on=KEY,validate='one_to_one')
        denom=f.receiving_component+f.rushing_component
        positive=f.loc[denom>0]
        error=(positive.receiving_points_fraction-positive.receiving_component/(positive.receiving_component+positive.rushing_component)).abs()
        diagnostics.append({'validation_season':season,'train_through_season':season-1,
                            'rows':len(f),'positive_component_rows':len(positive),'undefined_zero_component_rows':int((denom==0).sum()),
                            'fraction_mae':float(error.mean()),'cold_start_rows':int(f.history_games.eq(0).sum()),
                            'prior_receiving_fraction':model.receiving_prior/(model.receiving_prior+model.rushing_prior),
                            'status':'development_auxiliary_not_PPR_accuracy'})
    return diagnostics


def summarize_draws(rows):
    # Group in generator insertion order, then release draw dictionaries promptly.
    frame=pd.DataFrame(rows,columns=['player_id','fantasy_points'])
    return frame.groupby('player_id',sort=False).fantasy_points.mean()


def run(root, output):
    manifest=checked_inputs(root)
    if output.exists():
        raise ValueError('output exists; use a new directory to preserve evidence')
    output.mkdir(parents=True)
    cohort,games,excluded,offered=prepare_cohort(root,manifest)
    history=pd.read_csv(root/'inputs/rb_history_through_2025.csv')
    model=fit_rb_receiving_fraction(history,train_through_season=CONFIG['final_train_through'],**CONFIG['candidate'])
    write_json(output/'selected_model.json',model.to_dict())
    write_json(output/'config.json',dict(CONFIG,scoring=SCORING))
    cohort.to_csv(output/'eligible_cohort.csv',index=False)
    excluded.to_csv(output/'exclusions.csv',index=False)
    freeze={'protocol_sha256':sha(root/'protocol.md'),'baseline_manifest_sha256':sha(root/'baseline_manifest.json'),
            'code_sha256':{name:sha(ROOT/name) for name in SOURCES},
            'model_sha256':sha(output/'selected_model.json'),'config_sha256':sha(output/'config.json'),
            'cohort_sha256':sha(output/'eligible_cohort.csv'),
            'archive_kind':'actual_git_archived_served_forecast; simulators retrospectively reconstructed',
            'confirmation_status':'BLOCKED: future 2026 weeks 4-18 outcomes and weekly archives unavailable',
            'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}}
    write_json(output/'freeze.json',freeze)
    # All candidate parameters, code and eligibility are now frozen, before scoring.
    write_json(output/'fraction_validation.json',fraction_diagnostics(history))
    observed=pd.read_csv(root/'inputs/observed_2026_full_scoring.csv')
    if observed.duplicated(KEY).any():
        raise ValueError('duplicate observed labels')
    raw=observed[list(SCORING)].to_numpy(float)
    if not np.isfinite(raw).all():
        raise ValueError('unknown recorded scoring component')
    observed['actual']=sum(observed[k]*weight for k,weight in SCORING.items())
    scoring_discrepancies=int((observed.actual-observed.fantasy_points).abs().gt(1e-7).sum())
    # Use raw full-PPR calculation; disclose any difference from stored total.
    prior2026=observed.loc[observed.week.lt(2)&observed.position.eq('RB')]
    full_history=pd.concat([history,prior2026],ignore_index=True)
    fractions=model.predict(cohort.loc[cohort.position.eq('RB'),KEY],full_history)
    cohort=cohort.merge(fractions,on=KEY,how='left',validate='one_to_one')
    cohort.loc[cohort.position.ne('RB'),'receiving_role']='not_rb'
    fraction_map=dict(zip(fractions.player_id,fractions.receiving_points_fraction))
    inputs=simulation_inputs_from_predictions(games,cohort)
    adapted_ids=[p.player_id for _,players in inputs.values() for p in players]
    if sorted(adapted_ids)!=sorted(cohort.player_id):
        raise ValueError('adapter lost/duplicated eligible forecasts')
    pinned=load_pinned_baseline(root)
    seed_rows=[]
    legacy_verified=True
    for game_index,(game_id,(game,players)) in enumerate(inputs.items(),start=1):
        # Exact baseline regression on every replay game, independent of scores.
        if pinned.simulate_players(game,players,n_draws=20,seed=42)!=simulate_players(game,players,n_draws=20,seed=42):
            raise ValueError('legacy simulator differs from pinned baseline')
        split=[replace(p,receiving_points_fraction=float(fraction_map[p.player_id])) if p.position=='RB' else p for p in players]
        for seed in CONFIG['simulation_seeds']:
            old=summarize_draws(pinned.simulate_players(game,players,n_draws=CONFIG['draws_per_seed'],seed=seed))
            new=summarize_draws(simulate_players(game,split,n_draws=CONFIG['draws_per_seed'],seed=seed))
            nonrb=[p.player_id for p in players if p.position!='RB']
            if not np.array_equal(old.reindex(nonrb),new.reindex(nonrb)):
                raise ValueError('non-RB forecasts changed')
            seed_rows.extend({'player_id':pid,'season':2026,'week':2,'seed':seed,'game_sim':float(old[pid]),'rb_split':float(new[pid])}
                             for pid in old.index)
        print(f'{game_index}/{len(inputs)} games: {game_id}',flush=True)
    per_seed=pd.DataFrame(seed_rows)
    per_seed.to_csv(output/'predictions_by_seed.csv',index=False,float_format='%.17g')
    means=per_seed.groupby(KEY,as_index=False)[['game_sim','rb_split']].mean()
    panel=cohort.merge(means,on=KEY,validate='one_to_one').rename(columns={'predicted_points':'served'})
    labels=observed.loc[observed.week.eq(2),KEY+['team','opponent','actual']]
    panel=panel.merge(labels,on=KEY+['team','opponent'],how='left',validate='one_to_one')
    panel['outcome_status']=np.where(panel.actual.notna(),'recorded_raw_stats','unresolved_no_stats_not_assumed_zero')
    panel.to_csv(output/'player_week_predictions.csv',index=False,float_format='%.17g')
    result=evaluate_paired_panel(panel,replicates=CONFIG['bootstrap_replicates'],seed=CONFIG['bootstrap_seed'])
    result['coverage']={'archived_forecasts':offered,'pregame_eligible':len(cohort),'excluded_already_played':len(excluded),
                        'missing_predictions':0,'missing_outcomes':int(panel.actual.isna().sum()),
                        'rb_cold_starts':int(fractions.history_games.eq(0).sum()),
                        'raw_scoring_vs_stored_total_discrepancies':scoring_discrepancies,
                        'legacy_exact_regression_all_games':legacy_verified}
    rb=panel.loc[panel.position.eq('RB')&panel.actual.notna(),KEY+['actual','served']]
    seed_mae=[]
    for seed,part in per_seed.groupby('seed'):
        data=part.merge(rb,on=KEY,validate='one_to_one')
        mae={arm:float((data[arm]-data.actual).abs().mean()) for arm in ['served','game_sim','rb_split']}
        seed_mae.append({'seed':int(seed),**mae,'delta_vs_sim':mae['rb_split']-mae['game_sim'],'delta_vs_served':mae['rb_split']-mae['served']})
    wide=per_seed.pivot(index=KEY,columns='seed',values=['game_sim','rb_split'])
    s0,s1=CONFIG['simulation_seeds']
    new_diff=(wide['rb_split'][s0]-wide['rb_split'][s1]).abs()
    paired_change=((wide['rb_split'][s0]-wide['game_sim'][s0])-(wide['rb_split'][s1]-wide['game_sim'][s1])).abs()
    result['monte_carlo']={'draws_per_seed':CONFIG['draws_per_seed'],'seeds':CONFIG['simulation_seeds'],
                            'total_draws':CONFIG['draws_per_seed']*len(CONFIG['simulation_seeds']),
                            'mean_abs_between_seed_forecast_change':float(new_diff.mean()),
                            'max_abs_between_seed_forecast_change':float(new_diff.max()),
                            'mean_abs_between_seed_paired_change':float(paired_change.mean()),
                            'max_abs_between_seed_paired_change':float(paired_change.max()),
                            'observed_rb_mae_by_seed':seed_mae}
    result['limitations']=['Not untouched confirmation: archived week-2 development replay.',
                          'Missing eligible outcomes prevent primary full-cohort metrics; observed-row metrics are descriptive only.',
                          'One calendar-week block cannot support the required bootstrap confidence intervals.',
                          'Historical component stats are retrospective snapshots; original correction vintages are unavailable.',
                          'Confirmation requires future 2026 weeks 4-18 pregame archives and complete independently resolved outcomes.']
    write_json(output/'comparison.json',result)
    pd.DataFrame(result['metrics']).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(result['comparisons']).to_csv(output/'comparisons.csv',index=False)
    write_report(output,result)
    if any(sha(ROOT/name)!=digest for name,digest in freeze['code_sha256'].items()):
        raise ValueError('code changed while experiment was running')
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'artifact_manifest.json',{'files':files,'decision':result['decision']})
    print(json.dumps({k:result[k] for k in ['decision','confirmation_status','eligible_rows','observed_rows','missing_outcomes']},indent=2))


def write_report(output,result):
    text=['# RB split experiment: INCONCLUSIVE','',
          'Production simulation remains explicitly disabled. No validated accuracy gain is claimed.', '',
          'This is a development replay of genuine archived served forecasts; both simulators are retrospective reconstructions. '
          'The full eligible cohort is retained in player_week_predictions.csv. Unknown outcomes are never filled with zero.', '',
          f"Eligible player-weeks: {result['eligible_rows']}; observed: {result['observed_rows']}; unresolved: {result['missing_outcomes']}.", '',
          '## Descriptive observed-outcome metrics (not acceptance evidence)', '',
          '| Population | Arm | Observed / eligible | MAE | RMSE |', '|---|---|---:|---:|---:|']
    for row in result['metrics']:
        if row['observed_n']:
            text.append(f"| {row['slice']} | {row['arm']} | {row['observed_n']} / {row['eligible_n']} | {row['mae']:.5f} | {row['rmse']:.5f} |")
    text+=['','| Population | Baseline | Updated MAE difference | Improvement | 95% interval |','|---|---|---:|---:|---|']
    for row in result['comparisons']:
        if row['delta_mae'] is not None:
            text.append(f"| {row['slice']} | {row['baseline']} | {row['delta_mae']:+.5f} | {row['relative_improvement_pct']:.3f}% | unavailable: one week |")
    text+=['','## Decision and missing prerequisites','',*['- '+s for s in result['limitations']], '',
           'Acceptance gates were not evaluated. Confirmation is BLOCKED, not FAIL or PASS. '
           'Freeze weekly pregame served/game inputs under the saved protocol, resolve every eligible outcome, '
           'and evaluate the fixed window once after completion. Do not tune this model against those results.', '',
           '## Reproduce','',
           '`python3 scripts/evaluate_rb_split.py --output-dir /tmp/rb-split-replay`','',
           '`python3 scripts/evaluate_rb_split.py --verify data/experiments/rb_split_20260927/results`','',
           'config.json, selected_model.json and freeze.json pin seeds, fitted parameters, code and protocol. '
           'artifact_manifest.json pins every deliverable. comparison.json includes Monte Carlo diagnostics and coverage.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')


def verify(output):
    manifest=json.loads((output/'artifact_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(output/name)!=digest:
            raise ValueError(f'output hash mismatch: {name}')
    freeze=json.loads((output/'freeze.json').read_text())
    for name,digest in freeze['code_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'code hash mismatch: {name}')
    checked_inputs(EXPERIMENT)
    if sha(EXPERIMENT/'protocol.md')!=freeze['protocol_sha256'] or sha(EXPERIMENT/'baseline_manifest.json')!=freeze['baseline_manifest_sha256']:
        raise ValueError('protocol or baseline manifest changed')
    panel=pd.read_csv(output/'player_week_predictions.csv',float_precision='round_trip')
    recomputed=evaluate_paired_panel(panel,replicates=CONFIG['bootstrap_replicates'],seed=CONFIG['bootstrap_seed'])
    saved=json.loads((output/'comparison.json').read_text())
    for key in recomputed:
        if json.dumps(recomputed[key],sort_keys=True)!=json.dumps(saved[key],sort_keys=True):
            raise ValueError(f'recomputed metric mismatch: {key}')
    print(f'Verified frozen inputs/code and recomputed paired metrics: {saved["decision"]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if bool(args.output_dir)==bool(args.verify):
        parser.error('choose exactly one of --output-dir or --verify')
    if args.verify:
        verify(args.verify)
    else:
        run(EXPERIMENT,args.output_dir)


if __name__=='__main__':
    main()
