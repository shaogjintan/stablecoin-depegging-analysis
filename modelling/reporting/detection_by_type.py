"""Report saved-model episode detections by structural event type.

Count final-model detections by structural event type using saved eligible
origins.
"""
import argparse
import json
from pathlib import Path
from modelling.utils.artifacts import evidence_dir, FINAL_RUNS
import pandas as pd
from modelling.evaluation.targetability import targetable_ids
from modelling.utils.io import sha256
RUNS = FINAL_RUNS

def main():
    """Export model detection counts grouped by structural event type."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-dir', type=Path, default=Path('reports/depeg_events'))
    parser.add_argument('--out-dir', type=Path, default=Path('reports/model_results'))
    parser.add_argument('--run-dir', type=Path, action='append')
    args = parser.parse_args()
    event_dir, out = args.event_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(event_dir / 'episode_event_mapping.csv')
    types = pd.read_csv(event_dir / 'event_classification.csv')
    eps = eps.merge(types[['event_id', 'type']], on='event_id', validate='many_to_one')
    rows = []
    caught_rows = []
    context_rows = []
    sources = []
    for run in args.run_dir or RUNS:
        info = json.loads((run / 'run.json').read_text())
        sources.append(dict(path=str(run), run_id=info['run_id'], sha256=sha256(run / 'run.json')))
        episodes = pd.read_parquet(evidence_dir(run) / 'episodes.parquet')
        alarms = pd.read_csv(run / 'alarm_events.csv')
        alarms['alarm_hour'] = pd.to_datetime(alarms.alarm_hour, utc=True)
        if set(episodes.episode_id.astype(str)) != set(eps.episode_id.astype(str)):
            raise ValueError('Episode identity mismatch')
        for model in info['models']:
            for config in info['configs']:
                for h in info['horizons']:
                    frames = [pd.read_parquet(evidence_dir(run) / 'fits' / f'{model}_{config}_h{h}_{y}' / 'test.parquet', columns=['coin', 'hour']) for y in info['years']]
                    targets = targetable_ids(pd.concat(frames, ignore_index=True), episodes, h)
                    a = alarms.loc[alarms.model.eq(model) & alarms.config.eq(config) & alarms.horizon_hours.eq(h)]
                    truth = a.is_true_alarm.astype(str).str.lower().isin(['true', '1'])
                    caught = set(a.loc[truth, 'target_episode_id'].dropna().astype(str)) & targets
                    # Preserve episode identities for descriptive case analysis.
                    cases = eps.loc[eps.episode_id.astype(str).isin(caught)].copy()
                    cases['model'] = model
                    cases['config'] = config
                    cases['horizon_hours'] = h
                    cases['run_id'] = info['run_id']
                    leads_by_episode = a.loc[truth].groupby(
                        'target_episode_id')['lead_hours'].max()
                    cases['first_alarm_lead_hours'] = (
                        cases.episode_id.map(leads_by_episode)
                    )
                    caught_rows.append(cases)
                    for context, group in eps.groupby('historical_context'):
                        ids = set(group.episode_id.astype(str))
                        eligible, hit = ids & targets, ids & caught
                        context_rows.append(dict(
                            model=model, config=config, horizon_hours=h,
                            historical_context=context,
                            targetable_episodes=len(eligible),
                            caught_episodes=len(hit),
                            missed_episodes=len(eligible - hit),
                            episode_recall=(len(hit) / len(eligible)
                                            if eligible else float('nan')),
                            caught_events=group.loc[
                                group.episode_id.astype(str).isin(hit),
                                'event_id'].nunique(),
                        ))
                    for typ, g in eps.groupby('type'):
                        ids = set(g.episode_id.astype(str))
                        eligible = ids & targets
                        hit = ids & caught
                        leads = a.loc[truth & a.target_episode_id.astype(str).isin(hit), 'lead_hours']
                        rows.append(dict(model=model, config=config, horizon_hours=h, type=typ, targetable_episodes=len(eligible), caught_episodes=len(hit), missed_episodes=len(eligible - hit), caught_events=g.loc[g.episode_id.astype(str).isin(hit), 'event_id'].nunique(), median_lead_hours=leads.median()))
    pd.DataFrame(rows).to_csv(out / 'model_comparison_by_event_type.csv', index=False)
    pd.concat(caught_rows, ignore_index=True).to_csv(
        out / 'caught_episodes.csv', index=False)
    pd.DataFrame(context_rows).to_csv(
        out / 'detection_by_historical_context.csv', index=False)

    (out / 'detection_manifest.json').write_text(json.dumps(dict(sources=sources, targetability='any eligible origin in [onset-h,onset); counts only'), indent=2) + '\n')
    print('Saved detection counts:', len(rows))
if __name__ == '__main__':
    main()
