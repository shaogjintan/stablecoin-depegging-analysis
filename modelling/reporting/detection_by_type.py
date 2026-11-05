"""Count final-model detections by structural event type using saved eligible origins."""
import json
from pathlib import Path
import pandas as pd
from modelling.robustness_refit import targetable_ids
from modelling.corrected_xgb import sha256
RUNS = [Path('artifacts/runs') / kind / name for kind, name in [('primary', 'REFIT-LOWEST-NEW'), ('primary', 'LR-REFIT-LOWEST-NEW'), ('supplementary', 'REFIT-LOWEST-H1'), ('supplementary', 'LR-REFIT-LOWEST-H1')]]

def main():
    out = Path('reports/final/events')
    eps = pd.read_csv(out / 'episodes.csv')
    types = pd.read_csv(out / 'types_events.csv')
    eps = eps.merge(types[['event_id', 'type']], on='event_id', validate='many_to_one')
    rows = []
    sources = []
    for run in RUNS:
        info = json.loads((run / 'run.json').read_text())
        sources.append(dict(path=str(run), run_id=info['run_id'], sha256=sha256(run / 'run.json')))
        episodes = pd.read_parquet(run / 'episodes.parquet')
        alarms = pd.read_csv(run / 'alarm_events.csv')
        alarms['alarm_hour'] = pd.to_datetime(alarms.alarm_hour, utc=True)
        if set(episodes.episode_id.astype(str)) != set(eps.episode_id.astype(str)):
            raise ValueError('Episode identity mismatch')
        for model in info['models']:
            for config in info['configs']:
                for h in info['horizons']:
                    frames = [pd.read_parquet(run / 'fits' / f'{model}_{config}_h{h}_{y}' / 'test.parquet', columns=['coin', 'hour']) for y in info['years']]
                    targets = targetable_ids(pd.concat(frames, ignore_index=True), episodes, h)
                    a = alarms.loc[alarms.model.eq(model) & alarms.config.eq(config) & alarms.horizon_hours.eq(h)]
                    truth = a.is_true_alarm.astype(str).str.lower().isin(['true', '1'])
                    caught = set(a.loc[truth, 'target_episode_id'].dropna().astype(str)) & targets
                    for typ, g in eps.groupby('type'):
                        ids = set(g.episode_id.astype(str))
                        eligible = ids & targets
                        hit = ids & caught
                        leads = a.loc[truth & a.target_episode_id.astype(str).isin(hit), 'lead_hours']
                        rows.append(dict(model=model, config=config, horizon_hours=h, type=typ, targetable_episodes=len(eligible), caught_episodes=len(hit), missed_episodes=len(eligible - hit), caught_events=g.loc[g.episode_id.astype(str).isin(hit), 'event_id'].nunique(), median_lead_hours=leads.median()))
    pd.DataFrame(rows).to_csv(out / 'caught_by_type.csv', index=False)
    (out / 'detection_manifest.json').write_text(json.dumps(dict(sources=sources, targetability='any eligible origin in [onset-h,onset); counts only'), indent=2) + '\n')
    print('Saved detection counts:', len(rows))
if __name__ == '__main__':
    main()
