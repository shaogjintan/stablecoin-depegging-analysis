"""Coverage over each coin's full observed master span, with explicit numerators."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_processing.config import ACTIVE_MASTER_PATH, FIAT_COINS
from modelling.corrected_xgb import sha256

RAW_ONCHAIN = ['tx_count', 'tx_volume', 'active_senders', 'active_receivers']

def coverage(master):
    rows = []
    for coin in FIAT_COINS:
        g = master.loc[master.coin.eq(coin)].sort_values('hour')
        if g.empty:
            continue
        if g.hour.duplicated().any():
            raise ValueError(f'Duplicate hours: {coin}')
        expected = int((g.hour.max() - g.hour.min()) / pd.Timedelta(hours=1)) + 1
        masks = {
            'market_volume': g.trading_volume.notna(),
            'onchain_all_raw_fields': g[RAW_ONCHAIN].notna().all(axis=1),
            'joint_volume_and_onchain': g[['trading_volume', *RAW_ONCHAIN]].notna().all(axis=1),
            'onchain_tx_count_common_sample': g.tx_count.notna() & g.common_sample.fillna(False).astype(bool),
        }
        for name, mask in masks.items():
            observed = int(mask.sum())
            rows.append(dict(coin=coin, coverage_definition=name, start=g.hour.min(), end=g.hour.max(),
                             expected_hours=expected, observed_hours=observed,
                             coverage_pct=100 * observed / expected,
                             denominator='full first-to-last timestamp span in corrected master'))
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=Path('reports/data_coverage'))
    args = p.parse_args()
    master = pd.read_parquet(ACTIVE_MASTER_PATH)
    master['hour'] = pd.to_datetime(master.hour, utc=True)
    result = coverage(master)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / 'coverage.csv', index=False)
    (args.out_dir / 'manifest.json').write_text(json.dumps(dict(source=str(ACTIVE_MASTER_PATH),
        sha256=sha256(ACTIVE_MASTER_PATH), note='Source completeness, not model-feature eligibility. All definitions use the same full-span denominator.'), indent=2)+'\n')
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()
