"""Build canonical corrected tables directly from hourly and interim inputs.

Both this CLI and data_processing.merge_and_label use the same in-memory path.
Existing canonical files are validated, never silently replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from data_processing.config import PROCESSED_DATA_DIR, MODELLING_DATA_VERSION
from data_processing.merge_and_label import merge_sources, flag_feature_samples
from modelling.corrected_data import prepare_corrected
from modelling.prepare_data import ONCHAIN_FEATURES

SOURCES = [Path("data/interim/hourly_stablecoin.parquet"),
           Path("data/interim/hourly_stablecoin_onchain_features.parquet"),
           Path("data/interim/hourly_stablecoin_market_features.parquet")]
SOURCE_MARKET_FEATURES = ["peg_deviation", "log_return_1h", "log_return_6h", "log_return_24h",
    "realised_vol_6h", "realised_vol_24h", "volume_pct_change_1h",
    "volume_pct_change_6h", "volume_pct_change_24h"]
# Stable public schema; no dependence on a historical master file.
MASTER_COLUMNS = ['hour',
 'coin',
 'price',
 'venue',
 'venue_switch',
 'hours_since_venue_start',
 'tx_count',
 'tx_volume',
 'active_senders',
 'active_receivers',
 'tx_count_logdiff_1h',
 'tx_count_logdiff_6h',
 'tx_count_logdiff_24h',
 'tx_volume_logdiff_1h',
 'tx_volume_logdiff_6h',
 'tx_volume_logdiff_24h',
 'active_senders_logdiff_1h',
 'active_senders_logdiff_6h',
 'active_senders_logdiff_24h',
 'active_receivers_logdiff_1h',
 'active_receivers_logdiff_6h',
 'active_receivers_logdiff_24h',
 'peg_deviation',
 'peg_deviation_bps',
 'return_1h',
 'log_return_1h',
 'return_6h',
 'log_return_6h',
 'return_24h',
 'log_return_24h',
 'realised_vol_6h',
 'realised_vol_24h',
 'trading_volume',
 'volume_pct_change_1h',
 'volume_pct_change_6h',
 'volume_pct_change_24h',
 'outside_band',
 'episode_id',
 'qualifying_depeg_hour',
 'qualifying_dip_start',
 'episode_end_hour',
 'in_episode',
 'label_h1',
 'label_h6',
 'label_h24',
 'market_sample',
 'common_sample',
 'prediction_eligible_market',
 'prediction_eligible_market_onchain',
 'prediction_eligible_market_h1',
 'prediction_eligible_market_onchain_h1',
 'prediction_eligible_market_h6',
 'prediction_eligible_market_onchain_h6',
 'prediction_eligible_market_h24',
 'prediction_eligible_market_onchain_h24',
 'peg_stress',
 'observable_at_risk',
 'bar_start',
 'dataset_version']


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_from_sources(sources=SOURCES):
    price, onchain, market = [pd.read_parquet(p) for p in sources]
    price = price.rename(columns={"timestamp": "hour"})
    merged = merge_sources(price, onchain, market)
    # Keep source-completeness flags distinct from corrected model eligibility.
    merged = flag_feature_samples(merged, SOURCE_MARKET_FEATURES, ONCHAIN_FEATURES)
    master, episodes = prepare_corrected(merged)
    master["venue_switch"] = pd.to_numeric(master.venue_switch).astype(float)
    master["dataset_version"] = MODELLING_DATA_VERSION
    episodes["dataset_version"] = MODELLING_DATA_VERSION
    master = master[MASTER_COLUMNS].reset_index(drop=True)
    episodes = episodes.reset_index(drop=True)
    if master.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate canonical coin-hour")
    if not (master.hour-master.bar_start).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Incorrect completed-bar clock")
    return master, episodes



def assert_rebuild_matches(rebuilt, frozen):
    """Require exact non-volatility data and bound rolling-reduction noise."""
    volatility = ["realised_vol_6h", "realised_vol_24h"]
    pd.testing.assert_frame_equal(rebuilt.drop(columns=volatility),
                                  frozen.drop(columns=volatility), check_exact=True)
    pd.testing.assert_frame_equal(rebuilt[volatility], frozen[volatility],
                                  check_exact=False, rtol=0, atol=1e-9)


def metadata(master, episodes, master_path, episodes_path):
    counts = {}
    for tier, name in [(1, "market"), (2, "market_onchain")]:
        for horizon in (1, 6, 24):
            mask = master[f"prediction_eligible_{name}_h{horizon}"]
            counts[f"tier{tier}_h{horizon}"] = {
                "eligible_rows": int(mask.sum()),
                "positive_hours": int(master.loc[mask, f"label_h{horizon}"].sum())}
    return dict(sources=[dict(path=str(p), sha256=checksum(p)) for p in SOURCES],
        pipeline="data/interim/ (hourly_stablecoin.parquet + hourly_stablecoin_market_features.parquet + hourly_stablecoin_onchain_features.parquet) -> merge on coin/hour -> compute calendar-based market features, merged-onset labels and causal prediction eligibility -> shift timestamps to completed-bar close -> master.parquet + depeg_episodes.parquet in the output directory",
        master=str(master_path), master_sha256=checksum(master_path),
        episodes=str(episodes_path), episodes_sha256=checksum(episodes_path),
        preparation_sha256=checksum(Path("modelling/corrected_data.py")),
        rows=len(master), episodes_defined=len(episodes), counts=counts,
        clock="bar close = source bar start + 1 hour",
        target="new merged episode starts in (origin, origin+h]",
        recovery="exclude through 24 hours after recovery anchor of confirmed two-hour dip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--rebuild", action="store_true", help="rebuild from interim inputs; compare if outputs already exist")
    args = parser.parse_args()
    master_path, episodes_path = args.out_dir/"master.parquet", args.out_dir/"depeg_episodes.parquet"
    exists = [p.exists() for p in [master_path, episodes_path]]
    if any(exists) and not all(exists):
        raise ValueError("Incomplete canonical pair; use a fresh --out-dir to rebuild both")
    if all(exists) and not args.rebuild:
        if args.out_dir.resolve() != PROCESSED_DATA_DIR.resolve():
            raise ValueError("Use --rebuild to validate a custom output directory")
        from data_processing.splits import load_master
        master, episodes = load_master()
        print(f"Validated canonical inputs: {len(master)} rows, {len(episodes)} episodes; no files overwritten.")
        return
    master, episodes = build_from_sources()
    if all(exists):
        # Floating rolling reductions can differ at machine precision after merging
        # source blocks. Do not replace frozen model inputs on that basis.
        assert_rebuild_matches(master, pd.read_parquet(master_path))
        pd.testing.assert_frame_equal(episodes, pd.read_parquet(episodes_path), check_exact=True)
        print("Direct rebuild matches canonical tables within numerical tolerance; existing files preserved.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    master.to_parquet(master_path, index=False)
    episodes.to_parquet(episodes_path, index=False)
    (args.out_dir/"dataset_metadata.json").write_text(json.dumps(metadata(master,episodes,master_path,episodes_path),indent=2)+"\n")
    print(f"Wrote {len(master)} rows to {master_path}; {len(episodes)} episodes to {episodes_path}")


if __name__ == "__main__":
    main()
