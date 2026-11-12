"""Build canonical corrected tables from retained source and feature tables.

Run ``python -m data_processing.build_dataset`` to validate existing outputs.
Add ``--rebuild`` to compare a reconstruction, or ``--out-dir`` to write a fresh
pair. Existing canonical files are never silently replaced.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from data_processing.config import (
    ACTIVE_EPISODES_PATH,
    ACTIVE_MASTER_PATH,
    DATASET_METADATA_PATH,
    HOURLY_MARKET_PATH,
    FORECAST_HORIZONS_HOURS,
    MARKET_FEATURES_PATH,
    MODELLING_DATA_VERSION,
    ONCHAIN_FEATURES_PATH,
    PROCESSED_DATA_DIR,
)
from data_processing.merge_and_label import flag_feature_samples, merge_sources
from data_processing.splits import load_master
from data_processing.dataset_transformation import prepare_corrected
from data_processing.build_market_features import VOLATILITY_METHOD
from data_processing.feature_schema import ONCHAIN_FEATURES, ONCHAIN_RAW_METRICS

SOURCES = [HOURLY_MARKET_PATH, ONCHAIN_FEATURES_PATH, MARKET_FEATURES_PATH]
SOURCE_MARKET_FEATURES = [
    "peg_deviation",
    "log_return_1h",
    "log_return_6h",
    "log_return_24h",
    "realised_vol_6h",
    "realised_vol_24h",
    "volume_pct_change_1h",
    "volume_pct_change_6h",
    "volume_pct_change_24h",
]
# Keep output column order stable without reading a historical master file.
MASTER_COLUMNS = [
    "hour",
    "coin",
    "price",
    "venue",
    "venue_switch",
    "hours_since_venue_start",
    *ONCHAIN_RAW_METRICS,
    *ONCHAIN_FEATURES,
    "peg_deviation",
    "peg_deviation_bps",
    "return_1h",
    "log_return_1h",
    "return_6h",
    "log_return_6h",
    "return_24h",
    "log_return_24h",
    "realised_vol_6h",
    "realised_vol_24h",
    "trading_volume",
    "volume_pct_change_1h",
    "volume_pct_change_6h",
    "volume_pct_change_24h",
    "outside_band",
    "episode_id",
    "qualifying_depeg_hour",
    "qualifying_dip_start",
    "episode_end_hour",
    "in_episode",
    "label_h1",
    "label_h6",
    "label_h24",
    "market_sample",
    "common_sample",
    "prediction_eligible_market",
    "prediction_eligible_market_onchain",
    "prediction_eligible_market_h1",
    "prediction_eligible_market_onchain_h1",
    "prediction_eligible_market_h6",
    "prediction_eligible_market_onchain_h6",
    "prediction_eligible_market_h24",
    "prediction_eligible_market_onchain_h24",
    "peg_stress",
    "observable_at_risk",
    "bar_start",
    "dataset_version",
]

PIPELINE_DESCRIPTION = (
    "configured raw root (hourly_stablecoin.parquet) + interim root ("
    "hourly_stablecoin_market_features.parquet + "
    "hourly_stablecoin_onchain_features.parquet) -> merge on coin/hour "
    "-> compute calendar-based market features, merged-onset labels and "
    "causal prediction eligibility -> shift timestamps to completed-bar "
    "close -> master.parquet + depeg_episodes.parquet in the output "
    "directory"
)


def checksum(path: Path) -> str:
    """Return the SHA-256 digest of a retained input or output file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_from_sources(sources=SOURCES):
    """Merge retained source tables and prepare corrected outputs in memory.

    Args:
        sources: Hourly observations, on-chain features and market features,
            in that order.

    Returns:
        The canonical master table and matching episode manifest.
    """
    price, onchain, market = [pd.read_parquet(path) for path in sources]
    price = price.rename(columns={"timestamp": "hour"})
    merged = merge_sources(price, onchain, market)
    master, episodes = prepare_corrected(merged)
    # Completeness describes current features, separately from eligibility.
    master = flag_feature_samples(
        master, SOURCE_MARKET_FEATURES, ONCHAIN_FEATURES
    )
    master["venue_switch"] = pd.to_numeric(master.venue_switch).astype(float)
    master["dataset_version"] = MODELLING_DATA_VERSION
    episodes["dataset_version"] = MODELLING_DATA_VERSION
    master = master[MASTER_COLUMNS].reset_index(drop=True)
    episodes = episodes.reset_index(drop=True)
    if master.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate canonical coin-hour")
    if not (master.hour - master.bar_start).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Incorrect completed-bar clock")
    return master, episodes


def assert_rebuild_matches(rebuilt, frozen, *, allow_historical_diagnostics=False,
                           stable_volatility=False):
    """Compare rebuilds, optionally reporting historical diagnostic changes.

    Args:
        rebuilt: Current reconstructed master.
        frozen: Retained master used for the reported model results.
        allow_historical_diagnostics: Allow named historical completeness and
            simple-return diagnostics to differ; print each changed count.
        stable_volatility: Require exact volatility for the window-local method.
    """
    diagnostics = []
    if allow_historical_diagnostics:
        diagnostics = ["market_sample", "common_sample", "return_1h",
                       "return_6h", "return_24h"]
        for column in diagnostics:
            same = rebuilt[column].eq(frozen[column]) | (
                rebuilt[column].isna() & frozen[column].isna()
            )
            print(f"Historical diagnostic {column}: "
                  f"{int((~same).sum())} changed rows")
    volatility = ["realised_vol_6h", "realised_vol_24h"]
    pd.testing.assert_frame_equal(
        rebuilt.drop(columns=volatility + diagnostics),
        frozen.drop(columns=volatility + diagnostics),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        rebuilt[volatility], frozen[volatility],
        check_exact=stable_volatility, rtol=0, atol=1e-9,
    )


def metadata(master, episodes, master_path, episodes_path):
    """Describe source hashes, output hashes and model eligibility counts."""
    counts = {}
    for tier, name in [(1, "market"), (2, "market_onchain")]:
        for horizon in FORECAST_HORIZONS_HOURS:
            mask = master[f"prediction_eligible_{name}_h{horizon}"]
            counts[f"tier{tier}_h{horizon}"] = {
                "eligible_rows": int(mask.sum()),
                "positive_hours": int(
                    master.loc[mask, f"label_h{horizon}"].sum()
                ),
            }
    return dict(
        volatility_method=VOLATILITY_METHOD,
        sources=[
            dict(path=str(path), sha256=checksum(path)) for path in SOURCES
        ],
        pipeline=PIPELINE_DESCRIPTION,
        master=str(master_path),
        master_sha256=checksum(master_path),
        episodes=str(episodes_path),
        episodes_sha256=checksum(episodes_path),
        preparation_source="data_processing/dataset_transformation.py",
        preparation_sha256=checksum(
            Path(__file__).with_name("dataset_transformation.py")
        ),
        market_features_source="data_processing/build_market_features.py",
        market_features_sha256=checksum(
            Path(__file__).with_name("build_market_features.py")
        ),
        rows=len(master),
        episodes_defined=len(episodes),
        counts=counts,
        clock="bar close = source bar start + 1 hour",
        target="new merged episode starts in (origin, origin+h]",
        recovery=(
            "exclude through 24 hours after recovery anchor "
            "of confirmed two-hour dip"
        ),
    )


def main():
    """Validate canonical inputs or build a fresh pair from interim tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild from source and feature tables; compare if outputs already exist.",
    )
    args = parser.parse_args()
    master_path = args.out_dir / ACTIVE_MASTER_PATH.name
    episodes_path = args.out_dir / ACTIVE_EPISODES_PATH.name
    exists = [path.exists() for path in [master_path, episodes_path]]
    if any(exists) and not all(exists):
        raise ValueError(
            "Incomplete canonical pair; use a fresh --out-dir to rebuild both"
        )
    if all(exists) and not args.rebuild:
        if args.out_dir.resolve() != PROCESSED_DATA_DIR.resolve():
            raise ValueError(
                "Use --rebuild to validate a custom output directory"
            )
        master, episodes = load_master()
        print(
            f"Validated canonical inputs: {len(master)} rows, "
            f"{len(episodes)} episodes; no files overwritten."
        )
        return

    master, episodes = build_from_sources()
    if all(exists):
        metadata_path = args.out_dir / DATASET_METADATA_PATH.name
        prior = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        stable = prior.get("volatility_method") == VOLATILITY_METHOD
        assert_rebuild_matches(
            master, pd.read_parquet(master_path),
            allow_historical_diagnostics=not stable, stable_volatility=stable,
        )
        pd.testing.assert_frame_equal(
            episodes, pd.read_parquet(episodes_path), check_exact=True
        )
        print(
            "Rebuild model inputs match "
            + ("exactly" if stable else "within historical numerical tolerance")
            + "; existing files preserved."
        )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    master.to_parquet(master_path, index=False)
    episodes.to_parquet(episodes_path, index=False)
    metadata_path = args.out_dir / DATASET_METADATA_PATH.name
    metadata_path.write_text(
        json.dumps(metadata(master, episodes, master_path, episodes_path),
                   indent=2) + "\n"
    )
    print(
        f"Wrote {len(master)} rows to {master_path}; "
        f"{len(episodes)} episodes to {episodes_path}"
    )


if __name__ == "__main__":
    main()
