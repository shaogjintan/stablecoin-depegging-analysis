"""Materialise corrected_v1 from the original master; never correct it twice.

From the repository root: python -m modelling.build_corrected_dataset
"""
import hashlib
import json
from pathlib import Path
import pandas as pd
from data_processing.config import ACTIVE_MASTER_PATH, ACTIVE_EPISODES_PATH
from modelling.corrected_data import prepare_corrected


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path("data/processed/master.parquet")
    raw = pd.read_parquet(source)
    if "bar_start" in raw.columns or "observable_at_risk" in raw.columns:
        raise ValueError("Input already looks corrected; refusing to shift clocks twice.")
    master, episodes = prepare_corrected(raw)
    assert not master.duplicated(["coin", "hour"]).any()
    assert (master.hour - master.bar_start).eq(pd.Timedelta(hours=1)).all()
    master["dataset_version"] = "corrected_v1"
    episodes["dataset_version"] = "corrected_v1"
    ACTIVE_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(ACTIVE_MASTER_PATH, index=False)
    episodes.to_parquet(ACTIVE_EPISODES_PATH, index=False)
    counts = {}
    for tier, name in [(1, "market"), (2, "market_onchain")]:
        for horizon in (1, 6, 24):
            mask = master[f"prediction_eligible_{name}_h{horizon}"]
            counts[f"tier{tier}_h{horizon}"] = {
                "eligible_rows": int(mask.sum()),
                "positive_hours": int(master.loc[mask, f"label_h{horizon}"].sum()),
            }
    metadata = {
        "dataset_version": "corrected_v1",
        "source": str(source), "source_sha256": checksum(source),
        "master": str(ACTIVE_MASTER_PATH), "master_sha256": checksum(ACTIVE_MASTER_PATH),
        "episodes": str(ACTIVE_EPISODES_PATH), "episodes_sha256": checksum(ACTIVE_EPISODES_PATH),
        "preparation_sha256": checksum(Path("modelling/corrected_data.py")),
        "rows": len(master), "episodes_defined": len(episodes),
        "target": "new merged episode starts in (origin, origin+h]",
        "clock": "bar close = source bar start + 1 hour",
        "recovery": "exclude through 24 hours after recovery anchor of confirmed two-hour dip",
        "counts": counts,
    }
    Path("data/processed/corrected_v1_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
