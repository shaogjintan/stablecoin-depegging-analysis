"""Compatibility entry point for data_processing.build_dataset.

Prefer ``python -m data_processing.build_dataset`` for new workflows.
"""

from data_processing.build_dataset import (
    MASTER_COLUMNS,
    SOURCE_MARKET_FEATURES,
    SOURCES,
    assert_rebuild_matches,
    build_from_sources,
    checksum,
    main,
    metadata,
)

if __name__ == "__main__":
    main()
