# Frozen research data

processed/ contains the canonical corrected_v1 master, episode manifest, typed manifest and original preparation metadata. snapshots/ preserves the original pre-correction master and acquired hourly market table. interim/ retains market/on-chain feature intermediates until rebuild equivalence is established. raw/dune_onchain/ contains six source exports; the root dune_onchain.zip is retained too.

manifest.json records checksums and schemas. The preparation metadata retains historical paths; docs/cleanup_moves.json records relocations. Missingness is preserved; do not fabricate zero activity. The corrected dataset builder validates existing inputs and only rebuilds from the original snapshot if needed. Original market API responses are not available, so exact reproduction starts from frozen snapshots.

The original master snapshot and canonical corrected inputs are explicitly allowed in Git. The acquired hourly snapshot, raw CSVs and intermediates remain local artifacts and need separate distribution for acquisition-level inspection.

All six extracted Dune CSVs were checked byte-for-byte against their matching ZIP members; see dune_export_comparison.json. Both forms are preserved.
