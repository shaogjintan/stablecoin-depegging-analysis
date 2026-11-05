# Run evidence

primary/ holds final RF/XGB and LR h6/h24 refits. supplementary/ holds h1 refits. sensitivity/ holds non-refit RF/XGB results plus their XGB source. Existing names and original metadata are preserved; historical path fields should be resolved through docs/cleanup_moves.json.

Models, prediction Parquets, calibration partitions and source snapshots remain local and ignored by Git. A Git clone alone is insufficient for saved-run verification. No public artifact download has been published. Use the supplied local artifacts, package them separately with checksums, or retrain using docs/reproduction.md. Do not describe summaries alone as a full artifact backup.

See index.json for manifests and verification scope. Final-model verification is separate from sensitivity-run verification. Run status=complete alone is not independent verification.
