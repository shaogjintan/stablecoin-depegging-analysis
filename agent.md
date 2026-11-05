# Repository working instructions

Read README.md and docs/reproduction.md for current entry points. The final runner is modelling.refit_after_calibration, not the historical refit_final implementation.

Preserve corrected input hashes, model parameters, saved evidence and statistical definitions. Use 25h purge and 24h embargo. LR keeps practice-selected C for final refit. Do not fetch network data during normal reproduction. Do not mix archived reports with final results.

No files may be permanently deleted during cleanup: move deletion candidates under delete/ preserving their original relative paths. Keep dune_onchain.zip. Keep source snapshots and original manifests immutable. Record relocations. Do not stage large model/partition artifacts or overwrite existing training-run directories.
