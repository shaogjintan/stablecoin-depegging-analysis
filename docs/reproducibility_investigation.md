# Numerical reproducibility

The published results use `artifacts/final_run/` and the canonical inputs in
`data/processed/`, selected by `reports/baseline.json`. Two independent full
source-data reproductions matched their scientific outputs in the tested
software environment. See [the comparison](stable_reproduction_comparison.md)
and [execution instructions](reproduction.md).

## Volatility calculations

An earlier processed dataset contained tiny rolling-volatility residuals in
constant-price windows. Independent calculations, including high-precision
arithmetic, confirmed that the largest discrepancies were numerical residuals:
constant prices imply zero returns and zero volatility. The precise historical
execution condition that produced them could not be established.

Although the differences were below 1e-9, some survived conversion to float32
and changed fitted tree splits. Controlled refits on one representative fold
isolated volatility inputs as the source of the model differences. Matching
random seeds alone therefore did not reproduce the earlier RF/XGBoost results.

`data_processing/build_market_features.py` now calculates each complete window's
sample standard deviation independently, centering its values and using
`math.fsum` for compensated summation. Constant windows return exactly zero;
genuine small variation is preserved without rounding or a numerical cutoff.
The interim writer and dataset builder use this shared calculation.

`data_processing/build_dataset.py` records the volatility method in dataset
metadata and requires exact feature equality when checking a current rebuild.
A compatibility tolerance applies only to earlier, unmarked datasets; the
published inputs do not rely on it.

## Random-forest prediction

Parallel accumulation of tree probabilities can change the last floating-point
bits. `modelling/models/random_forest_model.py` uses one prediction worker to
sum trees in a fixed order, restoring the configured worker count afterwards.
Fitting retains its configured parallelism and random seed.

## Validation and limits

Two complete rebuild-and-training workflows passed all 90 model-score audit
cells and matched across 490 comparison checks. Scientific tables were compared
exactly, with expected run identifiers and workspace paths checked separately.
Image bytes and arbitrary hardware or package versions were not validated.

The selected run, canonical datasets and current reports form one consistent
baseline. Earlier duplicate inputs, runs and report snapshots are no longer
required or retained. Small comparison and checksum records remain under
`reports/validation_checks/reproducibility/`. Original training metadata and
source snapshots retain their execution-time paths for provenance; explicit
relocations resolve the published inputs without rewriting that evidence.
