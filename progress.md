# Progress

Date: 2026-05-08

## Completed

- Added a minority-sample reliability guard in `src/evaluation/calibration_metrics.py`.
- Extended calibration metrics with multiclass macro class-conditional ECE support.
- Extended classification metrics with multiclass support for the Dry Bean path.
- Added a regression test for sparse minority ECE handling.
- Added a regression test for multiclass macro class-conditional calibration.
- Added a regression test for multiclass classification metrics.
- Confirmed the calibration metrics test file passes in `conda activate py312`.
- Confirmed the full repository test suite passes in `conda activate py312`.
- Added real-dataset configs for Mammography, NSL-KDD, Give Me Some Credit, Bank Marketing, Thyroid Disease, Dry Bean, and UCI Default of Credit Card Clients.
- Removed the proxy credit-card narrative from the active README and experiment helper scripts.

## In Progress

- Reviewing any remaining generated outputs that still mention the old `credit_card` proxy dataset.

## Next

- Optionally regenerate paper outputs from the updated source if the downstream tables need to be refreshed.