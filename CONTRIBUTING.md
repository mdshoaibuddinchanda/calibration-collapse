# Contributing

Thank you for your interest in contributing to this research framework.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run the test suite: `python -m pytest tests/ -v`
5. Submit a pull request

## Code Standards

### Style

- Python 3.12+
- Type hints on all public functions
- Docstrings on all classes and public methods
- Line length: 100 characters max

### Anti-Leakage Rules

This is a research framework. The most important rule is:

> **Never fit any statistic on validation or test data.**

Specifically:

- `PreprocessingPipeline.fit()` — only called with `X_train`
- `BaseCalibrator.fit()` — only called with `split_tag='val'`
- Resamplers — only called after splitting, never on val/test

Any contribution that violates these rules will not be merged.

### Adding a New Resampler

1. Create `src/resampling/your_resampler.py` implementing `BaseResampler`
2. Implement `fit_resample(X_train, y_train)` and `get_params()`
3. Register in `src/resampling/__init__.py`
4. Add feasibility validation if the resampler has geometric constraints

### Adding a New Calibrator

1. Create `src/calibration/your_calibrator.py` implementing `BaseCalibrator`
2. Call `self._enforce_val_split(split_tag)` at the start of `fit()`
3. Register in `src/calibration/__init__.py`

### Adding a New Dataset

1. Place CSV in `datasets/raw/your_dataset.csv`
2. Create `configs/datasets/your_dataset.yaml` following `_template.yaml`
3. Run `python scripts/inspect_dataset.py --dataset your_dataset`

### Adding a New Metric

1. Add function to `src/evaluation/calibration_metrics.py`
2. Add mathematical derivation to `reports/math/ece_formulation.md`
3. Add unit test with known-correct values

## Testing

All contributions must include tests. Run the full suite before submitting:

```bash
python -m pytest tests/ -v --tb=short
```

Tests must cover:
- Correct output values (not just "no crash")
- Edge cases (empty bins, extreme IR, single-class splits)
- Anti-leakage assertions

## Experiment Run Status

When adding new experiment logic, use the correct status values:

| Status | When to use |
|---|---|
| `completed` | Run finished successfully with valid results |
| `skipped` | Configuration is geometrically/logically invalid — raise `InvalidSamplingStrategyError` or similar |
| `failed` | Unexpected error — should be investigated and fixed |

Do not use `failed` for expected invalid configurations. Use `skipped`.

## Pull Request Checklist

- [ ] Tests pass: `python -m pytest tests/ -v`
- [ ] No new leakage vectors introduced
- [ ] Type hints added
- [ ] Docstring added or updated
- [ ] Mathematical derivation documented (for metric changes)
- [ ] `.gitignore` not modified to include generated outputs
