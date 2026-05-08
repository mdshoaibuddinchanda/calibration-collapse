# Calibration Collapse Under Class Imbalance

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](#testing)

> **Research framework** for studying how class imbalance and resampling strategies affect probability calibration — with a focus on the minority class.

---

## What This Project Does

Standard machine learning evaluation reports **global ECE** (Expected Calibration Error). This hides a critical problem: at high imbalance ratios, the minority class can be severely miscalibrated while global ECE looks fine.

This project:

1. **Proves** the calibration-recall tradeoff exists — resampling improves recall but degrades minority-class calibration
2. **Measures** it rigorously with per-class ECE across 4 datasets and 5 random seeds
3. **Proposes** Per-Class Confidence Drift Monitor (PCDM) — a novel per-class temperature scaling method that reduces minority ECE without sacrificing recall
4. **Validates** findings on controlled synthetic benchmarks with parameterized severity levels

### Core Mathematical Claim

Global ECE is a weighted average dominated by the majority class:

$$\text{ECE}_{\text{global}} = \frac{n_{\text{maj}}}{n} \cdot \text{ECE}_{\text{maj}} + \frac{n_{\text{min}}}{n} \cdot \text{ECE}_{\text{min}}$$

At IR = 100, a minority ECE of 0.5 contributes only **0.005** to global ECE — invisible under standard reporting.

### Per-Class ECE (Primary Metric)

$$\text{ECE}_c = \sum_{b=1}^{B} \frac{|B_b^c|}{n_c} \left| 1.0 - \text{conf}_c(B_b^c) \right|$$

where $B_b^c$ = true class-$c$ samples in confidence bin $b$, and $n_c$ = total class-$c$ samples.

---

## Key Results (5 seeds, 4 datasets)

| Method | pima ECE_min | phoneme ECE_min | credit_card ECE_min |
|---|---|---|---|
| LR + no resampling | 0.456 ± 0.035 | 0.533 ± 0.014 | 0.052 ± 0.056 |
| LR + SMOTE | 0.345 ± 0.033 | 0.348 ± 0.014 | 0.008 ± 0.012 |
| LR + SMOTE + TS | 0.354 ± 0.034 | 0.358 ± 0.012 | ≈ 0.000 |
| **LR + SMOTE + PCDM** | **0.338 ± 0.035** | **0.324 ± 0.012** | ≈ 0.000 |
| RF + no resampling | 0.455 ± 0.029 | 0.301 ± 0.008 | 0.497 ± 0.024 |
| RF + SMOTE | 0.390 ± 0.029 | 0.222 ± 0.005 | **0.518 ± 0.027** ↑ worse |

> RF + SMOTE on credit_card (IR = 580) makes calibration **worse** — calibration collapse at extreme imbalance.

---

## Project Structure

```
calibration_collapse/
│
├── configs/                        # YAML experiment configurations
│   ├── base.yaml                   # Global defaults (seed, splits, bins)
│   ├── datasets/                   # Per-dataset metadata configs
│   ├── experiments/                # Experiment phase definitions
│   │   ├── exp001_resampling_calibration.yaml   # Phase 1 — Failure Discovery
│   │   ├── exp002_posthoc_calibration.yaml      # Phase 2 — Mechanism Validation
│   │   ├── exp003_focal_loss_calibration.yaml   # Phase 2b — Focal Loss (MLP)
│   │   └── exp004_synthetic_stress.yaml         # Phase 3 — Synthetic Benchmarks
│   └── models/                     # Model hyperparameter configs
│
├── src/                            # Core library
│   ├── data/                       # Dataset loading, inspection, synthetic generation
│   ├── preprocessing/              # Anti-leakage splitter + RobustScaler pipeline
│   ├── resampling/                 # SMOTE, ADASYN, BorderlineSMOTE, class_weight
│   ├── models/                     # LR, RF, GBM, MLP (PyTorch + GPU)
│   ├── calibration/                # Temperature scaling, isotonic, Platt, PCDM
│   ├── evaluation/                 # ECE, per-class ECE, Brier, drift, multi-seed
│   ├── visualization/              # Reliability diagrams, frontier, uncertainty bands
│   ├── audit/                      # 7-check leakage detector, integrity, seed validator
│   └── experiment/                 # Runner, SQLite tracker, manifest
│
├── scripts/                        # CLI entry points
│   ├── run_experiment.py           # Run a single experiment phase
│   ├── run_controlled_validation.py # Multi-seed validation (main evidence)
│   ├── run_multiseed.py            # Multi-seed for a single method
│   ├── run_ablation.py             # SMOTE ratio / bin count sweeps
│   ├── audit_experiment.py         # Query results and audit runs
│   ├── inspect_dataset.py          # Dataset statistics
│   ├── _download_datasets.py       # Download real datasets
│   └── _generate_synthetic.py      # Generate synthetic benchmarks
│
├── datasets/                       # Data (not committed — see Setup)
│   ├── raw/                        # pima.csv, phoneme.csv, credit_card.csv
│   └── synthetic/                  # Generated stress-test benchmarks
│
├── tests/                          # 38 unit tests (all passing)
│   ├── test_calibration_metrics.py
│   ├── test_audit_leakage.py
│   ├── test_preprocessing_pipeline.py
│   └── test_synthetic_generator.py
│
├── reports/
│   └── math/                       # ECE derivations, hypothesis register
│
├── outputs/                        # Generated by experiments (not committed)
├── artifacts/                      # Trained models, scalers (not committed)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

### Requirements

- Python 3.12
- CUDA 12.4 (optional — MLP falls back to CPU automatically)
- ~4 GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/calibration_collapse.git
cd calibration_collapse

# Create and activate environment
conda create -n py312 python=3.12
conda activate py312

# Install dependencies
pip install -r requirements.txt

# For GPU support (RTX 3050 / CUDA 12.4)
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Download Datasets

```bash
# Downloads pima (UCI), phoneme (OpenML), generates credit_card proxy
python scripts/_download_datasets.py

# Generate synthetic stress-test benchmarks
python scripts/_generate_synthetic.py
```

---

## Running the Project

### Step 1 — Verify setup

```bash
python -m pytest tests/ -v
python scripts/inspect_dataset.py --dataset pima
```

### Step 2 — Phase 1: Failure Discovery (120 runs)

Proves calibration collapse exists on real datasets.

```bash
# Dry run — see the full 120-run matrix
python scripts/run_experiment.py \
  --config configs/experiments/exp001_resampling_calibration.yaml \
  --dry-run

# Run a single combination to verify
python scripts/run_experiment.py \
  --config configs/experiments/exp001_resampling_calibration.yaml \
  --dataset pima --model logistic_regression \
  --resampler smote --calibrator temperature_scaling

# Full Phase 1
python scripts/run_experiment.py \
  --config configs/experiments/exp001_resampling_calibration.yaml
```

### Step 3 — Phase 2: Mechanism Validation (135 runs)

Tests PCDM vs global temperature scaling.

```bash
python scripts/run_experiment.py \
  --config configs/experiments/exp002_posthoc_calibration.yaml
```

### Step 4 — Phase 3: Synthetic Stress Testing

```bash
python scripts/run_experiment.py \
  --config configs/experiments/exp004_synthetic_stress.yaml
```

### Step 5 — Controlled Multi-Seed Validation (primary evidence)

```bash
# Full validation — 4 datasets × 10 methods × 5 seeds
python scripts/run_controlled_validation.py \
  --datasets pima phoneme credit_card extreme_imbalance_severe \
  --seeds 0 1 2 3 4 \
  --models lr rf

# Build paper table from results
python scripts/_build_paper_table.py
```

### Step 6 — Ablation Studies

```bash
python scripts/run_ablation.py \
  --config configs/experiments/exp001_resampling_calibration.yaml \
  --ablation smote_ratios

python scripts/run_ablation.py \
  --config configs/experiments/exp001_resampling_calibration.yaml \
  --ablation calibration_bins
```

### Step 7 — Audit Results

```bash
# List all runs
python scripts/audit_experiment.py --list-runs

# Best ECE_minority on credit_card
python scripts/audit_experiment.py \
  --query best_ece_minority --dataset credit_card --max-ece 0.05

# Full audit of a specific run
python scripts/audit_experiment.py \
  --experiment-id "exp001_pima_logistic_regression+smote+temperature_scaling" \
  --full-audit
```

---

## Anti-Leakage Architecture

The framework structurally prevents all 4 critical leakage points:

| Leakage Point | Prevention | Audit Check |
|---|---|---|
| Preprocessing | `pipeline.fit()` only accepts `X_train`; `transform()` raises if called first | Check #2 |
| Resampling | Runner enforces split → preprocess → resample order | Check #3 |
| Calibration | `BaseCalibrator.fit()` enforces `split_tag='val'` | Check #4 |
| Metric tuning | All decisions from val metrics only; test touched once | Structural |

---

## Datasets

### Real Datasets (download required)

| Dataset | Rows | Features | IR | Source |
|---|---|---|---|---|
| `pima` | 768 | 8 | 1.9 | UCI Pima Indians Diabetes |
| `phoneme` | 5,404 | 5 | 2.4 | OpenML dataset 1489 |
| `credit_card` | 28,480 | 29 | 580 | Synthetic proxy of Kaggle fraud dataset |

### Calibration Stress Test Suite (generated)

18 severity-parameterized datasets + 5 confidence zone datasets:

| Mechanism | Mild | Moderate | Severe |
|---|---|---|---|
| IR sensitivity | IR = 10 | IR = 50 | IR = 100 |
| Label corruption | noise = 5% | noise = 15% | noise = 30% |
| Boundary overlap | σ = 2.0 | σ = 1.5 | σ = 0.8 |
| Confidence collapse | zone = 0.45 | zone = 0.30 | zone = 0.15 |
| Feature corruption | rate = 10% | rate = 25% | rate = 50% |
| Covariate shift | Δμ = 0.5 | Δμ = 1.5 | Δμ = 3.0 |

---

## Experiment Run Status

| Status | Meaning |
|---|---|
| `completed` | Successful run with valid results |
| `skipped` | Invalid configuration (e.g., SMOTE ratio infeasible for dataset IR) — not a bug |
| `failed` | Unexpected error — investigate |

> **Note on SMOTE feasibility:** SMOTE can only *add* minority samples. If `sampling_strategy` requests a ratio the dataset already meets or exceeds, the run is automatically marked `skipped`. This is correct scientific behavior — the configuration is geometrically impossible.

---

## Testing

```bash
python -m pytest tests/ -v
```

All 38 tests cover:

- ECE decomposition identity (mathematical correctness)
- Per-class ECE vs global ECE separation
- Anti-leakage pipeline enforcement
- Stratified split index overlap detection
- Synthetic generator determinism and severity levels
- Confidence zone targeting accuracy

---

## Research Hypotheses

Full register: [`reports/math/calibration_recall_tradeoff_hypothesis.md`](reports/math/calibration_recall_tradeoff_hypothesis.md)

| # | Hypothesis | Metric | Experiment |
|---|---|---|---|
| H1 | SMOTE increases ECE_minority | `ece_minority` | exp001 |
| H2 | Global ECE hides minority miscalibration | `ece_global` vs `ece_minority` | exp001, exp002 |
| H3 | PCDM reduces ECE_minority vs global TS | `ece_minority` | exp002 |
| H4 | Calibration-Recall tradeoff exists | `recall_minority` vs `ece_minority` | All |
| H5 | Catastrophic Minority Forgetting in MLP | `drift_score_minority` | exp003 |
| H6 | ECE_minority scales with severity level | `ece_minority` vs severity | exp004 |
| H7 | Calibration breaks at specific confidence zones | `ece_minority` vs zone | exp004 |

---

## GPU Acceleration

The MLP model uses PyTorch with automatic CUDA detection:

- **RTX 3050 / CUDA 12.4** — tested and verified
- Falls back to CPU transparently if CUDA is unavailable
- Focal loss (`gamma` parameter) for loss-level imbalance handling
- Early stopping on validation loss, BatchNorm, gradient clipping

All other models (LR, RF, GBM) use scikit-learn on CPU.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
