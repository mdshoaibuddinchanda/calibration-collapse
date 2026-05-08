# Calibration Collapse Under Class Imbalance

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen.svg)](#testing)

> **Research framework** for studying how class imbalance and resampling strategies affect probability calibration in tabular classification systems — with a focus on the minority class.

## Scope

This work focuses specifically on tabular imbalanced classification systems where calibrated probabilities are critical for structured decision-making.

We intentionally restrict the study to tabular domains to isolate imbalance-induced calibration effects without architectural confounders from large-scale vision or language models.

---

## What This Project Does

Standard machine learning evaluation reports **global ECE** (Expected Calibration Error). This hides a critical problem: at high imbalance ratios, the minority class can be severely miscalibrated while global ECE looks fine.

This project:

1. **Studies** how resampling and calibration interact under imbalance in tabular models
2. **Measures** it rigorously with per-class ECE across 4 core datasets and 5 random seeds
3. **Proposes** Per-Class Confidence Drift Monitor (PCDM) — a per-class calibration method that is strong for probabilistic linear models and model-specific in its behavior
4. **Extends** the study to a broader real-dataset suite plus controlled synthetic benchmarks with parameterized severity levels

### Core Mathematical Claim

Global ECE is a weighted average dominated by the majority class:

$$\text{ECE}_{\text{global}} = \frac{n_{\text{maj}}}{n} \cdot \text{ECE}_{\text{maj}} + \frac{n_{\text{min}}}{n} \cdot \text{ECE}_{\text{min}}$$

At IR = 100, a minority ECE of 0.5 contributes only **0.005** to global ECE — invisible under standard reporting.

### Per-Class ECE (Primary Metric)

$$\text{ECE}_c = \sum_{b=1}^{B} \frac{|B_b^c|}{n_c} \left| 1.0 - \text{conf}_c(B_b^c) \right|$$

where $B_b^c$ = true class-$c$ samples in confidence bin $b$, and $n_c$ = total class-$c$ samples.

---

## Key Results

The validated evidence currently centers on the core real datasets, where resampling and PCDM improve minority calibration for logistic regression while RF remains more unstable under post-hoc calibration.

The expanded final suite now includes additional real datasets for follow-up runs and a multiclass benchmark:

- Mammography
- NSL-KDD
- Give Me Some Credit
- Bank Marketing
- Thyroid Disease
- Default of Credit Card Clients
- Dry Bean

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
│   ├── raw/                        # pima.csv, phoneme.csv, default_credit_card_clients.csv, ...
│   └── synthetic/                  # Generated stress-test benchmarks
│
├── tests/                          # 41 unit tests (all passing)
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
# Downloads the full real-dataset suite and validates the local dataset layout
python scripts/_download_datasets.py

# Generate synthetic stress-test benchmarks
python scripts/_generate_synthetic.py
```

---

## Running the Project

Use the following PowerShell sequence on Windows. It is ordered from data download to final paper outputs, and the commands are written as single lines so they can be copied directly into PowerShell.

```powershell
conda activate py312

python scripts/_download_datasets.py
python scripts/_generate_synthetic.py

python -m pytest tests/ -v
python scripts/inspect_dataset.py --dataset pima

python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml --dry-run
python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml --dataset pima --model logistic_regression --resampler smote --calibrator temperature_scaling
python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml

python scripts/run_experiment.py --config configs/experiments/exp002_posthoc_calibration.yaml
python scripts/run_experiment.py --config configs/experiments/exp004_synthetic_stress.yaml

python scripts/run_controlled_validation.py --datasets pima phoneme default_credit_card_clients extreme_imbalance_severe --seeds 0 1 2 3 4 --models lr rf
python scripts/_build_paper_table.py

python scripts/run_ablation.py --config configs/experiments/exp001_resampling_calibration.yaml --ablation smote_ratios
python scripts/run_ablation.py --config configs/experiments/exp001_resampling_calibration.yaml --ablation calibration_bins

python scripts/audit_experiment.py --list-runs
python scripts/audit_experiment.py --query best_ece_minority --dataset default_credit_card_clients --max-ece 0.05
python scripts/audit_experiment.py --experiment-id "exp001_pima_logistic_regression+smote+temperature_scaling" --full-audit

python paper_output/generate_paper.py
```

If you only want the minimum full workflow, run the download, synthetic generation, tests, controlled validation, paper-table build, and paper output generator from the block above.

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
| `default_credit_card_clients` | 30,000 | 23 | ~11.6 | UCI Default of Credit Card Clients |
| `mammography` | 11,183 | 6 | ~42 | UCI Mammography |
| `nsl_kdd` | 148,517 | 41 | variable | NSL-KDD via KaggleHub |
| `give_me_some_credit` | 150,000 | 10 | ~14 | Give Me Some Credit |
| `bank_marketing` | 45,211 | 16 | ~8 | UCI Bank Marketing |
| `thyroid_disease` | 3,772 | 29 | variable | UCI Thyroid Disease |
| `dry_bean` | 13,611 | 16 | multiclass | UCI Dry Bean |

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

All 41 tests cover:

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
| H8 | Multiclass macro class-conditional ECE is stable on Dry Bean | `ece_macro_class` | multiclass subsection |

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
