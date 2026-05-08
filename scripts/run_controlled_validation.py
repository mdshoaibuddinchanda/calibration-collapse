"""
Controlled Multi-Seed Validation — the primary statistical evidence for the paper.

Runs the full validation matrix:
  Datasets:  pima, phoneme, credit_card, extreme_imbalance_severe
  Models:    logistic_regression, random_forest
  Methods:   5 (resampler + calibrator combinations)
  Seeds:     5 (0, 1, 2, 3, 4)

Total runs: 4 datasets × 2 models × 5 methods × 5 seeds = 200 runs

Produces:
  - Per-dataset summary tables with mean ± std and 95% CI
  - Uncertainty-band reliability diagrams (minority class)
  - Instability comparison plots (all methods on one plot per dataset)
  - Consolidated cross-dataset summary JSON
  - Paper-ready metric table

Usage
-----
    # Full run (all 4 datasets, 5 seeds)
    python scripts/run_controlled_validation.py

    # Single dataset
    python scripts/run_controlled_validation.py --datasets credit_card

    # Quick test (3 seeds, LR only)
    python scripts/run_controlled_validation.py --quick

    # Resume from a specific dataset
    python scripts/run_controlled_validation.py --datasets phoneme credit_card extreme_imbalance_severe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.registry import DatasetRegistry
from src.evaluation.multi_seed import MultiSeedValidator, MultiSeedReport
from src.visualization.uncertainty_reliability import UncertaintyReliabilityDiagram


# ---------------------------------------------------------------------------
# Validation matrix definition
# ---------------------------------------------------------------------------

ALL_DATASETS = ["pima", "phoneme", "credit_card", "extreme_imbalance_severe"]

# 5 methods covering the paper's core comparisons:
#   baseline (no resampling, no calibration)
#   resampling only (SMOTE, class_weight)
#   resampling + global calibration (temperature scaling)
#   resampling + novel calibration (PCDM)
METHODS_LR = [
    ("logistic_regression", "none",         "none"),
    ("logistic_regression", "smote",        "none"),
    ("logistic_regression", "class_weight", "none"),
    ("logistic_regression", "smote",        "temperature_scaling"),
    ("logistic_regression", "smote",        "per_class_adaptive"),
]

METHODS_RF = [
    ("random_forest", "none",         "none"),
    ("random_forest", "smote",        "none"),
    ("random_forest", "class_weight", "none"),
    ("random_forest", "smote",        "temperature_scaling"),
    ("random_forest", "smote",        "per_class_adaptive"),
]

# RF config override: use 100 trees for speed (200 is overkill for 3 real datasets)
RF_N_ESTIMATORS = 100

SEEDS = [0, 1, 2, 3, 4]

KEY_METRICS = [
    "ece_global", "ece_minority", "ece_majority",
    "brier_minority", "recall_minority", "f1_minority", "auc_roc",
]


def setup_logging(level: str = "WARNING") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_dataset(
    dataset_name: str,
    methods: list[tuple[str, str, str]],
    seeds: list[int],
    config_dict: dict,
    registry: DatasetRegistry,
    output_dir: Path,
    plot_dir: Path,
    experiment_id: str,
    logger: logging.Logger,
) -> list[MultiSeedReport]:
    """Run all methods for one dataset. Returns list of reports."""
    reports = []

    # Run each method
    # For synthetic datasets: LR only (mechanism isolation, not model comparison)
    # For real datasets: both LR and RF
    synthetic_datasets = {"extreme_imbalance_severe", "confidence_collapse_severe",
                          "boundary_overlap_severe", "noisy_minority_severe"}
    is_synthetic = dataset_name in synthetic_datasets

    if is_synthetic:
        active_methods = [m for m in methods if m[0] == "logistic_regression"]
        if not active_methods:
            active_methods = methods  # fallback if only RF was requested
    else:
        active_methods = methods

    for model, resampler, calibrator in active_methods:
        method_str = f"{model}+{resampler}+{calibrator}"
        logger.info("  Running: %s", method_str)
        t0 = time.time()

        validator = MultiSeedValidator(
            seeds=seeds,
            project_root=PROJECT_ROOT,
            base_config={**config_dict, "rf_n_estimators": RF_N_ESTIMATORS},
            output_dir=output_dir,
        )
        try:
            report = validator.run(
                dataset_name=dataset_name,
                model_name=model,
                resampler_name=resampler,
                calibrator_name=calibrator,
                experiment_id=f"{experiment_id}_{dataset_name}",
                dataset_registry=registry,
            )
            reports.append(report)
            elapsed = time.time() - t0

            # Print key metrics inline
            ece_min = report.metric_stats.get("ece_minority")
            recall = report.metric_stats.get("recall_minority")
            auc = report.metric_stats.get("auc_roc")
            failed = len(report.failed_seeds)
            status = f"({len(seeds)-failed}/{len(seeds)} seeds)"
            print(
                f"    {method_str:<50} "
                f"ECE_min={ece_min.mean:.4f}±{ece_min.std:.4f}  "
                f"recall={recall.mean:.4f}±{recall.std:.4f}  "
                f"AUC={auc.mean:.4f}±{auc.std:.4f}  "
                f"{status}  [{elapsed:.0f}s]"
                if ece_min and recall and auc else
                f"    {method_str:<50} FAILED {status}"
            )

        except Exception as exc:
            logger.error("  Method %s failed: %s", method_str, exc)
            print(f"    {method_str:<50} ERROR: {str(exc)[:60]}")

    return reports


def print_dataset_summary(
    dataset_name: str,
    reports: list[MultiSeedReport],
    seeds: list[int],
) -> None:
    """Print formatted summary table for one dataset."""
    print(f"\n{'='*100}")
    print(f"DATASET: {dataset_name.upper()}  |  Seeds: {seeds}  |  Methods: {len(reports)}")
    print(f"{'='*100}")

    header = f"{'Method':<52}" + "".join(f"{m:<24}" for m in KEY_METRICS)
    print(header)
    print("-" * (52 + 24 * len(KEY_METRICS)))

    for report in reports:
        method = f"{report.model}+{report.resampler}+{report.calibrator}"
        row = f"{method:<52}"
        for m in KEY_METRICS:
            if m in report.metric_stats:
                s = report.metric_stats[m]
                cell = f"{s.mean:.4f}±{s.std:.4f}"
                stability = "✓" if s.is_stable else "⚠"
                row += f"{cell+stability:<24}"
            else:
                row += f"{'N/A':<24}"
        print(row)
    print()


def generate_plots(
    dataset_name: str,
    reports: list[MultiSeedReport],
    plot_dir: Path,
    experiment_id: str,
) -> None:
    """Generate all plots for one dataset."""
    diagram = UncertaintyReliabilityDiagram(n_bins=10)

    # Per-method uncertainty reliability diagrams
    for report in reports:
        if report.reliability_data_per_seed:
            diagram.plot_from_multiseed_report(
                report=report,
                output_dir=plot_dir / "per_method",
            )

    # Instability comparison (all methods on one minority plot)
    valid_reports = [r for r in reports if r.reliability_data_per_seed]
    if len(valid_reports) > 1:
        diagram.plot_instability_comparison(
            reports=valid_reports,
            dataset=dataset_name,
            experiment_id=experiment_id,
            output_dir=plot_dir / "comparison",
        )


def build_paper_table(
    all_reports: dict[str, list[MultiSeedReport]],
    output_dir: Path,
    experiment_id: str,
) -> None:
    """
    Build the paper-ready metric table: rows = methods, columns = datasets.
    Primary metrics: ECE_minority and recall_minority.
    """
    table = {}
    for dataset, reports in all_reports.items():
        for report in reports:
            method = f"{report.model}+{report.resampler}+{report.calibrator}"
            if method not in table:
                table[method] = {}
            table[method][dataset] = {
                k: {
                    "mean": v.mean, "std": v.std,
                    "ci_lower": v.ci_lower, "ci_upper": v.ci_upper,
                    "cv": v.cv, "is_stable": v.is_stable,
                }
                for k, v in report.metric_stats.items()
                if k in KEY_METRICS
            }

    # Save as JSON
    table_path = output_dir / f"{experiment_id}_paper_table.json"
    with open(table_path, "w") as fh:
        json.dump(table, fh, indent=2)

    # Print cross-dataset ECE_minority table
    datasets = list(all_reports.keys())
    print(f"\n{'='*100}")
    print("PAPER TABLE: ECE_minority (mean ± std) across datasets")
    print(f"{'='*100}")
    header = f"{'Method':<52}" + "".join(f"{d:<26}" for d in datasets)
    print(header)
    print("-" * (52 + 26 * len(datasets)))

    for method, ds_data in table.items():
        row = f"{method:<52}"
        for ds in datasets:
            if ds in ds_data and "ece_minority" in ds_data[ds]:
            	m = ds_data[ds]["ece_minority"]["mean"]
            	s = ds_data[ds]["ece_minority"]["std"]
            	row += f"{m:.4f}±{s:.4f}          "
            else:
                row += f"{'N/A':<26}"
        print(row)

    print(f"\nPaper table saved to: {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled multi-seed validation across 4 datasets."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS + ["all"],
        help="Datasets to run (default: all 4)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["lr", "rf"],
        choices=["lr", "rf", "both"],
        help="Models: lr=LogisticRegression, rf=RandomForest (default: both)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 seeds, LR only, skip RF",
    )
    parser.add_argument(
        "--experiment-id", default="controlled_validation",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Quick mode overrides
    if args.quick:
        args.seeds = args.seeds[:3]
        args.models = ["lr"]
        print("Quick mode: 3 seeds, LR only")

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    # Build method list
    methods = []
    if "lr" in args.models or "both" in args.models:
        methods.extend(METHODS_LR)
    if "rf" in args.models or "both" in args.models:
        methods.extend(METHODS_RF)

    # Load config
    base_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    exp_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "experiments" / "exp001_resampling_calibration.yaml")
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Registry
    registry = DatasetRegistry(PROJECT_ROOT / "configs" / "datasets")

    output_dir = PROJECT_ROOT / "outputs" / "multiseed"
    plot_dir = PROJECT_ROOT / "outputs" / "plots" / "uncertainty"

    total_runs = len(datasets) * len(methods) * len(args.seeds)
    print(f"\n{'='*80}")
    print(f"CONTROLLED MULTI-SEED VALIDATION")
    print(f"{'='*80}")
    print(f"  Datasets:  {datasets}")
    print(f"  Models:    {args.models}")
    print(f"  Methods:   {len(methods)} per dataset")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*80}\n")

    all_reports: dict[str, list[MultiSeedReport]] = {}
    t_total = time.time()

    for dataset in datasets:
        print(f"\n{'─'*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'─'*80}")

        # Verify dataset is registered
        try:
            registry.get(dataset)
        except KeyError:
            print(f"  ERROR: Dataset '{dataset}' not in registry. Skipping.")
            continue

        t_ds = time.time()
        reports = run_dataset(
            dataset_name=dataset,
            methods=methods,
            seeds=args.seeds,
            config_dict=config_dict,
            registry=registry,
            output_dir=output_dir,
            plot_dir=plot_dir,
            experiment_id=args.experiment_id,
            logger=logger,
        )
        all_reports[dataset] = reports

        print_dataset_summary(dataset, reports, args.seeds)
        generate_plots(dataset, reports, plot_dir, args.experiment_id)

        elapsed_ds = time.time() - t_ds
        print(f"  Dataset completed in {elapsed_ds/60:.1f} min")

    # Cross-dataset paper table
    if len(all_reports) > 1:
        build_paper_table(all_reports, output_dir, args.experiment_id)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETE in {total_elapsed/60:.1f} min")
    print(f"  Outputs: {output_dir}")
    print(f"  Plots:   {plot_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
