"""
Multi-seed validation runner.

Runs a focused set of (dataset, model, resampler, calibrator) combinations
across multiple seeds to produce:
  - Mean ± std for all metrics
  - 95% confidence intervals
  - Uncertainty-band reliability diagrams
  - Instability comparison plots

This is the primary statistical robustness evidence for the paper.

Usage
-----
    # Full multi-seed run (recommended: 5 seeds)
    python scripts/run_multiseed.py --dataset credit_card --seeds 0 1 2 3 4

    # Quick validation (3 seeds)
    python scripts/run_multiseed.py --dataset pima --seeds 0 1 2 --quick

    # Compare specific methods
    python scripts/run_multiseed.py --dataset credit_card --seeds 0 1 2 3 4 \\
        --methods "none+none" "smote+none" "smote+temperature_scaling" "smote+per_class_adaptive"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.registry import DatasetRegistry
from src.evaluation.multi_seed import MultiSeedValidator, MultiSeedReport, MetricStats
from src.visualization.uncertainty_reliability import UncertaintyReliabilityDiagram


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_method(method_str: str) -> tuple[str, str, str]:
    """Parse 'model+resampler+calibrator' or 'resampler+calibrator' string."""
    parts = method_str.split("+")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return "logistic_regression", parts[0], parts[1]
    else:
        raise ValueError(f"Method must be 'model+resampler+calibrator', got: {method_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed validation for calibration collapse experiments."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (pima, phoneme, credit_card, or synthetic dataset name)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Random seeds to use (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=[
            "logistic_regression+none+none",
            "logistic_regression+smote+none",
            "logistic_regression+smote+temperature_scaling",
            "logistic_regression+smote+per_class_adaptive",
            "logistic_regression+class_weight+temperature_scaling",
        ],
        help="Methods as 'model+resampler+calibrator' strings",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: use only LR, 3 seeds, skip GBM/RF",
    )
    parser.add_argument(
        "--experiment-id", default="multiseed",
        help="Experiment ID prefix",
    )
    parser.add_argument(
        "--config", default="configs/experiments/exp001_resampling_calibration.yaml",
        help="Base experiment config",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if args.quick and len(args.seeds) > 3:
        args.seeds = args.seeds[:3]
        logger.info("Quick mode: using seeds %s", args.seeds)

    # Load config
    base_path = PROJECT_ROOT / "configs" / "base.yaml"
    exp_path = PROJECT_ROOT / args.config
    base_cfg = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    exp_cfg = OmegaConf.load(exp_path) if exp_path.exists() else OmegaConf.create({})
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Dataset registry
    registry = DatasetRegistry(PROJECT_ROOT / "configs" / "datasets")
    errors = registry.validate_all()
    if errors:
        for e in errors:
            logger.debug("Registry: %s", e)

    # Verify dataset exists
    try:
        registry.get(args.dataset)
    except KeyError:
        logger.error("Dataset '%s' not found. Available: %s",
                     args.dataset, registry.list_registered())
        sys.exit(1)

    output_dir = PROJECT_ROOT / "outputs" / "multiseed"
    plot_dir = PROJECT_ROOT / "outputs" / "plots" / "uncertainty"

    logger.info(
        "Multi-seed validation: dataset=%s, seeds=%s, methods=%d",
        args.dataset, args.seeds, len(args.methods),
    )

    # Run each method
    reports: list[MultiSeedReport] = []
    for method_str in args.methods:
        try:
            model, resampler, calibrator = parse_method(method_str)
        except ValueError as e:
            logger.error("Invalid method '%s': %s", method_str, e)
            continue

        logger.info("\n%s", "=" * 60)
        logger.info("Method: %s | %s | %s", model, resampler, calibrator)
        logger.info("=" * 60)

        validator = MultiSeedValidator(
            seeds=args.seeds,
            project_root=PROJECT_ROOT,
            base_config=config_dict,
            output_dir=output_dir,
        )

        try:
            report = validator.run(
                dataset_name=args.dataset,
                model_name=model,
                resampler_name=resampler,
                calibrator_name=calibrator,
                experiment_id=f"{args.experiment_id}_{args.dataset}",
                dataset_registry=registry,
            )
            reports.append(report)
        except Exception as exc:
            logger.error("Method %s failed: %s", method_str, exc, exc_info=True)

    if not reports:
        logger.error("No methods succeeded.")
        sys.exit(1)

    # Generate uncertainty-band reliability diagrams
    diagram = UncertaintyReliabilityDiagram(n_bins=config_dict.get("calibration_bins", 10))

    for report in reports:
        diagram.plot_from_multiseed_report(
            report=report,
            output_dir=plot_dir,
        )

    # Generate instability comparison plot (all methods on one plot)
    if len(reports) > 1:
        diagram.plot_instability_comparison(
            reports=reports,
            dataset=args.dataset,
            experiment_id=f"{args.experiment_id}_{args.dataset}",
            output_dir=plot_dir,
        )

    # Print consolidated summary table
    print("\n" + "=" * 80)
    print(f"MULTI-SEED VALIDATION SUMMARY — {args.dataset.upper()}")
    print(f"Seeds: {args.seeds}")
    print("=" * 80)

    key_metrics = ["ece_global", "ece_minority", "recall_minority", "f1_minority", "auc_roc"]
    header = f"{'Method':<45}" + "".join(f"{m:<22}" for m in key_metrics)
    print(header)
    print("-" * (45 + 22 * len(key_metrics)))

    for report in reports:
        method = f"{report.model}+{report.resampler}+{report.calibrator}"
        row = f"{method:<45}"
        for m in key_metrics:
            if m in report.metric_stats:
                s = report.metric_stats[m]
                row += f"{s.mean:.4f}±{s.std:.4f}      "
            else:
                row += f"{'N/A':<22}"
        print(row)

    print("=" * 80)

    # Save consolidated summary
    summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "n_methods": len(reports),
        "methods": [
            {
                "method": f"{r.model}+{r.resampler}+{r.calibrator}",
                "metrics": {
                    k: {"mean": v.mean, "std": v.std, "ci_lower": v.ci_lower,
                        "ci_upper": v.ci_upper, "cv": v.cv, "is_stable": v.is_stable}
                    for k, v in r.metric_stats.items()
                    if k in key_metrics
                },
                "failed_seeds": r.failed_seeds,
            }
            for r in reports
        ],
    }
    summary_path = output_dir / f"{args.experiment_id}_{args.dataset}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Consolidated summary saved to %s", summary_path)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Plots saved to:   {plot_dir}")


if __name__ == "__main__":
    main()
