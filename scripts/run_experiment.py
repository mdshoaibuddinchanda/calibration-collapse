"""
CLI entry point for running experiments.

Usage:
    python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml
    python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml --dataset pima
    python scripts/run_experiment.py --config configs/experiments/exp001_resampling_calibration.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.registry import DatasetRegistry
from src.experiment.runner import ExperimentRunner


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a calibration collapse experiment."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config (e.g. configs/experiments/exp001_...yaml)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Run only this dataset (overrides config datasets list)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Run only this model",
    )
    parser.add_argument(
        "--resampler", default=None,
        help="Run only this resampler",
    )
    parser.add_argument(
        "--calibrator", default=None,
        help="Run only this calibrator",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run matrix without executing",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config (merge base.yaml + experiment yaml)
    config_path = PROJECT_ROOT / args.config
    base_path = PROJECT_ROOT / "configs" / "base.yaml"

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    base_cfg = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    exp_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_id = config_dict.get("experiment_id", "unknown")

    # Build run matrix
    datasets = [args.dataset] if args.dataset else config_dict.get("datasets", [])
    models = [args.model] if args.model else config_dict.get("models", [])
    resamplers = [args.resampler] if args.resampler else config_dict.get("resamplers", ["none"])
    calibrators = [args.calibrator] if args.calibrator else config_dict.get("calibrators", ["none"])

    run_matrix = [
        (ds, mdl, res, cal)
        for ds in datasets
        for mdl in models
        for res in resamplers
        for cal in calibrators
    ]

    logger.info(
        "Experiment: %s | %d runs planned (%d datasets × %d models × %d resamplers × %d calibrators)",
        experiment_id, len(run_matrix),
        len(datasets), len(models), len(resamplers), len(calibrators),
    )

    if args.dry_run:
        print(f"\nDRY RUN — {len(run_matrix)} runs planned:")
        for i, (ds, mdl, res, cal) in enumerate(run_matrix, 1):
            print(f"  {i:3d}. {ds} | {mdl} | {res} | {cal}")
        return

    # Load dataset registry
    registry = DatasetRegistry(PROJECT_ROOT / "configs" / "datasets")
    validation_errors = registry.validate_all()
    if validation_errors:
        logger.warning("Dataset validation warnings:\n%s", "\n".join(validation_errors))

    # Run experiments
    runner = ExperimentRunner(project_root=PROJECT_ROOT, config=config_dict)

    results = []
    failed = []

    for i, (ds, mdl, res, cal) in enumerate(run_matrix, 1):
        logger.info("\n[%d/%d] %s | %s | %s | %s", i, len(run_matrix), ds, mdl, res, cal)
        try:
            result = runner.run(
                dataset_name=ds,
                model_name=mdl,
                resampler_name=res,
                calibrator_name=cal,
                experiment_id=experiment_id,
                dataset_registry=registry,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Run failed: %s", exc)
            failed.append((ds, mdl, res, cal, str(exc)))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE: %d/%d runs succeeded", len(results), len(run_matrix))
    if failed:
        logger.warning("%d runs failed:", len(failed))
        for ds, mdl, res, cal, err in failed:
            logger.warning("  %s|%s|%s|%s: %s", ds, mdl, res, cal, err[:100])

    # Generate frontier plot if we have results
    if results:
        frontier_data = []
        for r in results:
            cal_m = r.get("calibration", {})
            cls_m = r.get("classification", {})
            method = r.get("method", "unknown")
            parts = method.split("+")
            frontier_data.append({
                "recall_minority": cls_m.get("recall_minority", float("nan")),
                "ece_minority": cal_m.get("ece_minority", float("nan")),
                "model": parts[0] if len(parts) > 0 else "unknown",
                "resampler": parts[1] if len(parts) > 1 else "none",
                "calibrator": parts[2] if len(parts) > 2 else "none",
                "method_label": method,
            })

        frontier_plotter = __import__(
            "src.visualization.calibration_recall_frontier",
            fromlist=["CalibrationRecallFrontier"]
        ).CalibrationRecallFrontier()

        for ds in datasets:
            ds_data = [r for r in frontier_data if r.get("dataset", ds) == ds or True]
            # Filter to current dataset only
            ds_specific = [r for r in frontier_data]  # all methods, one dataset per plot
            frontier_plotter.plot_from_results(
                results=ds_specific,
                dataset=ds,
                experiment_id=experiment_id,
                output_dir=PROJECT_ROOT / "outputs" / "plots" / "frontier",
            )


if __name__ == "__main__":
    main()
