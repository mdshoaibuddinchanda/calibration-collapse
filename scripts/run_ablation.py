"""
Run ablation studies — sweeps over SMOTE ratios, calibration bins, etc.

Usage:
    python scripts/run_ablation.py --config configs/experiments/exp001_resampling_calibration.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.registry import DatasetRegistry
from src.experiment.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--ablation", default="smote_ratios",
                        choices=["smote_ratios", "calibration_bins"],
                        help="Which ablation to run")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config_path = PROJECT_ROOT / args.config
    base_path = PROJECT_ROOT / "configs" / "base.yaml"
    base_cfg = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    exp_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_id = config_dict.get("experiment_id", "unknown")
    ablations = config_dict.get("ablations", {})

    registry = DatasetRegistry(PROJECT_ROOT / "configs" / "datasets")
    datasets = [args.dataset] if args.dataset else config_dict.get("datasets", [])

    ablation_results = []

    if args.ablation == "smote_ratios":
        ratios = ablations.get("smote_ratios", [0.5, 1.0, 2.0])
        logger.info("Running SMOTE ratio ablation: %s", ratios)

        for ratio in ratios:
            for ds in datasets:
                for model in config_dict.get("models", ["logistic_regression"]):
                    ablation_config = {**config_dict}
                    # Pass ratio via config — runner will pick it up
                    ablation_config["smote_sampling_strategy"] = ratio
                    ablation_config["experiment_id"] = f"{experiment_id}_smote{ratio}"

                    runner = ExperimentRunner(
                        project_root=PROJECT_ROOT, config=ablation_config
                    )
                    try:
                        result = runner.run(
                            dataset_name=ds,
                            model_name=model,
                            resampler_name="smote",
                            calibrator_name="temperature_scaling",
                            experiment_id=f"{experiment_id}_smote{ratio}",
                            dataset_registry=registry,
                        )
                        cal = result.get("calibration", {})
                        cls = result.get("classification", {})
                        ablation_results.append({
                            "ablation": "smote_ratio",
                            "param_value": ratio,
                            "dataset": ds,
                            "model": model,
                            "ece_global": cal.get("ece_global"),
                            "ece_minority": cal.get("ece_minority"),
                            "recall_minority": cls.get("recall_minority"),
                            "f1_minority": cls.get("f1_minority"),
                            "status": "completed",
                        })
                    except Exception as exc:
                        status = "skipped" if "infeasible" in str(exc).lower() or "InvalidSampling" in type(exc).__name__ else "failed"
                        logger.warning("Ablation run %s (ratio=%s, %s/%s): %s", status, ratio, ds, model, str(exc)[:80])
                        ablation_results.append({
                            "ablation": "smote_ratio",
                            "param_value": ratio,
                            "dataset": ds,
                            "model": model,
                            "ece_global": None,
                            "ece_minority": None,
                            "recall_minority": None,
                            "f1_minority": None,
                            "status": status,
                        })

    elif args.ablation == "calibration_bins":
        bin_counts = ablations.get("calibration_bins", [10, 15, 20])
        logger.info("Running calibration bins ablation: %s", bin_counts)

        for n_bins in bin_counts:
            for ds in datasets:
                for model in config_dict.get("models", ["logistic_regression"]):
                    ablation_config = {**config_dict, "calibration_bins": n_bins}
                    runner = ExperimentRunner(
                        project_root=PROJECT_ROOT, config=ablation_config
                    )
                    try:
                        result = runner.run(
                            dataset_name=ds,
                            model_name=model,
                            resampler_name="smote",
                            calibrator_name="temperature_scaling",
                            experiment_id=f"{experiment_id}_bins{n_bins}",
                            dataset_registry=registry,
                        )
                        cal = result.get("calibration", {})
                        ablation_results.append({
                            "ablation": "calibration_bins",
                            "param_value": n_bins,
                            "dataset": ds,
                            "model": model,
                            "ece_global": cal.get("ece_global"),
                            "ece_minority": cal.get("ece_minority"),
                        })
                    except Exception as exc:
                        logger.error("Ablation run failed: %s", exc)

    # Write ablation summary CSV
    if ablation_results:
        out_dir = PROJECT_ROOT / "reports" / "ablations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{experiment_id}_{args.ablation}_ablation_summary.csv"
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ablation_results[0].keys())
            writer.writeheader()
            writer.writerows(ablation_results)
        logger.info("Ablation summary written to %s", out_path)
        print(f"\nAblation complete. Results: {out_path}")


if __name__ == "__main__":
    main()
