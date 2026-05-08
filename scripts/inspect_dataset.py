"""
Standalone dataset inspection.

Usage:
    python scripts/inspect_dataset.py --dataset pima
    python scripts/inspect_dataset.py --dataset pima --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DatasetRegistry
from src.data.loader import DatasetLoader
from src.data.inspector import DatasetInspector


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name (from configs/datasets/)")
    parser.add_argument("--verbose", action="store_true", help="Print full feature stats")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s: %(message)s",
    )

    registry = DatasetRegistry(PROJECT_ROOT / "configs" / "datasets")
    loader = DatasetLoader(project_root=PROJECT_ROOT)
    inspector = DatasetInspector(output_dir=PROJECT_ROOT / "outputs" / "logs")

    try:
        cfg = registry.get(args.dataset)
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available datasets: {registry.list_registered()}")
        sys.exit(1)

    errors = registry.validate_all()
    if errors:
        for err in errors:
            print(f"Warning: {err}")

    X, y, metadata = loader.load(cfg)
    report = inspector.inspect(X, y, args.dataset, write_report=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {report.dataset_name}")
    print(f"{'='*60}")
    print(f"  Samples:          {report.n_samples}")
    print(f"  Features:         {report.n_features}")
    print(f"  Imbalance ratio:  {report.imbalance_ratio:.2f}:1")
    print(f"  Minority class:   {report.minority_class} (n={report.class_counts.get(str(report.minority_class), '?')})")
    print(f"  Majority class:   {report.majority_class} (n={report.class_counts.get(str(report.majority_class), '?')})")
    print(f"  Missing rate:     {max(report.missing_rate_per_feature.values(), default=0):.1%} (max per feature)")
    print(f"  Cal. sensitivity: {report.calibration_sensitivity:.3f} (0=easy, 1=hard)")
    print(f"  Feature dominance:{report.feature_dominance_score:.3f} (max |corr(f, y)|)")
    print(f"  Constant features:{report.constant_features or 'none'}")
    print(f"  High-corr features:{report.high_correlation_features or 'none'}")
    print(f"  Min val samples:  {report.recommended_min_val_samples} (recommended)")

    print(f"\n  Calibration sensitivity factors:")
    for k, v in report.calibration_sensitivity_factors.items():
        print(f"    {k}: {v:.3f}")

    if args.verbose:
        print(f"\n  Feature means (top 5):")
        for feat, mean in list(report.feature_means.items())[:5]:
            std = report.feature_stds.get(feat, 0)
            print(f"    {feat}: mean={mean:.3f}, std={std:.3f}")

    print(f"\n  Report written to: outputs/logs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
