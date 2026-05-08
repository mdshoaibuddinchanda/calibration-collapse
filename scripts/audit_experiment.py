"""
Run the full audit suite on a completed experiment.

Usage:
    python scripts/audit_experiment.py --experiment-id exp001_run_003 --full-audit
    python scripts/audit_experiment.py --list-runs
    python scripts/audit_experiment.py --query "best_ece_minority" --dataset pima
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.tracker import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed experiment.")
    parser.add_argument("--experiment-id", default=None, help="Experiment run ID to audit")
    parser.add_argument("--full-audit", action="store_true", help="Run full audit suite")
    parser.add_argument("--list-runs", action="store_true", help="List all experiment runs")
    parser.add_argument(
        "--query", default=None,
        choices=["best_ece_minority", "completed_no_leakage", "baseline_comparison"],
        help="Run a canned query",
    )
    parser.add_argument("--dataset", default=None, help="Dataset for query")
    parser.add_argument("--max-ece", type=float, default=0.1, help="ECE threshold for best_ece_minority query")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    tracker = ExperimentTracker(PROJECT_ROOT / "outputs" / "experiments.db")

    if args.list_runs:
        df = tracker.all_runs()
        if df.empty:
            print("No experiment runs found.")
        else:
            print(df[["run_id", "dataset", "model", "resampler", "calibrator",
                       "status", "ece_minority", "recall_minority"]].to_string(index=False))
        return

    if args.query:
        if args.query == "best_ece_minority":
            if not args.dataset:
                print("--dataset required for best_ece_minority query")
                sys.exit(1)
            df = tracker.best_by_ece_minority(args.dataset, args.max_ece)
            print(f"\nBest runs on '{args.dataset}' with ECE_minority < {args.max_ece}:")
            print(df.to_string(index=False) if not df.empty else "No results.")

        elif args.query == "completed_no_leakage":
            if not args.experiment_id:
                print("--experiment-id required")
                sys.exit(1)
            df = tracker.completed_without_leakage(args.experiment_id)
            print(f"\nCompleted runs without leakage for '{args.experiment_id}':")
            print(df.to_string(index=False) if not df.empty else "No results.")

        elif args.query == "baseline_comparison":
            if not args.dataset:
                print("--dataset required")
                sys.exit(1)
            df = tracker.baseline_comparison(args.dataset)
            print(f"\nBaseline comparison for '{args.dataset}':")
            print(df.to_string(index=False) if not df.empty else "No results.")
        return

    if args.full_audit and args.experiment_id:
        # Load manifest and re-run audit checks
        manifest_dir = PROJECT_ROOT / "reports" / "summaries"
        manifests = list(manifest_dir.glob(f"{args.experiment_id}*_manifest.json"))

        if not manifests:
            print(f"No manifest found for experiment ID: {args.experiment_id}")
            sys.exit(1)

        for manifest_path in manifests:
            with open(manifest_path) as fh:
                manifest = json.load(fh)

            print(f"\n{'='*60}")
            print(f"Audit: {manifest_path.name}")
            print(f"{'='*60}")
            print(f"  Status:    {manifest.get('status', 'unknown')}")
            print(f"  Timestamp: {manifest.get('timestamp', 'unknown')}")
            print(f"  Git hash:  {manifest.get('git_hash', 'unknown')}")

            audit = manifest.get("audit_results", {})
            leakage = audit.get("leakage", {})
            integrity = audit.get("integrity", {})

            print(f"\n  Leakage audit: {leakage.get('overall_status', 'N/A')}")
            for check in leakage.get("checks", []):
                status_icon = "✓" if check["status"] == "PASS" else ("✗" if check["status"] == "FATAL" else "⚠")
                print(f"    {status_icon} [{check['status']}] {check['name']}: {check['message'][:80]}")

            print(f"\n  Integrity check: {integrity.get('overall_status', 'N/A')}")
            for check in integrity.get("checks", []):
                status_icon = "✓" if check["status"] == "PASS" else "✗"
                print(f"    {status_icon} [{check['status']}] {check['name']}: {check['message'][:80]}")

            results = manifest.get("results", {})
            if results:
                cal = results.get("calibration", {})
                cls = results.get("classification", {})
                print(f"\n  Results:")
                print(f"    ECE_global:   {cal.get('ece_global', 'N/A'):.4f}")
                print(f"    ECE_minority: {cal.get('ece_minority', 'N/A'):.4f}")
                print(f"    ECE_majority: {cal.get('ece_majority', 'N/A'):.4f}")
                print(f"    F1_minority:  {cls.get('f1_minority', 'N/A'):.4f}")
                print(f"    Recall_min:   {cls.get('recall_minority', 'N/A'):.4f}")
                print(f"    AUC_ROC:      {cls.get('auc_roc', 'N/A'):.4f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
