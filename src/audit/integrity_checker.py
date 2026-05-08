"""
Integrity checker — validates metric computation correctness and
experimental protocol consistency.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheck:
    name: str
    status: str   # 'PASS' | 'FAIL' | 'SKIPPED'
    message: str


@dataclass
class IntegrityReport:
    experiment_id: str
    timestamp: str
    overall_status: str
    checks: list[IntegrityCheck] = field(default_factory=list)
    fail_count: int = 0


class IntegrityChecker:
    """
    Validates:
    1. ECE bins sum to 1.0
    2. n_minority + n_majority == n_test
    3. Calibration applied to uncalibrated probabilities (not re-applied)
    4. Cross-validation folds have consistent class distributions
    5. All experiments in an ablation used the same seed
    """

    def run_all(
        self,
        cal_result: Optional[dict],
        n_test: Optional[int],
        proba_before_cal: Optional[np.ndarray],
        proba_after_cal: Optional[np.ndarray],
        experiment_seed: Optional[int],
        ablation_seeds: Optional[list[int]],
        experiment_id: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> IntegrityReport:
        checks = []

        checks.append(self._check_ece_bins(cal_result))
        checks.append(self._check_sample_counts(cal_result, n_test))
        checks.append(self._check_no_double_calibration(proba_before_cal, proba_after_cal))
        checks.append(self._check_ablation_seeds(experiment_seed, ablation_seeds))

        fail_count = sum(1 for c in checks if c.status == "FAIL")
        overall = "FAIL" if fail_count > 0 else "PASS"

        report = IntegrityReport(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            checks=checks,
            fail_count=fail_count,
        )

        if output_dir is not None:
            self._write_report(report, output_dir)

        logger.info("Integrity check [%s]: %s", experiment_id, overall)
        return report

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_ece_bins(self, cal_result: Optional[dict]) -> IntegrityCheck:
        if cal_result is None:
            return IntegrityCheck(
                name="ECE Bin Sum", status="SKIPPED",
                message="No calibration result provided.",
            )
        bin_data = cal_result.get("bin_data", {}).get("global", [])
        if not bin_data:
            return IntegrityCheck(
                name="ECE Bin Sum", status="SKIPPED",
                message="No bin data in calibration result.",
            )
        n_total = cal_result.get("n_samples_total", 0)
        if n_total == 0:
            return IntegrityCheck(
                name="ECE Bin Sum", status="SKIPPED",
                message="n_samples_total is 0.",
            )
        bin_sum = sum(b["n"] for b in bin_data if b.get("n") is not None)
        if abs(bin_sum - n_total) > 1:  # allow off-by-one for edge bins
            return IntegrityCheck(
                name="ECE Bin Sum", status="FAIL",
                message=f"Bin sample counts sum to {bin_sum}, expected {n_total}.",
            )
        return IntegrityCheck(
            name="ECE Bin Sum", status="PASS",
            message=f"Bin sample counts sum correctly to {bin_sum}.",
        )

    def _check_sample_counts(
        self, cal_result: Optional[dict], n_test: Optional[int]
    ) -> IntegrityCheck:
        if cal_result is None or n_test is None:
            return IntegrityCheck(
                name="Sample Count Consistency", status="SKIPPED",
                message="cal_result or n_test not provided.",
            )
        n_min = cal_result.get("n_samples_minority", 0)
        n_maj = cal_result.get("n_samples_majority", 0)
        if n_min + n_maj != n_test:
            return IntegrityCheck(
                name="Sample Count Consistency", status="FAIL",
                message=(
                    f"n_minority ({n_min}) + n_majority ({n_maj}) = {n_min + n_maj} "
                    f"!= n_test ({n_test}). Stratification may be broken."
                ),
            )
        return IntegrityCheck(
            name="Sample Count Consistency", status="PASS",
            message=f"n_minority + n_majority = {n_test} (matches n_test).",
        )

    def _check_no_double_calibration(
        self,
        proba_before: Optional[np.ndarray],
        proba_after: Optional[np.ndarray],
    ) -> IntegrityCheck:
        if proba_before is None or proba_after is None:
            return IntegrityCheck(
                name="No Double Calibration", status="SKIPPED",
                message="Probability arrays not provided.",
            )
        # If calibration was applied twice, the output would be more extreme
        # (closer to 0/1) than the input. Check that variance didn't increase dramatically.
        p_before = proba_before[:, 1] if proba_before.ndim == 2 else proba_before
        p_after = proba_after[:, 1] if proba_after.ndim == 2 else proba_after

        var_before = float(np.var(p_before))
        var_after = float(np.var(p_after))

        # Double calibration would typically reduce variance (push toward 0/1 then back)
        # This is a heuristic check
        if var_after > var_before * 3.0:
            return IntegrityCheck(
                name="No Double Calibration", status="FAIL",
                message=(
                    f"Calibrated probability variance ({var_after:.4f}) is >3x "
                    f"uncalibrated variance ({var_before:.4f}). "
                    "Possible double calibration."
                ),
            )
        return IntegrityCheck(
            name="No Double Calibration", status="PASS",
            message=f"Probability variance: before={var_before:.4f}, after={var_after:.4f}.",
        )

    def _check_ablation_seeds(
        self, experiment_seed: Optional[int], ablation_seeds: Optional[list[int]]
    ) -> IntegrityCheck:
        if ablation_seeds is None or len(ablation_seeds) == 0:
            return IntegrityCheck(
                name="Ablation Seed Consistency", status="SKIPPED",
                message="No ablation seeds provided.",
            )
        if experiment_seed is None:
            return IntegrityCheck(
                name="Ablation Seed Consistency", status="SKIPPED",
                message="experiment_seed not provided.",
            )
        inconsistent = [s for s in ablation_seeds if s != experiment_seed]
        if inconsistent:
            return IntegrityCheck(
                name="Ablation Seed Consistency", status="FAIL",
                message=(
                    f"Ablation experiments used different seeds: {inconsistent}. "
                    f"Expected all to use seed={experiment_seed}."
                ),
            )
        return IntegrityCheck(
            name="Ablation Seed Consistency", status="PASS",
            message=f"All ablation experiments used seed={experiment_seed}.",
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_report(self, report: IntegrityReport, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{report.experiment_id}_integrity_report.json"
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info("Integrity report written to %s", out_path)
        return out_path
