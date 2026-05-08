"""
Calibration drift analysis.

Tracks how calibration metrics change across:
  - Training epochs (for MLP)
  - Cross-validation folds (for RF, LR, GBM)

Produces a drift_score = slope of ECE_minority over time.
Positive slope = calibration worsening (Catastrophic Minority Forgetting).
Negative slope = calibration improving.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration_metrics import CalibrationMetrics

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    experiment_id: str
    dataset: str
    method: str
    drift_type: str          # 'epoch' or 'fold'
    n_steps: int
    ece_minority_per_step: list[float]
    ece_global_per_step: list[float]
    drift_score_minority: float   # slope of ECE_minority (positive = worsening)
    drift_score_global: float
    catastrophic_forgetting_detected: bool
    forgetting_threshold: float = 0.01  # slope threshold for detection


class DriftAnalyzer:
    """
    Tracks calibration drift across epochs or folds.

    For MLP: pass epoch-wise probability arrays.
    For RF/LR/GBM: pass fold-wise probability arrays.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._cal_metrics = CalibrationMetrics(n_bins=n_bins)

    def analyze_epoch_drift(
        self,
        proba_per_epoch: list[np.ndarray],
        y_true: np.ndarray,
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> DriftResult:
        """
        Analyze calibration drift across training epochs.

        Parameters
        ----------
        proba_per_epoch : list of (n_test, 2) arrays, one per epoch
        y_true          : true labels
        """
        return self._analyze(
            proba_per_step=proba_per_epoch,
            y_true=y_true,
            drift_type="epoch",
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            output_dir=output_dir,
        )

    def analyze_fold_drift(
        self,
        proba_per_fold: list[np.ndarray],
        y_per_fold: list[np.ndarray],
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> DriftResult:
        """
        Analyze calibration drift across cross-validation folds.

        Parameters
        ----------
        proba_per_fold : list of (n_fold_test, 2) arrays
        y_per_fold     : list of (n_fold_test,) label arrays
        """
        ece_minority_steps = []
        ece_global_steps = []

        for proba_fold, y_fold in zip(proba_per_fold, y_per_fold):
            result = self._cal_metrics.compute(proba_fold, y_fold)
            ece_minority_steps.append(result.ece_minority)
            ece_global_steps.append(result.ece_global)

        return self._build_result(
            ece_minority_steps=ece_minority_steps,
            ece_global_steps=ece_global_steps,
            drift_type="fold",
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze(
        self,
        proba_per_step: list[np.ndarray],
        y_true: np.ndarray,
        drift_type: str,
        experiment_id: str,
        dataset: str,
        method: str,
        output_dir: Optional[Path],
    ) -> DriftResult:
        ece_minority_steps = []
        ece_global_steps = []

        for proba in proba_per_step:
            result = self._cal_metrics.compute(proba, y_true)
            ece_minority_steps.append(result.ece_minority)
            ece_global_steps.append(result.ece_global)

        return self._build_result(
            ece_minority_steps=ece_minority_steps,
            ece_global_steps=ece_global_steps,
            drift_type=drift_type,
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            output_dir=output_dir,
        )

    def _build_result(
        self,
        ece_minority_steps: list[float],
        ece_global_steps: list[float],
        drift_type: str,
        experiment_id: str,
        dataset: str,
        method: str,
        output_dir: Optional[Path],
    ) -> DriftResult:
        n = len(ece_minority_steps)
        if n < 2:
            drift_min = 0.0
            drift_global = 0.0
        else:
            x = np.arange(n, dtype=float)
            drift_min = float(np.polyfit(x, ece_minority_steps, deg=1)[0])
            drift_global = float(np.polyfit(x, ece_global_steps, deg=1)[0])

        forgetting_threshold = 0.01
        catastrophic = drift_min > forgetting_threshold

        if catastrophic:
            logger.warning(
                "[%s/%s/%s] Catastrophic Minority Forgetting detected! "
                "ECE_minority drift_score=%.4f (threshold=%.4f)",
                experiment_id, dataset, method, drift_min, forgetting_threshold,
            )

        result = DriftResult(
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            drift_type=drift_type,
            n_steps=n,
            ece_minority_per_step=ece_minority_steps,
            ece_global_per_step=ece_global_steps,
            drift_score_minority=drift_min,
            drift_score_global=drift_global,
            catastrophic_forgetting_detected=catastrophic,
            forgetting_threshold=forgetting_threshold,
        )

        if output_dir is not None:
            self._write_result(result, output_dir)

        return result

    def _write_result(self, result: DriftResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{result.experiment_id}_{result.dataset}_drift_analysis.json"
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2)
        logger.info("Drift analysis written to %s", out_path)
        return out_path
