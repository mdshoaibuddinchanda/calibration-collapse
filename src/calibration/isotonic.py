"""Isotonic regression calibration."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base import BaseCalibrator, SplitTag

logger = logging.getLogger(__name__)

_EPS = 1e-7


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression calibration (non-parametric, monotone).

    Fits a monotone non-decreasing function mapping uncalibrated
    probabilities to calibrated ones.
    """

    def __init__(self) -> None:
        self._iso: Optional[IsotonicRegression] = None

    def fit(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        split_tag: SplitTag = "val",
    ) -> None:
        self._enforce_val_split(split_tag)

        p = self._extract_positive_proba(proba_val)
        p = np.clip(p, _EPS, 1 - _EPS)

        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(p, y_val.astype(float))

        logger.info("IsotonicCalibrator fitted on %d validation samples", len(p))

    def calibrate(self, proba_test: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator not fitted. Call fit() first.")

        p = self._extract_positive_proba(proba_test)
        p = np.clip(p, _EPS, 1 - _EPS)
        p_cal = np.clip(self._iso.predict(p), _EPS, 1 - _EPS)

        if proba_test.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal

    def get_params(self) -> dict:
        return {
            "calibrator": "isotonic",
            "n_thresholds": len(self._iso.X_thresholds_) if self._iso is not None else None,
        }
