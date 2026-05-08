"""
Per-Class Confidence Drift Monitor (PCDM) — the novel calibration contribution.

Fits a separate temperature T_k for each class k on validation data,
then applies per-class temperature scaling at inference time.

Mathematical formulation:
    For each class k:
        T_k* = argmin_T NLL(sigmoid(logit(p_val[y_val == k]) / T), y_val[y_val == k])

    At inference:
        For samples predicted as class k (argmax of proba):
            p_cal = sigmoid(logit(p) / T_k)

This is separately ablatable vs. global temperature scaling.
Its effect on minority-class ECE is the primary research finding.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit

from .base import BaseCalibrator, SplitTag

logger = logging.getLogger(__name__)

_EPS = 1e-7
_MIN_SAMPLES_PER_CLASS = 5  # minimum samples to fit per-class temperature


class PerClassAdaptiveCalibrator(BaseCalibrator):
    """
    Per-class temperature scaling (PCDM).

    Each class gets its own temperature T_k, fitted independently on
    validation samples belonging to that class.

    Falls back to global temperature scaling for classes with fewer than
    _MIN_SAMPLES_PER_CLASS validation samples.

    use_logit: True for LR/MLP (logit-based probs), False for RF/GBM (power scaling).
    """

    def __init__(self, use_logit: bool = True) -> None:
        self._use_logit = use_logit
        self._temperatures: dict[int, float] = {}
        self._global_T: Optional[float] = None
        self._classes: Optional[np.ndarray] = None
        self._fit_info: dict = {}

    def fit(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        split_tag: SplitTag = "val",
    ) -> None:
        self._enforce_val_split(split_tag)

        p = self._extract_positive_proba(proba_val)
        p = np.clip(p, _EPS, 1 - _EPS)

        self._classes = np.unique(y_val)

        # Fit global temperature as fallback
        self._global_T = self._fit_temperature(p, y_val.astype(float))

        # Fit per-class temperatures
        for cls in self._classes:
            cls_mask = y_val == cls
            n_cls = cls_mask.sum()

            if n_cls < _MIN_SAMPLES_PER_CLASS:
                logger.warning(
                    "Class %s has only %d validation samples — "
                    "using global T=%.4f as fallback.",
                    cls, n_cls, self._global_T,
                )
                self._temperatures[int(cls)] = self._global_T
                self._fit_info[int(cls)] = {
                    "n_samples": int(n_cls),
                    "T": self._global_T,
                    "fallback": True,
                }
                continue

            # For class 1: fit on P(y=1) vs y=1 labels
            # For class 0: fit on P(y=0) = 1-p vs y=0 labels
            if int(cls) == 1:
                p_cls = p[cls_mask]
                y_cls = y_val[cls_mask].astype(float)
            else:
                p_cls = 1.0 - p[cls_mask]
                y_cls = (1 - y_val[cls_mask]).astype(float)

            T_k = self._fit_temperature(p_cls, y_cls)
            self._temperatures[int(cls)] = T_k
            self._fit_info[int(cls)] = {
                "n_samples": int(n_cls),
                "T": T_k,
                "fallback": False,
            }
            logger.info("Class %s: T_k=%.4f (n=%d)", cls, T_k, n_cls)

        logger.info(
            "PerClassAdaptiveCalibrator fitted (use_logit=%s). Global T=%.4f, per-class: %s",
            self._use_logit, self._global_T,
            {k: f"{v:.4f}" for k, v in self._temperatures.items()},
        )

    def calibrate(self, proba_test: np.ndarray) -> np.ndarray:
        """
        Apply per-class temperature scaling.

        DESIGN: We apply a weighted blend of per-class temperatures, where the
        weight for each class temperature is the model's predicted probability
        for that class. This is mathematically coherent: rather than hard-routing
        by predicted class (which creates discontinuities at the decision boundary),
        we compute a soft-weighted calibration:

            logit_cal = logit(p) / T_blend
            T_blend = T_1 * p + T_0 * (1 - p)

        This ensures:
          - Samples with high P(y=1) are primarily calibrated by T_1
          - Samples with high P(y=0) are primarily calibrated by T_0
          - No discontinuity at the decision boundary
          - Reduces to global TS when T_0 == T_1
        """
        if not self._temperatures:
            raise RuntimeError("PerClassAdaptiveCalibrator not fitted. Call fit() first.")

        p = self._extract_positive_proba(proba_test)
        p = np.clip(p, _EPS, 1 - _EPS)
        logits_p = logit(p)

        T_1 = self._temperatures.get(1, self._global_T)
        T_0 = self._temperatures.get(0, self._global_T)

        # Soft-weighted temperature blend
        T_blend = T_1 * p + T_0 * (1.0 - p)
        T_blend = np.clip(T_blend, 0.01, 10.0)

        p_cal = expit(logits_p / T_blend)
        p_cal = np.clip(p_cal, _EPS, 1 - _EPS)

        if proba_test.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal

    def get_params(self) -> dict:
        return {
            "calibrator": "per_class_adaptive",
            "use_logit": self._use_logit,
            "global_T": self._global_T,
            "per_class_temperatures": self._temperatures,
            "fit_info": self._fit_info,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fit_temperature(self, p: np.ndarray, y: np.ndarray) -> float:
        """Fit a single temperature via NLL minimisation."""
        p = np.clip(p, _EPS, 1 - _EPS)

        def nll(T: float) -> float:
            if T <= 0:
                return 1e9
            if self._use_logit:
                p_cal = expit(logit(p) / T)
            else:
                p_cal = np.power(p, 1.0 / T)
            p_cal = np.clip(p_cal, _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        return float(result.x)
