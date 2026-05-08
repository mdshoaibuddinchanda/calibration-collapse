"""
Temperature Scaling calibration.

Finds scalar T that minimises NLL on validation probabilities.

IMPORTANT: Temperature scaling is theoretically motivated for models that
produce logit-based probabilities (LR, MLP). For tree-based models (RF, GBM)
whose probabilities are NOT logit-calibrated, we offer two modes:

  use_logit=True  (default for LR/MLP):
      p_cal = sigmoid(logit(p) / T)
      Valid when p = sigmoid(logit) — i.e. the model's raw output is a logit.

  use_logit=False (for RF/GBM):
      p_cal = p^(1/T)  (power scaling, monotone, no logit assumption)
      T > 1 → pushes probabilities toward 1 (corrects underconfidence)
      T < 1 → pushes probabilities toward 0 (corrects overconfidence)
      Note: this is the OPPOSITE direction from logit-mode temperature scaling.
      For an overconfident RF (probabilities too extreme), use T < 1.
      The NLL optimizer finds the correct T automatically.
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

# Models that produce logit-based probabilities
_LOGIT_MODELS = {"logistic_regression", "mlp"}


class TemperatureScaling(BaseCalibrator):
    """
    Temperature scaling with logit-mode and power-mode variants.

    use_logit=True  → sigmoid(logit(p) / T)  — for LR, MLP
        T > 1: overconfident model → reduces confidence (pushes toward 0.5)
        T < 1: underconfident model → sharpens confidence (pushes toward 0/1)

    use_logit=False → p^(1/T)  — for RF, GBM, isotonic outputs
        T > 1: pushes probabilities toward 1 (corrects underconfidence)
        T < 1: pushes probabilities toward 0 (corrects overconfidence)
        The NLL optimizer selects T automatically — direction is data-driven.
    """

    def __init__(self, use_logit: bool = True) -> None:
        self._use_logit = use_logit
        self._T: Optional[float] = None
        self._nll_before: Optional[float] = None
        self._nll_after: Optional[float] = None

    def fit(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        split_tag: SplitTag = "val",
    ) -> None:
        self._enforce_val_split(split_tag)

        p = self._extract_positive_proba(proba_val)
        p = np.clip(p, _EPS, 1 - _EPS)

        self._nll_before = self._nll(p, y_val, T=1.0)

        result = minimize_scalar(
            lambda T: self._nll(p, y_val, T),
            bounds=(0.01, 10.0),
            method="bounded",
        )
        self._T = float(result.x)
        self._nll_after = float(result.fun)

        logger.info(
            "TemperatureScaling fitted (use_logit=%s): T=%.4f, NLL %.4f → %.4f",
            self._use_logit, self._T, self._nll_before, self._nll_after,
        )

    def calibrate(self, proba_test: np.ndarray) -> np.ndarray:
        if self._T is None:
            raise RuntimeError("TemperatureScaling not fitted. Call fit() first.")

        p = self._extract_positive_proba(proba_test)
        p = np.clip(p, _EPS, 1 - _EPS)

        if self._use_logit:
            p_cal = expit(logit(p) / self._T)
        else:
            # Power scaling: p^(1/T) — valid for any probability output
            p_cal = np.power(p, 1.0 / self._T)

        p_cal = np.clip(p_cal, _EPS, 1 - _EPS)

        if proba_test.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal

    def get_params(self) -> dict:
        return {
            "calibrator": "temperature_scaling",
            "use_logit": self._use_logit,
            "T": self._T,
            "nll_before": self._nll_before,
            "nll_after": self._nll_after,
        }

    @property
    def temperature(self) -> Optional[float]:
        return self._T

    # ------------------------------------------------------------------
    # NLL objective — works for both modes
    # ------------------------------------------------------------------

    def _nll(self, p: np.ndarray, y: np.ndarray, T: float) -> float:
        """Negative log-likelihood of calibrated probabilities against y."""
        if T <= 0:
            return 1e9
        p_c = np.clip(p, _EPS, 1 - _EPS)
        if self._use_logit:
            p_cal = expit(logit(p_c) / T)
        else:
            p_cal = np.power(p_c, 1.0 / T)
        p_cal = np.clip(p_cal, _EPS, 1 - _EPS)
        return float(-np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))
