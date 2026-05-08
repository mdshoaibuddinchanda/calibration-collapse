"""Platt scaling calibration (logistic regression on logits)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression

from .base import BaseCalibrator, SplitTag

logger = logging.getLogger(__name__)

_EPS = 1e-7


class PlattScaling(BaseCalibrator):
    """
    Platt scaling: fits a logistic regression on the logits of the
    uncalibrated probabilities.

    p_cal = sigmoid(A * logit(p) + B)
    where A, B are fitted on validation data.
    """

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None
        self._A: Optional[float] = None
        self._B: Optional[float] = None

    def fit(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        split_tag: SplitTag = "val",
    ) -> None:
        self._enforce_val_split(split_tag)

        p = self._extract_positive_proba(proba_val)
        p = np.clip(p, _EPS, 1 - _EPS)
        logits = logit(p).reshape(-1, 1)

        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._lr.fit(logits, y_val.astype(int))

        self._A = float(self._lr.coef_[0, 0])
        self._B = float(self._lr.intercept_[0])

        logger.info("PlattScaling fitted: A=%.4f, B=%.4f", self._A, self._B)

    def calibrate(self, proba_test: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("PlattScaling not fitted. Call fit() first.")

        p = self._extract_positive_proba(proba_test)
        p = np.clip(p, _EPS, 1 - _EPS)
        logits = logit(p).reshape(-1, 1)
        p_cal = self._lr.predict_proba(logits)[:, 1]

        if proba_test.ndim == 2:
            return np.column_stack([1 - p_cal, p_cal])
        return p_cal

    def get_params(self) -> dict:
        return {
            "calibrator": "platt",
            "A": self._A,
            "B": self._B,
        }
