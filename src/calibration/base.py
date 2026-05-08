"""
Abstract base class for all calibrators.

ANTI-LEAKAGE CONTRACT:
  fit() MUST only be called with validation set probabilities.
  split_tag='val' is enforced — passing 'test' raises RuntimeError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

SplitTag = Literal["val", "test", "train"]


class BaseCalibrator(ABC):
    """
    Abstract base for all post-hoc calibration methods.

    fit() is called with (proba_val, y_val) — validation data ONLY.
    calibrate() transforms uncalibrated probabilities to calibrated ones.
    """

    @abstractmethod
    def fit(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        split_tag: SplitTag = "val",
    ) -> None:
        """
        Fit calibrator on validation probabilities.

        Parameters
        ----------
        proba_val  : shape (n_val,) or (n_val, n_classes) — uncalibrated probs
        y_val      : shape (n_val,) — true labels
        split_tag  : MUST be 'val'. Raises RuntimeError if 'test' or 'train'.
        """
        ...

    @abstractmethod
    def calibrate(self, proba_test: np.ndarray) -> np.ndarray:
        """
        Transform uncalibrated probabilities to calibrated ones.

        Parameters
        ----------
        proba_test : shape (n_test,) or (n_test, n_classes)

        Returns
        -------
        np.ndarray of same shape — calibrated probabilities
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return fitted calibration parameters for manifest logging."""
        ...

    def _enforce_val_split(self, split_tag: SplitTag) -> None:
        """Structural guard: calibrators must be fitted on validation data only."""
        if split_tag != "val":
            raise RuntimeError(
                f"Calibrator.fit() called with split_tag='{split_tag}'. "
                "Calibrators MUST be fitted on validation data only (split_tag='val'). "
                "This is a data leakage violation."
            )

    @staticmethod
    def _extract_positive_proba(proba: np.ndarray) -> np.ndarray:
        """Extract P(y=1) from either (n,) or (n, 2) probability arrays."""
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
