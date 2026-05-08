"""Logistic Regression classifier wrapper."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class LogisticClassifier(BaseClassifier):
    """
    Logistic Regression with a clean BaseClassifier interface.
    Supports sample_weight for class-weight resampling strategy.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self._C = C
        self._max_iter = max_iter
        self._solver = solver
        self._class_weight = class_weight
        self._seed = seed
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        logger.info(
            "Training LogisticRegression on %d samples, %d features",
            X.shape[0], X.shape[1],
        )
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {
            "model": "logistic_regression",
            "C": self._C,
            "max_iter": self._max_iter,
            "solver": self._solver,
            "class_weight": self._class_weight,
            "seed": self._seed,
        }
