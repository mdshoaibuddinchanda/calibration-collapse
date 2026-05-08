"""Gradient Boosting classifier wrapper."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class GradientBoostingClassifier_(BaseClassifier):
    """
    Gradient Boosting with a clean BaseClassifier interface.
    Note: sklearn GBM does not support sample_weight in the same way;
    class_weight is handled via subsample weighting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._subsample = subsample
        self._seed = seed
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        logger.info(
            "Training GradientBoosting (%d estimators) on %d samples",
            self._n_estimators, X.shape[0],
        )
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {
            "model": "gradient_boosting",
            "n_estimators": self._n_estimators,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "subsample": self._subsample,
            "seed": self._seed,
        }
