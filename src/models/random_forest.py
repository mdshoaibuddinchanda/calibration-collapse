"""Random Forest classifier wrapper."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class RandomForestClassifier_(BaseClassifier):
    """
    Random Forest with a clean BaseClassifier interface.
    Supports sample_weight for class-weight resampling strategy.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        class_weight: Optional[str] = None,
        n_jobs: int = 1,   # n_jobs=1 for full determinism; set -1 for speed at cost of reproducibility
        seed: int = 42,
        **kwargs,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._class_weight = class_weight
        self._n_jobs = n_jobs
        self._seed = seed
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        logger.info(
            "Training RandomForest (%d trees) on %d samples",
            self._n_estimators, X.shape[0],
        )
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {
            "model": "random_forest",
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "min_samples_leaf": self._min_samples_leaf,
            "class_weight": self._class_weight,
            "n_jobs": self._n_jobs,
            "seed": self._seed,
        }
