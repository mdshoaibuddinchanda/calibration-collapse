"""
Abstract base class for all classifiers.

CONTRACT: Every model must expose predict_proba() — required for calibration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BaseClassifier(ABC):
    """
    Abstract base for all classifiers in the calibration collapse framework.

    All subclasses must implement fit(), predict_proba(), and get_params().
    predict_proba() must return shape (n_samples, n_classes) with column 1 = P(y=1).
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Column 0 = P(y=0), Column 1 = P(y=1) for binary classification.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return all hyperparameters for manifest logging."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def save(self, path: Path) -> None:
        """Save model to disk. Override in subclasses for custom serialisation."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseClassifier":
        import joblib
        return joblib.load(path)
