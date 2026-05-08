"""ADASYN resampler — adaptive synthetic sampling."""
from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import ADASYN

from .base import BaseResampler

logger = logging.getLogger(__name__)


class ADASYNResampler(BaseResampler):
    """
    Wraps imbalanced-learn ADASYN.

    ADASYN focuses synthetic sample generation on harder-to-learn minority
    samples (those near the decision boundary), which can amplify
    calibration collapse in the minority class.
    """

    def __init__(
        self,
        sampling_strategy: float | str = "auto",
        n_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        self._sampling_strategy = sampling_strategy
        self._n_neighbors = n_neighbors
        self._seed = seed
        self._adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=seed,
        )
        self._metadata: dict = {}

    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_before = len(X_train)
        try:
            X_res, y_res = self._adasyn.fit_resample(X_train, y_train)
        except ValueError as exc:
            # ADASYN can fail when minority class is too small
            logger.warning(
                "ADASYN failed (%s). Falling back to original data.", exc
            )
            X_res, y_res = X_train, y_train

        n_after = len(X_res)
        n_synthetic = n_after - n_before

        self._metadata = {
            "resampler": "adasyn",
            "n_original": n_before,
            "n_resampled": n_after,
            "n_synthetic_created": n_synthetic,
            "sampling_strategy": str(self._sampling_strategy),
            "n_neighbors": self._n_neighbors,
        }

        logger.info("ADASYN: %d → %d samples (%d synthetic)", n_before, n_after, n_synthetic)
        return X_res, y_res

    def get_params(self) -> dict:
        return {
            "sampling_strategy": self._sampling_strategy,
            "n_neighbors": self._n_neighbors,
            "seed": self._seed,
        }

    def get_metadata(self) -> dict:
        return self._metadata
