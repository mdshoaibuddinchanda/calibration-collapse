"""Borderline-SMOTE resampler."""
from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE

from .base import BaseResampler

logger = logging.getLogger(__name__)


class BorderlineSMOTEResampler(BaseResampler):
    """
    Wraps imbalanced-learn BorderlineSMOTE.

    Generates synthetic samples only for minority samples near the decision
    boundary — the most calibration-sensitive region.
    """

    def __init__(
        self,
        sampling_strategy: float | str = "auto",
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        kind: str = "borderline-1",
        seed: int = 42,
    ) -> None:
        self._sampling_strategy = sampling_strategy
        self._k_neighbors = k_neighbors
        self._m_neighbors = m_neighbors
        self._kind = kind
        self._seed = seed
        self._resampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            kind=kind,
            random_state=seed,
        )
        self._metadata: dict = {}

    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_before = len(X_train)
        try:
            X_res, y_res = self._resampler.fit_resample(X_train, y_train)
        except ValueError as exc:
            logger.warning("BorderlineSMOTE failed (%s). Using original data.", exc)
            X_res, y_res = X_train, y_train

        n_after = len(X_res)
        self._metadata = {
            "resampler": "borderline_smote",
            "n_original": n_before,
            "n_resampled": n_after,
            "n_synthetic_created": n_after - n_before,
            "kind": self._kind,
        }
        logger.info("BorderlineSMOTE: %d → %d samples", n_before, n_after)
        return X_res, y_res

    def get_params(self) -> dict:
        return {
            "sampling_strategy": self._sampling_strategy,
            "k_neighbors": self._k_neighbors,
            "m_neighbors": self._m_neighbors,
            "kind": self._kind,
            "seed": self._seed,
        }

    def get_metadata(self) -> dict:
        return self._metadata
