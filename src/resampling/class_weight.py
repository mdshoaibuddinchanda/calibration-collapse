"""
Class-weight resampler (passthrough).

Does not modify the data — instead computes and returns sample weights
that downstream models can use to upweight minority samples.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .base import BaseResampler

logger = logging.getLogger(__name__)


class ClassWeightResampler(BaseResampler):
    """
    Passthrough resampler that computes class weights.

    X and y are returned unchanged. The computed sample_weights array
    is stored in metadata for use by models that support sample_weight.
    """

    def __init__(
        self,
        use_class_weight: bool = True,
        class_weight_strategy: str = "balanced",
        seed: int = 42,   # accepted but unused — passthrough has no stochasticity
    ) -> None:
        self._use_class_weight = use_class_weight
        self._class_weight_strategy = class_weight_strategy
        # seed is accepted for interface consistency but not used
        self._sample_weights: np.ndarray | None = None
        self._class_weights: dict | None = None
        self._metadata: dict = {}

    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._use_class_weight:
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight=self._class_weight_strategy,
                classes=classes,
                y=y_train,
            )
            self._class_weights = dict(zip(classes.tolist(), weights.tolist()))
            self._sample_weights = np.array(
                [self._class_weights[c] for c in y_train]
            )
            logger.info("Class weights computed: %s", self._class_weights)
        else:
            self._sample_weights = np.ones(len(y_train))
            self._class_weights = {}

        self._metadata = {
            "resampler": "class_weight" if self._use_class_weight else "none",
            "n_original": len(X_train),
            "n_resampled": len(X_train),
            "n_synthetic_created": 0,
            "use_class_weight": self._use_class_weight,
            "class_weights": self._class_weights,
        }
        return X_train, y_train

    def get_params(self) -> dict:
        return {
            "use_class_weight": self._use_class_weight,
            "class_weight_strategy": self._class_weight_strategy,
        }

    def get_metadata(self) -> dict:
        return self._metadata

    @property
    def sample_weights(self) -> np.ndarray | None:
        return self._sample_weights

    @property
    def class_weights(self) -> dict | None:
        return self._class_weights
