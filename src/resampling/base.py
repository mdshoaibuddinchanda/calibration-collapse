"""
Abstract base class for all resamplers.

CONTRACT: fit_resample() MUST only be called on training data.
The interface makes this auditable — the runner enforces the call order.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseResampler(ABC):
    """
    Abstract base for all resampling strategies.

    Subclasses must implement fit_resample() and get_params().
    fit_resample() MUST only ever be called with training data.
    """

    @abstractmethod
    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample the training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
        y_train : np.ndarray, shape (n_train,)

        Returns
        -------
        X_resampled : np.ndarray
        y_resampled : np.ndarray

        MUST only be called on training data.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return all hyperparameters for manifest logging."""
        ...

    def get_metadata(self) -> dict:
        """
        Return resampling metadata after fit_resample() has been called.
        Override in subclasses to provide richer metadata.
        """
        return {"resampler": self.__class__.__name__, "params": self.get_params()}
