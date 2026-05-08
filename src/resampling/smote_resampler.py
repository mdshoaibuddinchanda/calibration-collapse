"""
SMOTE resampler with gradient noise risk scoring and feasibility validation.

Research note (Mechanism 5):
    Synthetic samples near class boundaries have high k-NN distance variance,
    predicting unstable gradient contributions during model training.
    The gradient_noise_risk score quantifies this risk.

Feasibility validation:
    SMOTE can only ADD minority samples. It cannot reduce the minority class.
    If the requested sampling_strategy would require reducing the minority class
    (i.e., current minority ratio already exceeds the requested ratio), the
    resampler raises InvalidSamplingStrategyError and the run is marked SKIPPED
    (not FAILED) — this is an invalid configuration, not a bug.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from imblearn.over_sampling import SMOTE

from .base import BaseResampler

logger = logging.getLogger(__name__)


class InvalidSamplingStrategyError(ValueError):
    """
    Raised when the requested SMOTE sampling_strategy is geometrically
    infeasible for the given dataset.

    SMOTE can only ADD minority samples — it cannot reduce them.
    If the current minority ratio already exceeds the requested ratio,
    the configuration is invalid and the run should be marked SKIPPED.

    This is NOT a bug. It is correct rejection of an impossible request.
    """
    pass


class SMOTEResampler(BaseResampler):
    """
    Wraps imbalanced-learn SMOTE with additional metadata logging.

    Records:
      - n_synthetic_created: number of synthetic minority samples added
      - synthetic_knn_distance_variance: variance of k-NN distances for synthetic samples
      - gradient_noise_risk: 'low' | 'medium' | 'high'
    """

    def __init__(
        self,
        sampling_strategy: float | str = "auto",
        k_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        self._sampling_strategy = sampling_strategy
        self._k_neighbors = k_neighbors
        self._seed = seed
        self._smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed,
        )
        self._metadata: dict = {}

    def fit_resample(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_before = len(X_train)

        # Identify actual minority class (smallest count) — do NOT assume it's class 1
        classes, counts = np.unique(y_train, return_counts=True)
        minority_class = int(classes[np.argmin(counts)])
        majority_class = int(classes[np.argmax(counts)])
        n_minority_before = int((y_train == minority_class).sum())
        n_majority_before = int((y_train == majority_class).sum())

        # ── Feasibility validation ────────────────────────────────────────────
        # SMOTE can only ADD minority samples. If sampling_strategy is a float,
        # it means "target minority/majority ratio". If the current ratio already
        # exceeds the target, SMOTE would need to REMOVE samples — impossible.
        if isinstance(self._sampling_strategy, float):
            current_ratio = n_minority_before / n_majority_before
            if current_ratio >= self._sampling_strategy:
                raise InvalidSamplingStrategyError(
                    f"SMOTE sampling_strategy={self._sampling_strategy} is infeasible: "
                    f"current minority/majority ratio ({current_ratio:.3f}) already "
                    f"meets or exceeds the requested ratio. "
                    f"SMOTE can only ADD minority samples, not remove them. "
                    f"Dataset: n_minority={n_minority_before}, n_majority={n_majority_before}. "
                    f"This configuration is SKIPPED (not a bug)."
                )
        # ─────────────────────────────────────────────────────────────────────

        X_res, y_res = self._smote.fit_resample(X_train, y_train)

        n_after = len(X_res)
        n_synthetic = n_after - n_before
        n_minority_after = int((y_res == minority_class).sum())

        # Compute gradient noise risk score
        knn_var, risk_level = self._compute_gradient_noise_risk(
            X_train, y_train, X_res, y_res, n_before, minority_class
        )

        self._metadata = {
            "resampler": "smote",
            "n_original": n_before,
            "n_resampled": n_after,
            "n_synthetic_created": n_synthetic,
            "minority_class": minority_class,
            "majority_class": majority_class,
            "n_minority_before": n_minority_before,
            "n_minority_after": n_minority_after,
            "sampling_strategy": str(self._sampling_strategy),
            "k_neighbors": self._k_neighbors,
            "synthetic_knn_distance_variance": float(knn_var),
            "gradient_noise_risk": risk_level,
        }

        logger.info(
            "SMOTE: %d → %d samples (%d synthetic created, minority_class=%d, gradient_noise_risk=%s)",
            n_before, n_after, n_synthetic, minority_class, risk_level,
        )
        return X_res, y_res

    def get_params(self) -> dict:
        return {
            "sampling_strategy": self._sampling_strategy,
            "k_neighbors": self._k_neighbors,
            "seed": self._seed,
        }

    def get_metadata(self) -> dict:
        return self._metadata

    # ------------------------------------------------------------------
    # Gradient noise risk
    # ------------------------------------------------------------------

    def _compute_gradient_noise_risk(
        self,
        X_orig: np.ndarray,
        y_orig: np.ndarray,
        X_res: np.ndarray,
        y_res: np.ndarray,
        n_orig: int,
        minority_class: int = 1,
    ) -> tuple[float, str]:
        """
        Compute variance of k-NN distances for synthetic samples.
        Uses the actual minority class, not hardcoded class 1.
        """
        try:
            synthetic_mask = np.zeros(len(X_res), dtype=bool)
            synthetic_mask[n_orig:] = True
            X_synthetic = X_res[synthetic_mask]

            if len(X_synthetic) == 0:
                return 0.0, "none"

            X_minority_orig = X_orig[y_orig == minority_class]
            if len(X_minority_orig) < 2:
                return 0.0, "low"

            diff = X_synthetic[:, np.newaxis, :] - X_minority_orig[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))

            k = min(self._k_neighbors, len(X_minority_orig) - 1)
            knn_dists = np.sort(dists, axis=1)[:, :k]
            knn_var = float(knn_dists.var())

            if knn_var < 0.1:
                risk = "low"
            elif knn_var < 0.5:
                risk = "medium"
            else:
                risk = "high"

            return knn_var, risk

        except Exception as exc:
            logger.warning("Could not compute gradient noise risk: %s", exc)
            return 0.0, "unknown"
