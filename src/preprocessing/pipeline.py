"""
Feature preprocessing pipeline.

ANTI-LEAKAGE CONTRACT:
  - fit() is called ONLY with X_train
  - transform() is called with X_val or X_test using train-fitted statistics
  - Calling transform() before fit() raises RuntimeError
  - fit_transform() is structurally forbidden (use fit() then transform())
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

logger = logging.getLogger(__name__)

SplitTag = Literal["train", "val", "test"]


class PreprocessingPipeline:
    """
    Applies imputation → scaling (→ optional encoding) to tabular data.

    RobustScaler is chosen over StandardScaler because it is robust to
    minority-class outliers (uses median and IQR, not mean and std).

    Usage
    -----
    pipeline = PreprocessingPipeline()
    X_train_proc = pipeline.fit(X_train)          # fits AND transforms train
    X_val_proc   = pipeline.transform(X_val,   split_tag='val')
    X_test_proc  = pipeline.transform(X_test,  split_tag='test')
    """

    def __init__(
        self,
        missing_strategy: str = "median",
        encode_categoricals: bool = True,
        artifact_dir: Optional[Path] = None,
        experiment_id: str = "unknown",
        dataset_name: str = "unknown",
    ) -> None:
        self._missing_strategy = missing_strategy
        self._encode_categoricals = encode_categoricals
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._experiment_id = experiment_id
        self._dataset_name = dataset_name

        self._numeric_pipeline: Optional[Pipeline] = None
        self._cat_encoder: Optional[OrdinalEncoder] = None
        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._constant_cols: list[str] = []
        self._fitted = False
        self._n_train_samples: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit the pipeline on X_train and return transformed training data.
        This is the ONLY method that may be called with training data.
        """
        logger.info(
            "Fitting preprocessing pipeline on %d training samples", len(X_train)
        )

        self._numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
        self._cat_cols = list(X_train.select_dtypes(exclude=[np.number]).columns)

        # Identify and drop constant columns
        self._constant_cols = [
            col for col in self._numeric_cols
            if X_train[col].nunique(dropna=True) <= 1
        ]
        if self._constant_cols:
            logger.warning("Dropping constant columns: %s", self._constant_cols)
            self._numeric_cols = [c for c in self._numeric_cols if c not in self._constant_cols]

        # Build numeric pipeline: imputer → scaler
        # Translate 'mode' → 'most_frequent' for sklearn SimpleImputer compatibility
        sklearn_strategy = "most_frequent" if self._missing_strategy == "mode" else self._missing_strategy
        self._numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=sklearn_strategy)),
            ("scaler", RobustScaler()),
        ])
        X_num = X_train[self._numeric_cols].values if self._numeric_cols else np.empty((len(X_train), 0))
        X_num_proc = self._numeric_pipeline.fit_transform(X_num)

        # Categorical encoding
        X_cat_proc = self._fit_transform_categoricals(X_train)

        self._fitted = True
        self._n_train_samples = len(X_train)

        result = self._concat(X_num_proc, X_cat_proc)

        if self._artifact_dir is not None:
            self._save_pipeline()

        logger.info(
            "Pipeline fitted. Output shape: %s (numeric: %d, categorical: %d)",
            result.shape, len(self._numeric_cols), len(self._cat_cols),
        )
        return result

    def transform(self, X: pd.DataFrame, split_tag: SplitTag = "val") -> np.ndarray:
        """
        Transform X using train-fitted statistics.

        Parameters
        ----------
        X        : feature DataFrame (val or test)
        split_tag: 'val' or 'test' — enforced to prevent accidental train transform
        """
        if not self._fitted:
            raise RuntimeError(
                "PreprocessingPipeline.transform() called before fit(). "
                "Call fit(X_train) first."
            )
        if split_tag not in ("val", "test"):
            raise ValueError(
                f"split_tag must be 'val' or 'test', got '{split_tag}'. "
                "fit() handles training data transformation."
            )

        X_num = X[self._numeric_cols].values if self._numeric_cols else np.empty((len(X), 0))
        X_num_proc = self._numeric_pipeline.transform(X_num)  # type: ignore[union-attr]

        X_cat_proc = self._transform_categoricals(X)

        return self._concat(X_num_proc, X_cat_proc)

    def get_feature_names(self) -> list[str]:
        """Return ordered list of output feature names."""
        return self._numeric_cols + self._cat_cols

    # ------------------------------------------------------------------
    # Categorical helpers
    # ------------------------------------------------------------------

    def _fit_transform_categoricals(self, X: pd.DataFrame) -> np.ndarray:
        if not self._cat_cols or not self._encode_categoricals:
            return np.empty((len(X), 0))
        self._cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        return self._cat_encoder.fit_transform(X[self._cat_cols].astype(str).values)

    def _transform_categoricals(self, X: pd.DataFrame) -> np.ndarray:
        if not self._cat_cols or not self._encode_categoricals or self._cat_encoder is None:
            return np.empty((len(X), 0))
        return self._cat_encoder.transform(X[self._cat_cols].astype(str).values)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _concat(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        if X_cat.shape[1] == 0:
            return X_num
        if X_num.shape[1] == 0:
            return X_cat
        return np.hstack([X_num, X_cat])

    def _save_pipeline(self) -> None:
        self._artifact_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        fname = f"{self._experiment_id}_{self._dataset_name}_scaler.pkl"
        path = self._artifact_dir / fname  # type: ignore[operator]
        joblib.dump(self, path)
        logger.info("Preprocessing pipeline saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PreprocessingPipeline":
        return joblib.load(path)

    @property
    def n_train_samples(self) -> Optional[int]:
        return self._n_train_samples

    @property
    def is_fitted(self) -> bool:
        return self._fitted
