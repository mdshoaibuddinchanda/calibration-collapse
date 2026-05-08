"""
Loads raw tabular data from disk.

CRITICAL ANTI-LEAKAGE RULE:
    Missing value imputation statistics are NOT computed here.
    loader.py returns raw (X, y) — imputation is deferred to
    preprocessing/pipeline.py which is fitted ONLY on X_train.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .registry import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    class_counts: dict[int | str, int]
    imbalance_ratio: float          # majority / minority
    missing_rate: float             # fraction of cells that are NaN
    feature_names: list[str]
    target_name: str
    file_sha256: str
    positive_class: int | str
    is_binary: bool = True
    constant_features: list[str] = field(default_factory=list)


class DatasetLoader:
    """
    Loads a dataset according to its DatasetConfig, validates schema,
    and returns (X, y, metadata).

    No imputation is performed here — raw NaN values are preserved.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._project_root = Path(project_root) if project_root else Path.cwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self, config: DatasetConfig
    ) -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
        """
        Load dataset from disk.

        Returns
        -------
        X : pd.DataFrame  — features (raw, may contain NaN)
        y : pd.Series     — target labels
        metadata : DatasetMetadata
        """
        path = self._resolve_path(config.path)
        logger.info("Loading dataset '%s' from %s", config.name, path)

        df = pd.read_csv(
            path,
            encoding=config.encoding,
            sep=config.separator,
        )

        self._validate_schema(df, config)

        y = df[config.target_column].copy()
        if config.feature_columns is not None:
            X = df[config.feature_columns].copy()
        else:
            X = df.drop(columns=[config.target_column]).copy()

        self._validate_no_target_in_features(X, config)
        self._validate_no_duplicate_indices(X)
        self._validate_no_constant_features(X, config)

        metadata = self._build_metadata(X, y, config, path)
        self._validate_imbalance_range(metadata, config)

        logger.info(
            "Loaded '%s': %d samples, %d features, IR=%.2f",
            config.name, metadata.n_samples, metadata.n_features,
            metadata.imbalance_ratio,
        )
        return X, y, metadata

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        candidate = self._project_root / path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Dataset file not found: tried '{path}' and '{candidate}'"
        )

    def _validate_schema(self, df: pd.DataFrame, config: DatasetConfig) -> None:
        if config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found in dataset "
                f"'{config.name}'. Available columns: {list(df.columns)}"
            )
        if config.feature_columns is not None:
            missing_cols = set(config.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Feature columns {missing_cols} not found in dataset '{config.name}'"
                )

    def _validate_no_target_in_features(
        self, X: pd.DataFrame, config: DatasetConfig
    ) -> None:
        if config.target_column in X.columns:
            raise ValueError(
                f"Target column '{config.target_column}' appears in feature set — "
                "this is target leakage."
            )

    def _validate_no_duplicate_indices(self, X: pd.DataFrame) -> None:
        if X.index.duplicated().any():
            n_dups = X.index.duplicated().sum()
            raise ValueError(
                f"Dataset has {n_dups} duplicate row indices. "
                "Reset the index before loading."
            )

    def _validate_no_constant_features(
        self, X: pd.DataFrame, config: DatasetConfig
    ) -> list[str]:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        constant = [
            col for col in numeric_cols
            if X[col].nunique(dropna=True) <= 1
        ]
        if constant:
            logger.warning(
                "Dataset '%s' has constant features (zero variance): %s. "
                "These will be dropped during preprocessing.",
                config.name, constant,
            )
        return constant

    def _validate_imbalance_range(
        self, meta: DatasetMetadata, config: DatasetConfig
    ) -> None:
        lo, hi = config.expected_imbalance_range
        if not (lo <= meta.imbalance_ratio <= hi):
            logger.warning(
                "Dataset '%s' imbalance ratio %.2f is outside expected range [%.1f, %.1f].",
                config.name, meta.imbalance_ratio, lo, hi,
            )

    # ------------------------------------------------------------------
    # Metadata builder
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: DatasetConfig,
        path: Path,
    ) -> DatasetMetadata:
        class_counts = y.value_counts().to_dict()
        counts_sorted = sorted(class_counts.values())
        minority_count = counts_sorted[0]
        majority_count = counts_sorted[-1]
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float("inf")

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        constant_features = [
            col for col in numeric_cols if X[col].nunique(dropna=True) <= 1
        ]

        missing_rate = X.isnull().values.mean()

        sha256 = self._sha256_file(path)

        return DatasetMetadata(
            name=config.name,
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=y.nunique(),
            class_counts={int(k) if isinstance(k, (int, np.integer)) else k: int(v)
                          for k, v in class_counts.items()},
            imbalance_ratio=float(imbalance_ratio),
            missing_rate=float(missing_rate),
            feature_names=list(X.columns),
            target_name=config.target_column,
            file_sha256=sha256,
            positive_class=config.positive_class,
            is_binary=(y.nunique() == 2),
            constant_features=constant_features,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
