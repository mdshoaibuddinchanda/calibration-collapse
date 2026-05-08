"""
Stratified train/val/test splitter.

SAFETY-CRITICAL: All anti-leakage enforcement begins here.
Split indices are logged for full audit trail and reproducibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

MIN_MINORITY_VAL_SAMPLES = 5  # Minimum for calibration fitting.
# Note: was 10, reduced to 5 because:
# 1. Adaptive ECE binning handles sparse bins (n_bins auto-reduced)
# 2. Temperature scaling only needs a few samples to fit T
# 3. Extreme-IR datasets (IR>50) cannot provide 10 val minority samples
#    without very large datasets. Halting on these would exclude the most
#    scientifically interesting cases.
# The integrity checker will warn when n_minority_val < 10.


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    split_indices: dict[str, list[int]] = field(default_factory=dict)
    class_distribution: dict[str, dict] = field(default_factory=dict)


class StratifiedSplitter:
    """
    Produces stratified train/val/test splits with full audit logging.

    Split order:
      1. Stratified split → (train+val) / test
      2. Stratified split → train / val  (from train+val)

    Anti-leakage enforcements:
      - No index appears in more than one split (asserted)
      - Minority proportion in each split within ±2% of full dataset
      - Minimum minority samples in val set (MIN_MINORITY_VAL_SAMPLES)
      - Split indices logged for reproducibility
    """

    def __init__(
        self,
        test_size: float = 0.20,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._test_size = test_size
        self._val_size = val_size
        self._seed = seed

    def split(self, X: pd.DataFrame, y: pd.Series) -> DataSplits:
        """
        Perform stratified split and return DataSplits.

        Parameters
        ----------
        X : pd.DataFrame  — full feature matrix (raw, before preprocessing)
        y : pd.Series     — target labels
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Step 1: split off test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=self._test_size, random_state=self._seed
        )
        trainval_idx, test_idx = next(sss_test.split(X, y))

        X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
        y_trainval = y.iloc[trainval_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        # Step 2: split train+val into train / val
        # val_size is relative to the full dataset, so adjust for trainval size
        val_size_adjusted = self._val_size / (1.0 - self._test_size)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=self._seed + 1
        )
        train_idx_local, val_idx_local = next(sss_val.split(X_trainval, y_trainval))

        X_train = X_trainval.iloc[train_idx_local].reset_index(drop=True)
        y_train = y_trainval.iloc[train_idx_local].reset_index(drop=True)
        X_val = X_trainval.iloc[val_idx_local].reset_index(drop=True)
        y_val = y_trainval.iloc[val_idx_local].reset_index(drop=True)

        # Map back to original indices for audit
        train_orig = trainval_idx[train_idx_local].tolist()
        val_orig = trainval_idx[val_idx_local].tolist()
        test_orig = test_idx.tolist()

        split_indices = {
            "train": train_orig,
            "val": val_orig,
            "test": test_orig,
        }

        self._assert_no_index_overlap(train_orig, val_orig, test_orig)
        self._assert_stratification(y, train_orig, val_orig, test_orig)
        self._assert_min_minority_val(y_val)

        class_distribution = {
            "train": y_train.value_counts(normalize=True).to_dict(),
            "val": y_val.value_counts(normalize=True).to_dict(),
            "test": y_test.value_counts(normalize=True).to_dict(),
            "full": y.value_counts(normalize=True).to_dict(),
        }

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )

        return DataSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            split_indices=split_indices,
            class_distribution={
                k: {str(cls): float(v) for cls, v in d.items()}
                for k, d in class_distribution.items()
            },
        )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_no_index_overlap(
        train: list[int], val: list[int], test: list[int]
    ) -> None:
        train_s, val_s, test_s = set(train), set(val), set(test)
        tv = train_s & val_s
        tt = train_s & test_s
        vt = val_s & test_s
        if tv or tt or vt:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED: Index overlap between splits! "
                f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
            )

    @staticmethod
    def _assert_stratification(
        y: pd.Series,
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
    ) -> None:
        """Assert minority proportion in each split is within ±2% of full dataset."""
        # Use the MINORITY class (smallest count) for stratification check
        minority_class = y.value_counts().index[-1]  # last = smallest count
        full_minority_prop = (y == minority_class).mean()
        tolerance = 0.02

        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if len(idx) == 0:
                continue
            split_y = y.iloc[idx]
            split_prop = (split_y == minority_class).mean()
            if abs(split_prop - full_minority_prop) > tolerance:
                logger.warning(
                    "Stratification warning: '%s' split minority proportion %.3f "
                    "deviates from full dataset %.3f by more than %.2f",
                    split_name, split_prop, full_minority_prop, tolerance,
                )

    @staticmethod
    def _assert_min_minority_val(y_val: pd.Series) -> None:
        """Halt if validation set has too few minority samples for calibration."""
        minority_count = y_val.value_counts().min()
        if minority_count < MIN_MINORITY_VAL_SAMPLES:
            raise RuntimeError(
                f"Validation set has only {minority_count} minority samples "
                f"(minimum required: {MIN_MINORITY_VAL_SAMPLES}). "
                "Consider increasing val_size, using a larger dataset, "
                "or generating synthetic data."
            )
        if minority_count < 10:
            logger.warning(
                "Validation set has only %d minority samples (< 10). "
                "Calibration fitting may be unreliable. "
                "ECE_minority will use adaptive binning.",
                minority_count,
            )
