"""
Unit tests for leakage_detector.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.audit.leakage_detector import LeakageDetector


class TestIndexOverlap:
    def test_no_overlap_passes(self):
        detector = LeakageDetector()
        result = detector.run_all(
            split_indices={"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]},
            n_train_samples=3, n_val_samples=2,
            calibrator_split_tag="val",
            X_train=np.zeros((3, 5)), X_val=np.zeros((2, 5)), X_test=np.zeros((2, 5)),
            y_train=np.array([0, 0, 1]),
            feature_names=["f0", "f1", "f2", "f3", "f4"],
        )
        assert result.overall_status in ("PASS", "WARNING")
        check1 = next(c for c in result.checks if c.check_id == 1)
        assert check1.status == "PASS"

    def test_overlap_raises(self):
        detector = LeakageDetector()
        with pytest.raises(RuntimeError, match="LEAKAGE DETECTED"):
            detector.run_all(
                split_indices={"train": [0, 1, 2, 5], "val": [3, 4], "test": [5, 6]},
                n_train_samples=4, n_val_samples=2,
                calibrator_split_tag="val",
                X_train=np.zeros((4, 5)), X_val=np.zeros((2, 5)), X_test=np.zeros((2, 5)),
                y_train=np.array([0, 0, 1, 0]),
                feature_names=None,
            )


class TestCalibrationSplitTag:
    def test_test_split_tag_raises(self):
        detector = LeakageDetector()
        with pytest.raises(RuntimeError, match="LEAKAGE DETECTED"):
            detector.run_all(
                split_indices={"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]},
                n_train_samples=3, n_val_samples=2,
                calibrator_split_tag="test",  # WRONG — should be 'val'
                X_train=np.zeros((3, 5)), X_val=np.zeros((2, 5)), X_test=np.zeros((2, 5)),
                y_train=np.array([0, 0, 1]),
                feature_names=None,
            )

    def test_val_split_tag_passes(self):
        detector = LeakageDetector()
        result = detector.run_all(
            split_indices={"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]},
            n_train_samples=3, n_val_samples=2,
            calibrator_split_tag="val",
            X_train=np.zeros((3, 5)), X_val=np.zeros((2, 5)), X_test=np.zeros((2, 5)),
            y_train=np.array([0, 0, 1]),
            feature_names=None,
        )
        check4 = next(c for c in result.checks if c.check_id == 4)
        assert check4.status == "PASS"


class TestTargetLeakage:
    def test_high_correlation_warns(self):
        detector = LeakageDetector()
        # Feature perfectly correlated with target
        y = np.array([0, 0, 0, 1, 1, 1])
        X = np.column_stack([y.astype(float), np.random.default_rng(0).random(6)])
        result = detector.run_all(
            split_indices={"train": list(range(6)), "val": [], "test": []},
            n_train_samples=6, n_val_samples=0,
            calibrator_split_tag=None,
            X_train=X, X_val=np.zeros((0, 2)), X_test=np.zeros((0, 2)),
            y_train=y,
            feature_names=["target_copy", "noise"],
        )
        check5 = next(c for c in result.checks if c.check_id == 5)
        assert check5.status == "WARNING"
