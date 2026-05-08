"""
Unit tests for calibration_metrics.py.

Tests against known-correct values so reviewers can independently verify.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.classification_metrics import ClassificationMetrics


class TestECEGlobal:
    """Test global ECE with known-correct values."""

    def test_perfect_calibration(self):
        """A perfectly calibrated model should have ECE = 0."""
        # p = y exactly (perfect calibration)
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        y = np.array([0,   0,   0,   0,   1,   1,   1,   1,   1,   1])
        # With 10 bins, each sample is in its own bin
        # acc = y, conf = p → gap = |y - p| per bin
        # ECE = mean(|y - p|) = mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        # Not exactly 0, but let's test a truly perfect case
        p_perfect = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        y_perfect = np.array([0,   0,   0,   0,   0,   1,   1,   1,   1,   1])
        cal = CalibrationMetrics(n_bins=2)
        result = cal.compute(p_perfect, y_perfect)
        assert result.ece_global == pytest.approx(0.0, abs=1e-6)

    def test_maximally_miscalibrated(self):
        """A model predicting 1.0 for positives and 0.0 for negatives (swapped) → high ECE."""
        # 50 negatives predicted as 1.0, 50 positives predicted as 0.0
        p = np.array([1.0] * 50 + [0.0] * 50)
        y = np.array([0] * 50 + [1] * 50)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        # Bin 0 (p≈0): 50 positives → acc=1, conf=0 → gap=1
        # Bin 9 (p≈1): 50 negatives → acc=0, conf=1 → gap=1
        # ECE = 0.5 * 1 + 0.5 * 1 = 1.0
        assert result.ece_global == pytest.approx(1.0, abs=1e-6)

    def test_ece_range(self):
        """ECE must be in [0, 1]."""
        rng = np.random.default_rng(42)
        p = rng.random(500)
        y = (rng.random(500) > 0.7).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        assert 0.0 <= result.ece_global <= 1.0
        assert 0.0 <= result.ece_minority <= 1.0
        assert 0.0 <= result.ece_majority <= 1.0

    def test_bin_counts_sum_to_n(self):
        """Global bin sample counts must sum to n_samples_total."""
        rng = np.random.default_rng(0)
        p = rng.random(200)
        y = (rng.random(200) > 0.8).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        bin_data = result.bin_data["global"]
        total = sum(b["n"] for b in bin_data if b.get("n") is not None)
        assert total == result.n_samples_total

    def test_per_class_bin_counts_sum_to_n_total(self):
        """Per-class bin counts must sum to n_c (class sample count), not n_total."""
        rng = np.random.default_rng(3)
        p = rng.random(300)
        y = (rng.random(300) > 0.8).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)

        # Minority bin counts sum to n_samples_minority
        min_bin_data = result.bin_data["minority"]
        min_total = sum(b["n"] for b in min_bin_data if b.get("n") is not None)
        assert min_total == result.n_samples_minority, (
            f"Minority bin counts sum to {min_total}, expected {result.n_samples_minority}"
        )

        # Majority bin counts sum to n_samples_majority
        maj_bin_data = result.bin_data["majority"]
        maj_total = sum(b["n"] for b in maj_bin_data if b.get("n") is not None)
        assert maj_total == result.n_samples_majority, (
            f"Majority bin counts sum to {maj_total}, expected {result.n_samples_majority}"
        )

    def test_minority_majority_sum_to_total(self):
        """n_minority + n_majority must equal n_samples_total."""
        rng = np.random.default_rng(1)
        p = rng.random(300)
        y = (rng.random(300) > 0.85).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        assert result.n_samples_minority + result.n_samples_majority == result.n_samples_total


class TestPerClassECE:
    """Test per-class ECE — the primary research metric."""

    def test_minority_ece_differs_from_global(self):
        """
        Core research claim: global ECE can hide minority miscalibration.

        Setup: majority class is well-calibrated (p≈0.1 for y=0),
               minority class is badly calibrated (p≈0.5 for y=1).

        With the corrected per-class ECE (bins all samples by class-c confidence):
          - ECE_minority measures: among samples where P(y=1)≈0.5, what fraction are y=1?
            Answer: ~100/1000 = 0.1, so gap = |0.1 - 0.5| = 0.4 → high ECE_minority
          - ECE_global is dominated by majority class → low
        """
        rng = np.random.default_rng(42)
        n_majority = 900
        n_minority = 100

        # Majority: well calibrated (p ≈ 0.1, y = 0)
        p_maj = rng.normal(0.1, 0.02, n_majority).clip(0.01, 0.15)
        y_maj = np.zeros(n_majority, dtype=int)

        # Minority: badly calibrated (p ≈ 0.5 but y = 1 — model is uncertain)
        p_min = rng.normal(0.5, 0.03, n_minority).clip(0.40, 0.60)
        y_min = np.ones(n_minority, dtype=int)

        p = np.concatenate([p_maj, p_min])
        y = np.concatenate([y_maj, y_min])

        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)

        # ECE_minority should be substantially higher than ECE_global
        # (global is dominated by well-calibrated majority)
        assert result.ece_minority > result.ece_global, (
            f"Expected ece_minority ({result.ece_minority:.4f}) > "
            f"ece_global ({result.ece_global:.4f})"
        )
        # ECE_minority should be meaningfully large (model is badly calibrated for minority)
        assert result.ece_minority > 0.1, (
            f"ECE_minority ({result.ece_minority:.4f}) should be > 0.1 for badly calibrated minority"
        )

    def test_per_class_ece_with_2d_proba(self):
        """Per-class ECE should work with (n, 2) probability arrays."""
        rng = np.random.default_rng(5)
        p_pos = rng.random(200)
        p_neg = 1 - p_pos
        proba_2d = np.column_stack([p_neg, p_pos])
        y = (rng.random(200) > 0.7).astype(int)

        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(proba_2d, y)
        assert not np.isnan(result.ece_minority)
        assert not np.isnan(result.ece_majority)

    def test_minority_ece_is_skipped_when_too_sparse(self):
        """Minority ECE should be flagged unreliable below the 30-sample threshold."""
        rng = np.random.default_rng(11)
        p = rng.random(129)
        y = np.array([0] * 100 + [1] * 29)
        cal = CalibrationMetrics(n_bins=10)

        result = cal.compute(p, y)

        assert np.isnan(result.ece_minority)
        assert result.ece_minority_reliable is False
        assert result.n_bins_minority == 0
        assert not np.isnan(result.brier_minority)


class TestBrierScore:
    """Test Brier Score computation."""

    def test_perfect_brier(self):
        """Perfect predictions → Brier Score = 0."""
        p = np.array([0.0, 0.0, 1.0, 1.0])
        y = np.array([0, 0, 1, 1])
        cal = CalibrationMetrics(n_bins=4)
        result = cal.compute(p, y)
        assert result.brier_global == pytest.approx(0.0, abs=1e-6)

    def test_worst_brier(self):
        """Worst predictions → Brier Score = 1."""
        p = np.array([1.0, 1.0, 0.0, 0.0])
        y = np.array([0, 0, 1, 1])
        cal = CalibrationMetrics(n_bins=4)
        result = cal.compute(p, y)
        assert result.brier_global == pytest.approx(1.0, abs=1e-6)

    def test_random_brier_range(self):
        """Brier Score must be in [0, 1]."""
        rng = np.random.default_rng(99)
        p = rng.random(1000)
        y = (rng.random(1000) > 0.7).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        assert 0.0 <= result.brier_global <= 1.0


class TestAdaptiveECE:
    """Test Adaptive ECE (equal-mass bins)."""

    def test_ace_range(self):
        rng = np.random.default_rng(7)
        p = rng.random(500)
        y = (rng.random(500) > 0.8).astype(int)
        cal = CalibrationMetrics(n_bins=10)
        result = cal.compute(p, y)
        assert 0.0 <= result.ace_global <= 1.0


class TestMulticlassCalibration:
    """Test multiclass macro class-conditional calibration metrics."""

    def test_multiclass_macro_class_conditional_ece(self):
        """A perfectly calibrated multiclass model should have near-zero macro ECE."""
        y = np.array([0] * 40 + [1] * 40 + [2] * 40)
        proba = np.zeros((120, 3))
        proba[np.arange(120), y] = 1.0

        cal = CalibrationMetrics(n_bins=5)
        result = cal.compute(proba, y)

        assert result.n_classes == 3
        assert result.class_labels == [0, 1, 2]
        assert result.ece_macro_class == pytest.approx(0.0, abs=1e-6)
        assert result.brier_macro_class == pytest.approx(0.0, abs=1e-6)
        assert all(
            value == pytest.approx(0.0, abs=1e-6)
            for value in result.ece_per_class.values()
        )


class TestMulticlassClassification:
    """Test multiclass classification metrics."""

    def test_multiclass_classification_metrics(self):
        """Perfect multiclass predictions should produce perfect classification scores."""
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        proba = np.zeros((90, 3))
        proba[np.arange(90), y] = 1.0

        cls = ClassificationMetrics()
        result = cls.compute(proba, y)

        assert result.f1_macro == pytest.approx(1.0, abs=1e-6)
        assert result.recall_minority == pytest.approx(1.0, abs=1e-6)
        assert result.recall_majority == pytest.approx(1.0, abs=1e-6)
        assert result.auc_roc == pytest.approx(1.0, abs=1e-6)
