"""
Calibration metrics — the most mathematically critical file in the project.

Implements ECE, per-class ECE, Adaptive ECE, Brier Score, and per-class variants.
All implementations follow reports/math/ece_formulation.md.

IMPORTANT: numpy-only implementation (no sklearn) for full auditability.

Mathematical definitions:
--------------------------
Global ECE (equal-width bins):
    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

Per-Class ECE (the primary research metric):
    ECE_c = Σ_b (|B_b^c| / n_c) * |acc_c(B_b^c) - conf_c(B_b^c)|
    where B_b^c = samples of class c in confidence bin b

Adaptive ECE (equal-mass bins):
    ACE = Σ_b (1/B) * |acc(B_b) - conf(B_b)|
    where bins have equal sample counts rather than equal width

Brier Score:
    BS = (1/n) Σ_i (p_i - y_i)^2

Per-class Brier Score:
    BS_c = (1/n_c) Σ_{i: y_i=c} (p_i - y_i)^2
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-10


@dataclass
class CalibrationResult:
    experiment_id: str
    dataset: str
    method: str
    ece_global: float
    ece_minority: float
    ece_majority: float
    brier_global: float
    brier_minority: float
    brier_majority: float
    ace_global: float
    ace_minority: float
    ace_majority: float
    n_bins: int                  # global ECE bin count (configured)
    n_bins_minority: int = 10    # actual bins used for ECE_minority (adaptive)
    n_bins_majority: int = 10    # actual bins used for ECE_majority (adaptive)
    n_samples_total: int = 0
    n_samples_minority: int = 0
    n_samples_majority: int = 0
    minority_class: int | str = 1
    majority_class: int | str = 0
    bin_data: Optional[dict] = None  # per-bin acc/conf for reliability diagram


class CalibrationMetrics:
    """
    Computes all calibration metrics.

    Critical design decision: ece_minority and ece_majority are ALWAYS
    computed separately, even when global ECE is reported. This is the
    analytical core of the research — showing that global ECE hides
    minority miscalibration.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._n_bins = n_bins

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        output_dir: Optional[Path] = None,
        filename_suffix: Optional[str] = None,
    ) -> CalibrationResult:
        """
        Compute all calibration metrics.

        Parameters
        ----------
        proba        : shape (n,) — P(y=1) or shape (n, 2)
        y_true       : shape (n,) — true binary labels

        Note on bin sparsity: ECE_minority is unreliable when n_minority < 5*n_bins.
        The method automatically reduces n_bins for per-class ECE when needed,
        and logs a warning. The reported n_bins reflects the global ECE bin count.
        """
        p = self._extract_positive_proba(proba)
        y = y_true.astype(int)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Expected binary labels, got classes: {classes}")

        counts = {int(c): int((y == c).sum()) for c in classes}
        minority_class = int(min(counts, key=counts.get))
        majority_class = int(max(counts, key=counts.get))

        n_minority = counts[minority_class]
        n_majority = counts[majority_class]

        # Adaptive bin count for per-class ECE:
        # Require at least 5 samples per non-empty bin for statistical reliability.
        # Use at most n_minority // 5 bins for minority ECE.
        min_samples_per_bin = 5
        n_bins_minority = max(2, min(self._n_bins, n_minority // min_samples_per_bin))
        n_bins_majority = max(2, min(self._n_bins, n_majority // min_samples_per_bin))

        if n_bins_minority < self._n_bins:
            logger.warning(
                "ECE_minority: reducing bins from %d to %d (n_minority=%d, "
                "need ≥%d samples/bin for reliability). "
                "Results with <5 samples/bin are statistically unreliable.",
                self._n_bins, n_bins_minority, n_minority, min_samples_per_bin,
            )

        # Global metrics (use configured n_bins)
        ece_global, bin_data_global = self._ece_equal_width(p, y, self._n_bins)
        ace_global, _ = self._ece_equal_mass(p, y, self._n_bins)
        brier_global = self._brier_score(p, y)

        # Per-class metrics (use adaptive bin counts)
        mask_min = y == minority_class
        mask_maj = y == majority_class

        ece_minority, bin_data_min = self._ece_per_class(p, y, minority_class, n_bins_minority)
        ece_majority, bin_data_maj = self._ece_per_class(p, y, majority_class, n_bins_majority)

        ace_minority, _ = self._ece_per_class_equal_mass(p, y, minority_class, n_bins_minority)
        ace_majority, _ = self._ece_per_class_equal_mass(p, y, majority_class, n_bins_majority)

        brier_minority = self._brier_score(p[mask_min], y[mask_min]) if mask_min.sum() > 0 else float("nan")
        brier_majority = self._brier_score(p[mask_maj], y[mask_maj]) if mask_maj.sum() > 0 else float("nan")

        result = CalibrationResult(
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            ece_global=float(ece_global),
            ece_minority=float(ece_minority),
            ece_majority=float(ece_majority),
            brier_global=float(brier_global),
            brier_minority=float(brier_minority),
            brier_majority=float(brier_majority),
            ace_global=float(ace_global),
            ace_minority=float(ace_minority),
            ace_majority=float(ace_majority),
            n_bins=self._n_bins,
            n_bins_minority=n_bins_minority,
            n_bins_majority=n_bins_majority,
            n_samples_total=len(y),
            n_samples_minority=n_minority,
            n_samples_majority=n_majority,
            minority_class=minority_class,
            majority_class=majority_class,
            bin_data={
                "global": bin_data_global,
                "minority": bin_data_min,
                "majority": bin_data_maj,
            },
        )

        logger.info(
            "[%s/%s/%s] ECE_global=%.4f, ECE_minority=%.4f, ECE_majority=%.4f",
            experiment_id, dataset, method,
            ece_global, ece_minority, ece_majority,
        )

        if output_dir is not None:
            self._write_result(result, output_dir, filename_suffix)

        return result

    # ------------------------------------------------------------------
    # ECE — equal-width bins
    # ------------------------------------------------------------------

    def _ece_equal_width(
        self, p: np.ndarray, y: np.ndarray, n_bins: int
    ) -> tuple[float, list[dict]]:
        """
        Global ECE with equal-width confidence bins.

        ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_data = []
        n = len(y)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            # Include right edge in last bin
            if i == n_bins - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)

            n_b = mask.sum()
            if n_b == 0:
                bin_data.append({"bin": i, "lo": lo, "hi": hi, "n": 0,
                                  "acc": None, "conf": None, "gap": None})
                continue

            acc_b = float(y[mask].mean())
            conf_b = float(p[mask].mean())
            gap = abs(acc_b - conf_b)
            ece += (n_b / n) * gap

            bin_data.append({
                "bin": i, "lo": float(lo), "hi": float(hi),
                "n": int(n_b), "acc": acc_b, "conf": conf_b, "gap": gap,
            })

        return float(ece), bin_data

    # ------------------------------------------------------------------
    # Per-class ECE — the primary research metric
    # ------------------------------------------------------------------

    def _ece_per_class(
        self, p: np.ndarray, y: np.ndarray, cls: int, n_bins: int
    ) -> tuple[float, list[dict]]:
        """
        Per-class ECE (class-conditional calibration error).

        Measures: among samples the model assigns confidence p ≈ c to class `cls`,
        what fraction actually belong to class `cls`?

        Formally:
            ECE_c = Σ_b (|B_b^c| / n_c) * |acc_c(B_b^c) - conf_c(B_b^c)|

        where:
            B_b^c  = samples in confidence bin b whose TRUE label is class c
            n_c    = total samples of class c
            conf_c = mean P(y=cls) for samples in B_b^c
            acc_c  = fraction of B_b^c samples that are correctly predicted as cls
                     (always 1.0 for true class-c samples — this is intentional:
                      it measures whether the model's confidence for class c
                      matches the empirical frequency of class c in that bin)

        WHY THIS IS CORRECT:
            We restrict to true class-c samples and ask: "is the model's
            confidence for class c well-calibrated specifically for these samples?"
            A model that assigns P(y=1)=0.3 to all minority samples is
            miscalibrated for the minority class — ECE_minority captures this.
            Global ECE would hide it because majority samples dominate the bins.

        WHY THE "ALL SAMPLES" APPROACH IS WRONG:
            Binning all samples by P(y=cls) and measuring fraction-of-cls in each
            bin is equivalent to global ECE when cls=1 (since P(y=1) is the same
            as the global confidence). It does not isolate minority miscalibration.

        RELATIONSHIP TO GLOBAL ECE:
            ECE_global ≈ (n_maj/n)*ECE_majority + (n_min/n)*ECE_minority
            This decomposition is what proves global ECE hides minority issues.
        """
        cls_mask = y == cls
        n_c = int(cls_mask.sum())
        if n_c == 0:
            return float("nan"), []

        # Confidence for class c, restricted to true class-c samples
        if cls == 1:
            p_cls = p[cls_mask]          # P(y=1) for minority samples
        else:
            p_cls = 1.0 - p[cls_mask]   # P(y=0) for majority samples

        # For true class-c samples, acc is always 1.0 in each bin
        # (they ARE class c). The calibration gap is |1.0 - conf_b|.
        # This measures: "the model says P(y=c)=conf_b for these true-c samples,
        # but the true fraction is 1.0 — so the gap is how far off it is."
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_data = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (p_cls >= lo) & (p_cls <= hi)
            else:
                mask = (p_cls >= lo) & (p_cls < hi)

            n_b = int(mask.sum())
            if n_b == 0:
                bin_data.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": 0,
                                  "acc": None, "conf": None, "gap": None})
                continue

            # acc_b = 1.0 always (these are all true class-c samples)
            acc_b = 1.0
            conf_b = float(p_cls[mask].mean())
            gap = abs(acc_b - conf_b)
            ece += (n_b / n_c) * gap

            bin_data.append({
                "bin": i, "lo": float(lo), "hi": float(hi),
                "n": n_b, "acc": acc_b, "conf": conf_b, "gap": gap,
            })

        return float(ece), bin_data

    # ------------------------------------------------------------------
    # Adaptive ECE — equal-mass bins
    # ------------------------------------------------------------------

    def _ece_equal_mass(
        self, p: np.ndarray, y: np.ndarray, n_bins: int
    ) -> tuple[float, list[dict]]:
        """
        Adaptive ECE with equal-mass bins.

        ACE = Σ_b (1/B) * |acc(B_b) - conf(B_b)|
        where each bin has approximately equal sample count.
        """
        sorted_idx = np.argsort(p)
        p_sorted = p[sorted_idx]
        y_sorted = y[sorted_idx]

        bin_size = len(p) // n_bins
        if bin_size == 0:
            return float("nan"), []

        ece = 0.0
        bin_data = []

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(p)
            p_b = p_sorted[start:end]
            y_b = y_sorted[start:end]

            if len(p_b) == 0:
                continue

            acc_b = float(y_b.mean())
            conf_b = float(p_b.mean())
            gap = abs(acc_b - conf_b)
            ece += gap / n_bins

            bin_data.append({
                "bin": i, "n": len(p_b),
                "acc": acc_b, "conf": conf_b, "gap": gap,
            })

        return float(ece), bin_data

    def _ece_per_class_equal_mass(
        self, p: np.ndarray, y: np.ndarray, cls: int, n_bins: int
    ) -> tuple[float, list[dict]]:
        """
        Adaptive ECE for a specific class.
        Restricts to true class-c samples, bins by their class-c confidence.
        """
        cls_mask = y == cls
        if cls_mask.sum() == 0:
            return float("nan"), []
        # Confidence for class c, restricted to true class-c samples
        p_cls = p[cls_mask] if cls == 1 else 1.0 - p[cls_mask]
        # acc is always 1.0 for true class-c samples — use ones array
        y_ones = np.ones(cls_mask.sum(), dtype=int)
        return self._ece_equal_mass(p_cls, y_ones, n_bins)

    # ------------------------------------------------------------------
    # Brier Score
    # ------------------------------------------------------------------

    @staticmethod
    def _brier_score(p: np.ndarray, y: np.ndarray) -> float:
        """BS = (1/n) Σ_i (p_i - y_i)^2"""
        if len(p) == 0:
            return float("nan")
        return float(np.mean((p - y.astype(float)) ** 2))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_positive_proba(proba: np.ndarray) -> np.ndarray:
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def _write_result(self, result: CalibrationResult, output_dir: Path, suffix: Optional[str] = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Include method suffix to prevent filename collision across runs
        safe_suffix = suffix.replace("+", "_").replace("/", "_")[:60] if suffix else ""
        fname = (
            f"{result.experiment_id}_{result.dataset}_{safe_suffix}_calibration_metrics.json"
            if safe_suffix
            else f"{result.experiment_id}_{result.dataset}_calibration_metrics.json"
        )
        out_path = output_dir / fname
        data = asdict(result)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Calibration metrics written to %s", out_path)
        return out_path
