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
    ece_minority_reliable: bool = True
    ece_majority_reliable: bool = True
    bin_data: Optional[dict] = None  # per-bin acc/conf for reliability diagram
    ece_macro_class: float = float("nan")
    brier_macro_class: float = float("nan")
    ace_macro_class: float = float("nan")
    n_classes: int = 2
    class_labels: Optional[list[int | str]] = None
    ece_per_class: Optional[dict[str, float]] = None
    ace_per_class: Optional[dict[str, float]] = None
    brier_per_class: Optional[dict[str, float]] = None


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
        y_true       : shape (n,) — true labels (binary or multiclass)

        Note on bin sparsity: ECE_minority is unreliable when n_minority < 5*n_bins.
        In practice, datasets with fewer than 30 minority test samples are too
        sparse for a stable minority ECE estimate, so the metric is skipped and
        flagged as unreliable. Use Brier Score for those cases instead.
        The reported n_bins reflects the global ECE bin count.
        """
        p = self._extract_positive_proba(proba)
        y = np.asarray(y_true)

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(f"Expected at least 2 classes, got classes: {classes}")
        if len(classes) > 2:
            return self._compute_multiclass(
                proba=proba,
                y=y,
                classes=classes,
                experiment_id=experiment_id,
                dataset=dataset,
                method=method,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
            )

        y = y.astype(int)

        counts = {int(c): int((y == c).sum()) for c in classes}
        minority_class = int(min(counts, key=counts.get))
        majority_class = int(max(counts, key=counts.get))

        n_minority = counts[minority_class]
        n_majority = counts[majority_class]
        minority_ece_reliable = n_minority >= 30

        # Adaptive bin count for per-class ECE:
        # Require at least 5 samples per non-empty bin for statistical reliability.
        # Use at most n_minority // 5 bins for minority ECE.
        min_samples_per_bin = 5
        n_bins_minority = (
            max(2, min(self._n_bins, n_minority // min_samples_per_bin))
            if minority_ece_reliable
            else 0
        )
        n_bins_majority = max(2, min(self._n_bins, n_majority // min_samples_per_bin))

        if not minority_ece_reliable:
            logger.warning(
                "ECE_minority: skipping because n_minority=%d < 30. "
                "Use Brier Score for minority-class analysis.",
                n_minority,
            )
        elif n_bins_minority < self._n_bins:
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

        if minority_ece_reliable:
            ece_minority, bin_data_min = self._ece_per_class(
                p, y, minority_class, n_bins_minority
            )
        else:
            ece_minority, bin_data_min = float("nan"), []
        ece_majority, bin_data_maj = self._ece_per_class(p, y, majority_class, n_bins_majority)

        if minority_ece_reliable:
            ace_minority, _ = self._ece_per_class_equal_mass(
                p, y, minority_class, n_bins_minority
            )
        else:
            ace_minority = float("nan")
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
            ece_macro_class=float(np.mean([ece_minority, ece_majority])),
            brier_global=float(brier_global),
            brier_minority=float(brier_minority),
            brier_majority=float(brier_majority),
            brier_macro_class=float(np.mean([brier_minority, brier_majority])),
            ace_global=float(ace_global),
            ace_minority=float(ace_minority),
            ace_majority=float(ace_majority),
            ace_macro_class=float(np.mean([ace_minority, ace_majority])),
            n_bins=self._n_bins,
            n_bins_minority=n_bins_minority,
            n_bins_majority=n_bins_majority,
            n_classes=2,
            n_samples_total=len(y),
            n_samples_minority=n_minority,
            n_samples_majority=n_majority,
            minority_class=minority_class,
            majority_class=majority_class,
            ece_minority_reliable=minority_ece_reliable,
            ece_majority_reliable=True,
            class_labels=[int(c) for c in classes],
            ece_per_class={str(minority_class): float(ece_minority), str(majority_class): float(ece_majority)},
            ace_per_class={str(minority_class): float(ace_minority), str(majority_class): float(ace_majority)},
            brier_per_class={
                str(minority_class): float(brier_minority),
                str(majority_class): float(brier_majority),
            },
            bin_data={
                "global": bin_data_global,
                "minority": bin_data_min,
                "majority": bin_data_maj,
                "classes": {str(minority_class): bin_data_min, str(majority_class): bin_data_maj},
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

    def _compute_multiclass(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        experiment_id: str,
        dataset: str,
        method: str,
        output_dir: Optional[Path],
        filename_suffix: Optional[str],
    ) -> CalibrationResult:
        """Compute macro class-conditional calibration metrics for multiclass targets."""
        if proba.ndim != 2:
            raise ValueError("Multiclass calibration requires a 2D probability array.")
        if proba.shape[1] != len(classes):
            raise ValueError(
                f"Probability matrix has {proba.shape[1]} columns but target has {len(classes)} classes."
            )

        class_labels = list(classes)
        class_to_index = {cls: idx for idx, cls in enumerate(class_labels)}
        class_counts = {cls: int((y == cls).sum()) for cls in class_labels}
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        per_class_ece: dict[str, float] = {}
        per_class_ace: dict[str, float] = {}
        per_class_brier: dict[str, float] = {}
        per_class_bins: dict[str, list[dict]] = {}
        per_class_reliable: dict[str, bool] = {}

        for cls in class_labels:
            cls_key = str(cls)
            cls_mask = y == cls
            n_cls = int(cls_mask.sum())
            reliable = n_cls >= 30
            per_class_reliable[cls_key] = reliable

            if not reliable:
                per_class_ece[cls_key] = float("nan")
                per_class_ace[cls_key] = float("nan")
                per_class_brier[cls_key] = float("nan")
                per_class_bins[cls_key] = []
                continue

            cls_idx = class_to_index[cls]
            p_cls = np.clip(proba[cls_mask, cls_idx], _EPS, 1 - _EPS)
            n_bins_cls = max(2, min(self._n_bins, n_cls // 5))
            ece_cls, bin_data_cls = self._ece_class_samples(p_cls, n_bins_cls)
            ace_cls, _ = self._ece_equal_mass(p_cls, np.ones(n_cls, dtype=int), n_bins_cls)
            brier_cls = self._brier_score(p_cls, np.ones(n_cls, dtype=int))

            per_class_ece[cls_key] = float(ece_cls)
            per_class_ace[cls_key] = float(ace_cls)
            per_class_brier[cls_key] = float(brier_cls)
            per_class_bins[cls_key] = bin_data_cls

        reliable_values = [v for v in per_class_ece.values() if not np.isnan(v)]
        reliable_ace_values = [v for v in per_class_ace.values() if not np.isnan(v)]
        reliable_brier_values = [v for v in per_class_brier.values() if not np.isnan(v)]

        y_indices = np.array([class_to_index[label] for label in y])
        y_pred_idx = np.argmax(proba, axis=1)
        y_pred_conf = np.max(proba, axis=1)
        y_correct = (y_pred_idx == y_indices).astype(int)

        ece_global, bin_data_global = self._ece_equal_width(y_pred_conf, y_correct, self._n_bins)
        ace_global, _ = self._ece_equal_mass(y_pred_conf, y_correct, self._n_bins)
        brier_global = self._brier_score_multiclass(proba, y_indices)

        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        n_bins_minority = max(2, min(self._n_bins, n_minority // 5)) if n_minority >= 30 else 0
        n_bins_majority = max(2, min(self._n_bins, n_majority // 5)) if n_majority >= 30 else 0

        result = CalibrationResult(  # type: ignore[call-arg]
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            ece_global=float(ece_global),
            ece_minority=float(per_class_ece.get(str(minority_class), float("nan"))),
            ece_majority=float(per_class_ece.get(str(majority_class), float("nan"))),
            ece_macro_class=float(np.mean(reliable_values)) if reliable_values else float("nan"),
            brier_global=float(brier_global),
            brier_minority=float(per_class_brier.get(str(minority_class), float("nan"))),
            brier_majority=float(per_class_brier.get(str(majority_class), float("nan"))),
            brier_macro_class=float(np.mean(reliable_brier_values)) if reliable_brier_values else float("nan"),
            ace_global=float(ace_global),
            ace_minority=float(per_class_ace.get(str(minority_class), float("nan"))),
            ace_majority=float(per_class_ace.get(str(majority_class), float("nan"))),
            ace_macro_class=float(np.mean(reliable_ace_values)) if reliable_ace_values else float("nan"),
            n_bins=self._n_bins,
            n_bins_minority=n_bins_minority,
            n_bins_majority=n_bins_majority,
            n_classes=len(class_labels),
            n_samples_total=len(y),
            n_samples_minority=n_minority,
            n_samples_majority=n_majority,
            minority_class=minority_class,
            majority_class=majority_class,
            ece_minority_reliable=per_class_reliable.get(str(minority_class), False),
            ece_majority_reliable=per_class_reliable.get(str(majority_class), False),
            class_labels=class_labels,
            ece_per_class=per_class_ece,
            ace_per_class=per_class_ace,
            brier_per_class=per_class_brier,
            bin_data={
                "global": bin_data_global,
                "classes": per_class_bins,
                "minority": per_class_bins.get(str(minority_class), []),
                "majority": per_class_bins.get(str(majority_class), []),
            },
        )

        logger.info(
            "[%s/%s/%s] ECE_global=%.4f, ECE_macro_class=%.4f",
            experiment_id, dataset, method, ece_global, result.ece_macro_class,
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

    def _ece_class_samples(self, p_cls: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
        """ECE for a true-class sample subset, using the subset's confidence values."""
        if len(p_cls) == 0:
            return float("nan"), []

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_data = []
        n_c = len(p_cls)

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
    # Brier Score
    # ------------------------------------------------------------------

    @staticmethod
    def _brier_score(p: np.ndarray, y: np.ndarray) -> float:
        """BS = (1/n) Σ_i (p_i - y_i)^2"""
        if len(p) == 0:
            return float("nan")
        return float(np.mean((p - y.astype(float)) ** 2))

    @staticmethod
    def _brier_score_multiclass(proba: np.ndarray, y_indices: np.ndarray) -> float:
        """Multiclass Brier Score: mean squared error against one-hot labels."""
        if len(proba) == 0:
            return float("nan")
        y_onehot = np.zeros_like(proba)
        y_onehot[np.arange(len(y_indices)), y_indices] = 1.0
        return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

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
