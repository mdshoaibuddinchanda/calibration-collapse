"""
Classification metrics — always computed per-class, never just macro averages.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    experiment_id: str
    dataset: str
    method: str
    f1_minority: float
    f1_majority: float
    f1_macro: float
    f1_weighted: float
    precision_minority: float
    precision_majority: float
    recall_minority: float
    recall_majority: float
    auc_roc: float
    auc_pr: float          # area under precision-recall curve (better for imbalance)
    n_samples_minority: int
    n_samples_majority: int
    minority_class: int | str
    majority_class: int | str
    confusion_matrix: list[list[int]]
    threshold: float = 0.5


class ClassificationMetrics:
    """
    Computes F1, precision, recall, AUC-ROC, AUC-PR per class.
    Always reports per-class metrics — never just macro averages.
    """

    def compute(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        threshold: float = 0.5,
        output_dir: Optional[Path] = None,
        filename_suffix: Optional[str] = None,
    ) -> ClassificationResult:
        p = proba[:, 1] if proba.ndim == 2 else proba
        y = y_true.astype(int)
        y_pred = (p >= threshold).astype(int)

        classes = np.unique(y)
        counts = {int(c): int((y == c).sum()) for c in classes}
        minority_class = int(min(counts, key=counts.get))
        majority_class = int(max(counts, key=counts.get))

        # Per-class F1
        f1_per_class = f1_score(y, y_pred, average=None, labels=[0, 1], zero_division=0)
        f1_min = float(f1_per_class[minority_class])
        f1_maj = float(f1_per_class[majority_class])

        # Per-class precision / recall
        prec_per = precision_score(y, y_pred, average=None, labels=[0, 1], zero_division=0)
        rec_per = recall_score(y, y_pred, average=None, labels=[0, 1], zero_division=0)

        # AUC metrics
        try:
            auc_roc = float(roc_auc_score(y, p))
        except ValueError:
            auc_roc = float("nan")

        try:
            auc_pr = float(average_precision_score(y, p))
        except ValueError:
            auc_pr = float("nan")

        cm = confusion_matrix(y, y_pred).tolist()

        result = ClassificationResult(
            experiment_id=experiment_id,
            dataset=dataset,
            method=method,
            f1_minority=f1_min,
            f1_majority=f1_maj,
            f1_macro=float(f1_score(y, y_pred, average="macro", zero_division=0)),
            f1_weighted=float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            precision_minority=float(prec_per[minority_class]),
            precision_majority=float(prec_per[majority_class]),
            recall_minority=float(rec_per[minority_class]),
            recall_majority=float(rec_per[majority_class]),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            n_samples_minority=counts[minority_class],
            n_samples_majority=counts[majority_class],
            minority_class=minority_class,
            majority_class=majority_class,
            confusion_matrix=cm,
            threshold=threshold,
        )

        logger.info(
            "[%s/%s/%s] F1_minority=%.4f, AUC_ROC=%.4f, Recall_minority=%.4f",
            experiment_id, dataset, method,
            f1_min, auc_roc, float(rec_per[minority_class]),
        )

        if output_dir is not None:
            self._write_result(result, output_dir, filename_suffix)

        return result

    def _write_result(self, result: ClassificationResult, output_dir: Path, suffix: Optional[str] = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_suffix = suffix.replace("+", "_").replace("/", "_")[:60] if suffix else ""
        fname = (
            f"{result.experiment_id}_{result.dataset}_{safe_suffix}_classification_metrics.json"
            if safe_suffix
            else f"{result.experiment_id}_{result.dataset}_classification_metrics.json"
        )
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2)
        logger.info("Classification metrics written to %s", out_path)
        return out_path
