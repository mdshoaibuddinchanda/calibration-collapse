"""
Analyzes a loaded dataset and produces a DatasetReport.

Computes: imbalance ratio, per-class sample counts, feature statistics,
missing value rates, feature correlation data, and a calibration sensitivity
score (proxy for how difficult calibration will be).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    dataset_name: str
    timestamp: str
    n_samples: int
    n_features: int
    imbalance_ratio: float
    minority_class: int | str
    majority_class: int | str
    class_counts: dict
    missing_rate_per_feature: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_dominance_score: float      # max |corr(feature, target)|
    calibration_sensitivity: float      # [0, 1] — higher = harder to calibrate
    calibration_sensitivity_factors: dict[str, float]
    constant_features: list[str]
    high_correlation_features: list[str]  # |corr| > 0.9 with target
    recommended_min_val_samples: int


class DatasetInspector:
    """
    Produces a DatasetReport from (X, y) and optionally writes it to disk.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else None

    def inspect(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str,
        write_report: bool = True,
    ) -> DatasetReport:
        """
        Analyse the dataset and return a DatasetReport.
        If write_report=True and output_dir was set, saves JSON to disk.
        """
        logger.info("Inspecting dataset '%s'", dataset_name)

        class_counts = y.value_counts().to_dict()
        sorted_classes = sorted(class_counts, key=lambda k: class_counts[k])
        minority_class = sorted_classes[0]
        majority_class = sorted_classes[-1]
        minority_n = class_counts[minority_class]
        majority_n = class_counts[majority_class]
        imbalance_ratio = majority_n / minority_n if minority_n > 0 else float("inf")

        # Feature statistics
        numeric_X = X.select_dtypes(include=[np.number])
        feature_means = numeric_X.mean().to_dict()
        feature_stds = numeric_X.std().to_dict()
        missing_rate = {col: float(X[col].isnull().mean()) for col in X.columns}

        # Constant features
        constant_features = [
            col for col in numeric_X.columns
            if numeric_X[col].nunique(dropna=True) <= 1
        ]

        # Feature-target correlations (point-biserial for binary target)
        y_numeric = pd.to_numeric(y, errors="coerce")
        correlations: dict[str, float] = {}
        for col in numeric_X.columns:
            if col in constant_features:
                correlations[col] = 0.0
                continue
            valid_mask = numeric_X[col].notna() & y_numeric.notna()
            if valid_mask.sum() < 10:
                correlations[col] = 0.0
                continue
            corr = float(np.corrcoef(
                numeric_X.loc[valid_mask, col].values,
                y_numeric[valid_mask].values,
            )[0, 1])
            correlations[col] = corr if not np.isnan(corr) else 0.0

        feature_dominance_score = max(abs(v) for v in correlations.values()) if correlations else 0.0
        high_corr_features = [k for k, v in correlations.items() if abs(v) > 0.9]

        # Calibration sensitivity score
        cal_sensitivity, cal_factors = self._calibration_sensitivity(
            numeric_X, y_numeric, imbalance_ratio, minority_n
        )

        # Recommended minimum validation samples for minority class
        recommended_min_val = max(10, int(minority_n * 0.15))

        report = DatasetReport(
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            n_samples=len(X),
            n_features=X.shape[1],
            imbalance_ratio=float(imbalance_ratio),
            minority_class=minority_class,
            majority_class=majority_class,
            class_counts={str(k): int(v) for k, v in class_counts.items()},
            missing_rate_per_feature={k: float(v) for k, v in missing_rate.items()},
            feature_means={k: float(v) for k, v in feature_means.items()},
            feature_stds={k: float(v) for k, v in feature_stds.items()},
            feature_dominance_score=float(feature_dominance_score),
            calibration_sensitivity=float(cal_sensitivity),
            calibration_sensitivity_factors=cal_factors,
            constant_features=constant_features,
            high_correlation_features=high_corr_features,
            recommended_min_val_samples=recommended_min_val,
        )

        if write_report and self._output_dir is not None:
            self._write_report(report)

        logger.info(
            "Dataset '%s': IR=%.2f, cal_sensitivity=%.3f, n_minority=%d",
            dataset_name, imbalance_ratio, cal_sensitivity, minority_n,
        )
        return report

    # ------------------------------------------------------------------
    # Calibration sensitivity
    # ------------------------------------------------------------------

    def _calibration_sensitivity(
        self,
        X_num: pd.DataFrame,
        y: pd.Series,
        imbalance_ratio: float,
        minority_n: int,
    ) -> tuple[float, dict[str, float]]:
        """
        Proxy score [0, 1] for how difficult calibration will be.

        Factors:
          - imbalance_ratio_score: higher IR → harder
          - class_overlap_score: lower inter-class distance → harder
          - minority_size_score: fewer minority samples → harder
          - feature_variance_ratio: high variance ratio → harder
        """
        # Imbalance factor (log-scaled, capped at 1)
        ir_score = min(1.0, np.log1p(imbalance_ratio) / np.log1p(500))

        # Minority size factor
        size_score = max(0.0, 1.0 - minority_n / 500.0)

        # Class overlap: Bhattacharyya-like distance between class means
        overlap_score = 0.5  # default if not computable
        if len(X_num.columns) > 0 and y.notna().sum() > 10:
            try:
                classes = y.dropna().unique()
                if len(classes) == 2:
                    c0_mask = y == classes[0]
                    c1_mask = y == classes[1]
                    mu0 = X_num[c0_mask].mean()
                    mu1 = X_num[c1_mask].mean()
                    std_pool = X_num.std().replace(0, 1e-8)
                    mahal_approx = float(np.sqrt(((mu0 - mu1) ** 2 / std_pool ** 2).mean()))
                    # Low distance → high overlap → high sensitivity
                    overlap_score = max(0.0, 1.0 - min(1.0, mahal_approx / 3.0))
            except Exception:
                pass

        # Feature variance ratio
        var_ratio_score = 0.0
        if len(X_num.columns) > 0:
            variances = X_num.var()
            if variances.max() > 0:
                var_ratio_score = min(1.0, float(variances.max() / (variances.mean() + 1e-8)) / 100.0)

        # Weighted combination
        sensitivity = (
            0.35 * ir_score
            + 0.35 * overlap_score
            + 0.20 * size_score
            + 0.10 * var_ratio_score
        )

        factors = {
            "imbalance_ratio_score": float(ir_score),
            "class_overlap_score": float(overlap_score),
            "minority_size_score": float(size_score),
            "feature_variance_ratio_score": float(var_ratio_score),
        }
        return float(np.clip(sensitivity, 0.0, 1.0)), factors

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_report(self, report: DatasetReport) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"dataset_inspection_{report.dataset_name}_{ts}.json"
        out_path = self._output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info("Dataset report written to %s", out_path)
        return out_path
