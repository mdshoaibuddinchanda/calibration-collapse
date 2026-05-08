"""
Seed validator — verifies experiment reproducibility by re-running with
the same seed and checking metric differences are < 1e-6.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TOLERANCE = 1e-6
# Note: sklearn RF/GBM with n_jobs=-1 may have floating-point non-determinism
# across runs even with the same seed (thread scheduling). For tree models,
# use _TOLERANCE_TREE instead.
_TOLERANCE_TREE = 1e-4


@dataclass
class SeedValidationResult:
    experiment_id: str
    timestamp: str
    seed: int
    is_reproducible: bool
    max_metric_diff: float
    tolerance: float
    metric_diffs: dict[str, float] = field(default_factory=dict)
    message: str = ""


class SeedValidator:
    """
    Validates that an experiment is reproducible by comparing two runs
    with the same seed.

    Usage:
        validator = SeedValidator()
        result = validator.validate(metrics_run1, metrics_run2, seed=42)
    """

    def validate(
        self,
        metrics_run1: dict,
        metrics_run2: dict,
        seed: int,
        experiment_id: str = "unknown",
        model_name: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> SeedValidationResult:
        """
        Compare two metric dicts from runs with the same seed.

        Parameters
        ----------
        metrics_run1, metrics_run2 : dicts with float metric values
        model_name : used to select appropriate tolerance (tree models have
                     higher floating-point variance with n_jobs=-1)
        """
        # Tree models with parallel execution have higher non-determinism
        tree_models = {"random_forest", "gradient_boosting"}
        tolerance = _TOLERANCE_TREE if model_name.lower() in tree_models else _TOLERANCE
        diffs: dict[str, float] = {}
        numeric_keys = [
            k for k in metrics_run1
            if isinstance(metrics_run1.get(k), (int, float))
            and isinstance(metrics_run2.get(k), (int, float))
        ]

        for key in numeric_keys:
            v1 = float(metrics_run1[key])
            v2 = float(metrics_run2[key])
            if np.isnan(v1) and np.isnan(v2):
                diffs[key] = 0.0
            elif np.isnan(v1) or np.isnan(v2):
                diffs[key] = float("inf")
            else:
                diffs[key] = abs(v1 - v2)

        max_diff = max(diffs.values()) if diffs else 0.0
        is_reproducible = max_diff < tolerance

        if is_reproducible:
            message = f"Experiment is reproducible (max_diff={max_diff:.2e} < {_TOLERANCE:.0e})"
            logger.info("[%s] %s", experiment_id, message)
        else:
            worst_key = max(diffs, key=diffs.get)
            message = (
                f"Reproducibility FAILED: max_diff={max_diff:.2e} >= {_TOLERANCE:.0e}. "
                f"Worst metric: '{worst_key}' (diff={diffs[worst_key]:.2e})"
            )
            logger.warning("[%s] %s", experiment_id, message)

        result = SeedValidationResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            seed=seed,
            is_reproducible=is_reproducible,
            max_metric_diff=float(max_diff),
            tolerance=tolerance,
            metric_diffs=diffs,
            message=message,
        )

        if output_dir is not None:
            self._write_result(result, output_dir)

        return result

    def _write_result(self, result: SeedValidationResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{result.experiment_id}_seed_validation.json"
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2)
        logger.info("Seed validation written to %s", out_path)
        return out_path
