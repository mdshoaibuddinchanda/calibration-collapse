"""
Multi-seed validation framework.

Runs the same (dataset, model, resampler, calibrator) combination across
multiple seeds and computes:
  - mean and std of every metric
  - 95% confidence intervals (t-distribution, small sample)
  - coefficient of variation (CV) as a stability indicator
  - per-seed reliability diagram data for uncertainty band plots

This is the primary statistical robustness mechanism for the paper.
Without multi-seed results, all calibration claims are single-point estimates
with unknown variance — indefensible to reviewers.

Usage
-----
    validator = MultiSeedValidator(seeds=[0, 1, 2, 3, 4], project_root=ROOT)
    report = validator.run(
        dataset_name="credit_card",
        model_name="logistic_regression",
        resampler_name="smote",
        calibrator_name="temperature_scaling",
        experiment_id="exp001_multiseed",
        dataset_registry=registry,
    )
    print(report.summary_table())
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    """Statistics for a single metric across seeds."""
    metric_name: str
    values: list[float]          # one per seed
    mean: float
    std: float
    ci_lower: float              # 95% CI lower bound
    ci_upper: float              # 95% CI upper bound
    cv: float                    # coefficient of variation = std/mean
    n_seeds: int
    is_stable: bool              # CV < 0.15 → stable

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> "MetricStats":
        vals = [v for v in values if not (v != v)]  # remove NaN
        n = len(vals)
        if n == 0:
            return cls(name, values, float("nan"), float("nan"),
                       float("nan"), float("nan"), float("nan"), 0, False)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        # 95% CI using t-distribution (appropriate for small n)
        if n > 1:
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            margin = t_crit * std / np.sqrt(n)
        else:
            margin = 0.0
        cv = std / abs(mean) if abs(mean) > 1e-8 else float("nan")
        return cls(
            metric_name=name,
            values=values,
            mean=mean,
            std=std,
            ci_lower=mean - margin,
            ci_upper=mean + margin,
            cv=cv,
            n_seeds=n,
            is_stable=cv < 0.15 if not (cv != cv) else False,
        )

    def format(self) -> str:
        return (
            f"{self.mean:.4f} ± {self.std:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"CV={self.cv:.1%} {'✓' if self.is_stable else '⚠'}"
        )


@dataclass
class MultiSeedReport:
    """Complete multi-seed validation report for one (dataset, model, resampler, calibrator)."""
    experiment_id: str
    dataset: str
    model: str
    resampler: str
    calibrator: str
    seeds: list[int]
    timestamp: str
    metric_stats: dict[str, MetricStats] = field(default_factory=dict)
    # Per-seed reliability diagram data for uncertainty band plots
    reliability_data_per_seed: list[dict] = field(default_factory=list)
    failed_seeds: list[int] = field(default_factory=list)

    def summary_table(self) -> str:
        """Human-readable summary table."""
        lines = [
            f"\nMulti-Seed Validation: {self.dataset} | {self.model} | "
            f"{self.resampler} | {self.calibrator}",
            f"Seeds: {self.seeds} ({len(self.seeds) - len(self.failed_seeds)} succeeded, "
            f"{len(self.failed_seeds)} failed)",
            "-" * 80,
        ]
        key_metrics = [
            "ece_global", "ece_minority", "ece_majority",
            "brier_minority", "f1_minority", "recall_minority", "auc_roc",
        ]
        for key in key_metrics:
            if key in self.metric_stats:
                s = self.metric_stats[key]
                lines.append(f"  {key:<20} {s.format()}")
        lines.append("-" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        # MetricStats objects need special handling
        d["metric_stats"] = {k: asdict(v) for k, v in self.metric_stats.items()}
        return d

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        method = f"{self.model}+{self.resampler}+{self.calibrator}"
        safe = method.replace("+", "_")[:60]
        fname = f"{self.experiment_id}_{self.dataset}_{safe}_multiseed.json"
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Multi-seed report saved to %s", out_path)
        return out_path


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class MultiSeedValidator:
    """
    Runs a pipeline combination across multiple seeds and aggregates results.

    Reuses the existing ExperimentRunner infrastructure — each seed creates
    a fresh runner with that seed, runs the full pipeline, and collects metrics.
    """

    def __init__(
        self,
        seeds: list[int],
        project_root: Path,
        base_config: dict,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._seeds = seeds
        self._project_root = Path(project_root)
        self._base_config = base_config
        self._output_dir = output_dir or (self._project_root / "outputs" / "multiseed")

    def run(
        self,
        dataset_name: str,
        model_name: str,
        resampler_name: str,
        calibrator_name: str,
        experiment_id: str,
        dataset_registry,
    ) -> MultiSeedReport:
        """
        Run the combination across all seeds and return a MultiSeedReport.
        """
        from src.experiment.runner import ExperimentRunner

        logger.info(
            "Multi-seed validation: %s | %s | %s | %s | seeds=%s",
            dataset_name, model_name, resampler_name, calibrator_name, self._seeds,
        )

        all_cal_metrics: list[dict] = []
        all_cls_metrics: list[dict] = []
        reliability_data: list[dict] = []
        failed_seeds: list[int] = []

        for seed in self._seeds:
            seed_config = {**self._base_config, "seed": seed}
            runner = ExperimentRunner(
                project_root=self._project_root,
                config=seed_config,
            )
            try:
                result = runner.run(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    resampler_name=resampler_name,
                    calibrator_name=calibrator_name,
                    experiment_id=f"{experiment_id}_seed{seed}",
                    dataset_registry=dataset_registry,
                )
                all_cal_metrics.append(result.get("calibration", {}))
                all_cls_metrics.append(result.get("classification", {}))

                # Collect reliability diagram data for uncertainty bands
                cal = result.get("calibration", {})
                if cal.get("bin_data"):
                    reliability_data.append({
                        "seed": seed,
                        "bin_data": cal["bin_data"],
                        "n_bins_minority": cal.get("n_bins_minority", 10),
                    })

                logger.info(
                    "Seed %d: ECE_minority=%.4f, recall_minority=%.4f",
                    seed,
                    cal.get("ece_minority", float("nan")),
                    result.get("classification", {}).get("recall_minority", float("nan")),
                )

            except Exception as exc:
                logger.error("Seed %d failed: %s", seed, exc)
                failed_seeds.append(seed)

        # Aggregate metrics
        metric_stats = self._aggregate_metrics(all_cal_metrics, all_cls_metrics)

        report = MultiSeedReport(
            experiment_id=experiment_id,
            dataset=dataset_name,
            model=model_name,
            resampler=resampler_name,
            calibrator=calibrator_name,
            seeds=self._seeds,
            timestamp=datetime.now().isoformat(),
            metric_stats=metric_stats,
            reliability_data_per_seed=reliability_data,
            failed_seeds=failed_seeds,
        )

        report.save(self._output_dir)
        logger.info(report.summary_table())
        return report

    def _aggregate_metrics(
        self,
        cal_metrics: list[dict],
        cls_metrics: list[dict],
    ) -> dict[str, MetricStats]:
        """Compute MetricStats for every numeric metric across seeds."""
        stats_dict: dict[str, MetricStats] = {}

        cal_keys = [
            "ece_global", "ece_minority", "ece_majority",
            "brier_global", "brier_minority", "brier_majority",
            "ace_global", "ace_minority", "ace_majority",
        ]
        cls_keys = [
            "f1_minority", "f1_majority", "f1_macro",
            "precision_minority", "recall_minority",
            "auc_roc", "auc_pr",
        ]

        for key in cal_keys:
            values = [float(m[key]) for m in cal_metrics if key in m and m[key] is not None]
            if values:
                stats_dict[key] = MetricStats.from_values(key, values)

        for key in cls_keys:
            values = [float(m[key]) for m in cls_metrics if key in m and m[key] is not None]
            if values:
                stats_dict[key] = MetricStats.from_values(key, values)

        return stats_dict
