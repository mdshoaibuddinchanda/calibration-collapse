"""
Calibration-Recall Frontier plot — the central visualization of the paper.

X-axis: minority recall
Y-axis: minority ECE
Each point = one (resampler, calibrator, model) combination.

This plot does not exist in any existing paper. It is the visual proof
of the paper's central claim: resampling improves recall at the cost of
minority-class calibration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger(__name__)

# Marker styles per resampler
_RESAMPLER_MARKERS = {
    "none": "o",
    "smote": "s",
    "adasyn": "^",
    "borderline_smote": "D",
    "class_weight": "P",
}

# Color map per model
_MODEL_COLORS = {
    "logistic_regression": "#2196F3",
    "random_forest": "#4CAF50",
    "mlp": "#FF5722",
    "gradient_boosting": "#9C27B0",
}


class CalibrationRecallFrontier:
    """
    Generates the Calibration-Recall Frontier plot from experiment results.

    Each point represents one (model, resampler, calibrator) combination.
    The Pareto frontier (best recall for each ECE level) is highlighted.
    """

    def plot_from_results(
        self,
        results: list[dict],
        dataset: str = "unknown",
        experiment_id: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Parameters
        ----------
        results : list of dicts, each with keys:
            - recall_minority (float)
            - ece_minority (float)
            - model (str)
            - resampler (str)
            - calibrator (str)
            - method_label (str, optional)
        """
        if not results:
            logger.warning("No results to plot for frontier.")
            return None

        fig, ax = plt.subplots(figsize=(11, 7))

        # Plot each point
        for r in results:
            recall = r.get("recall_minority", float("nan"))
            ece = r.get("ece_minority", float("nan"))
            model = r.get("model", "unknown")
            resampler = r.get("resampler", "none")
            calibrator = r.get("calibrator", "none")

            if np.isnan(recall) or np.isnan(ece):
                continue

            color = _MODEL_COLORS.get(model, "#607D8B")
            marker = _RESAMPLER_MARKERS.get(resampler, "x")
            label = r.get("method_label", f"{model}+{resampler}+{calibrator}")

            ax.scatter(
                recall, ece,
                c=color, marker=marker, s=80, alpha=0.8,
                edgecolors="white", linewidths=0.5,
                zorder=4,
            )

        # Pareto frontier (min ECE for each recall level)
        valid = [
            (r["recall_minority"], r["ece_minority"])
            for r in results
            if not (np.isnan(r.get("recall_minority", float("nan")))
                    or np.isnan(r.get("ece_minority", float("nan"))))
        ]
        if len(valid) > 2:
            valid_sorted = sorted(valid, key=lambda x: x[0])
            pareto = self._pareto_frontier(valid_sorted)
            if len(pareto) > 1:
                px, py = zip(*pareto)
                ax.plot(px, py, "k--", linewidth=1.5, alpha=0.6, label="Pareto frontier", zorder=5)

        # Legend for models (colors)
        model_patches = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=9, label=m)
            for m, c in _MODEL_COLORS.items()
        ]
        # Legend for resamplers (markers)
        resampler_patches = [
            plt.Line2D([0], [0], marker=mk, color="#607D8B", markersize=9,
                       linestyle="None", label=rs)
            for rs, mk in _RESAMPLER_MARKERS.items()
        ]

        legend1 = ax.legend(
            handles=model_patches, title="Model", loc="upper left",
            fontsize=8, title_fontsize=9,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=resampler_patches, title="Resampler", loc="upper right",
            fontsize=8, title_fontsize=9,
        )

        ax.set_xlabel("Minority Recall", fontsize=12)
        ax.set_ylabel("Minority ECE (↓ better)", fontsize=12)
        ax.set_title(
            f"Calibration-Recall Frontier — {dataset}\n"
            "(Each point = one model × resampler × calibrator combination)",
            fontsize=11,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.01, None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotation: ideal region
        ax.annotate(
            "Ideal region\n(high recall, low ECE)",
            xy=(0.85, 0.02), fontsize=8, color="green",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
        )

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{experiment_id}_{dataset}_cal_recall_frontier.png"
            out_path = output_dir / fname
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Calibration-Recall Frontier saved to %s", out_path)
            return out_path

        plt.close(fig)
        return None

    def plot_from_metrics_dir(
        self,
        metrics_dir: Path,
        dataset: str,
        experiment_id: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Load results from JSON files in metrics_dir and plot."""
        results = []
        cal_dir = metrics_dir / "calibration"
        cls_dir = metrics_dir / "classification"

        if not cal_dir.exists():
            logger.warning("Calibration metrics dir not found: %s", cal_dir)
            return None

        for cal_file in cal_dir.glob(f"{experiment_id}_{dataset}_*.json"):
            try:
                with open(cal_file) as fh:
                    cal = json.load(fh)
                method = cal.get("method", "unknown")
                parts = method.split("+")
                results.append({
                    "recall_minority": float("nan"),  # will be filled from cls metrics
                    "ece_minority": cal.get("ece_minority", float("nan")),
                    "model": parts[0] if len(parts) > 0 else "unknown",
                    "resampler": parts[1] if len(parts) > 1 else "none",
                    "calibrator": parts[2] if len(parts) > 2 else "none",
                    "method_label": method,
                })
            except Exception as exc:
                logger.warning("Could not load %s: %s", cal_file, exc)

        # Fill recall from classification metrics
        for r in results:
            cls_file = cls_dir / f"{experiment_id}_{dataset}_classification_metrics.json"
            if cls_file.exists():
                try:
                    with open(cls_file) as fh:
                        cls = json.load(fh)
                    if cls.get("method") == r["method_label"]:
                        r["recall_minority"] = cls.get("recall_minority", float("nan"))
                except Exception:
                    pass

        return self.plot_from_results(results, dataset, experiment_id, output_dir)

    @staticmethod
    def _pareto_frontier(
        points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Extract Pareto-optimal points (max recall, min ECE).
        Points sorted by recall ascending.
        """
        pareto = []
        min_ece = float("inf")
        for recall, ece in points:
            if ece <= min_ece:
                pareto.append((recall, ece))
                min_ece = ece
        return pareto
