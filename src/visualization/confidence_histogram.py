"""
Confidence histogram — shows distribution of predicted probabilities
separately for minority and majority classes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceHistogram:
    """
    Plots the distribution of predicted P(y=1) for each class.
    A well-calibrated model should show minority class concentrated near 1
    and majority class concentrated near 0.
    """

    def plot(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        output_dir: Optional[Path] = None,
        n_bins: int = 30,
    ) -> Optional[Path]:
        p = proba[:, 1] if proba.ndim == 2 else proba
        y = y_true.astype(int)

        classes = np.unique(y)
        counts = {int(c): int((y == c).sum()) for c in classes}
        minority_class = int(min(counts, key=counts.get))
        majority_class = int(max(counts, key=counts.get))

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            p[y == majority_class], bins=n_bins, range=(0, 1),
            alpha=0.6, color="#2196F3", label=f"Majority (class={majority_class}, n={counts[majority_class]})",
            density=True,
        )
        ax.hist(
            p[y == minority_class], bins=n_bins, range=(0, 1),
            alpha=0.7, color="#FF5722", label=f"Minority (class={minority_class}, n={counts[minority_class]})",
            density=True,
        )

        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
        ax.set_xlabel("Predicted P(y=1)")
        ax.set_ylabel("Density")
        ax.set_title(f"Confidence Distribution — {dataset} | {method}")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{experiment_id}_{dataset}_{method}_confhist.png"
            out_path = output_dir / fname
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Confidence histogram saved to %s", out_path)
            return out_path

        plt.close(fig)
        return None
