"""
Publication-quality per-class reliability diagrams.

Always generates side-by-side subplots:
  Left:  Global reliability diagram
  Right: Minority class reliability diagram

The gap from the perfect calibration line is shaded.
A confidence histogram is overlaid at the bottom of each subplot.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


class ReliabilityDiagram:
    """
    Generates per-class reliability diagrams from calibration bin data.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._n_bins = n_bins

    def plot(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        proba_cal: Optional[np.ndarray] = None,
        experiment_id: str = "unknown",
        dataset: str = "unknown",
        method: str = "unknown",
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate and save reliability diagram.

        Parameters
        ----------
        proba     : uncalibrated probabilities (n,) or (n, 2)
        y_true    : true labels
        proba_cal : calibrated probabilities (optional, overlaid)
        """
        p = proba[:, 1] if proba.ndim == 2 else proba
        p_cal = (proba_cal[:, 1] if proba_cal is not None and proba_cal.ndim == 2
                 else proba_cal)
        y = y_true.astype(int)

        classes = np.unique(y)
        counts = {int(c): int((y == c).sum()) for c in classes}
        minority_class = int(min(counts, key=counts.get))

        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"Reliability Diagram — {dataset} | {method}",
                fontsize=13, fontweight="bold", y=1.01,
            )

            # Left: global
            self._draw_reliability(
                ax=axes[0],
                p=p,
                y=y,
                p_cal=p_cal,
                title="Global",
                color="#2196F3",
                cal_color="#FF5722",
            )

            # Right: minority class
            min_mask = y == minority_class
            self._draw_reliability(
                ax=axes[1],
                p=p[min_mask],
                y=y[min_mask],
                p_cal=p_cal[min_mask] if p_cal is not None else None,
                title=f"Minority Class (class={minority_class}, n={min_mask.sum()})",
                color="#4CAF50",
                cal_color="#FF5722",
            )

            plt.tight_layout()

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{experiment_id}_{dataset}_{method}_reliability.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("Reliability diagram saved to %s", out_path)
                return out_path

            plt.close(fig)
            return None

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_reliability(
        self,
        ax: plt.Axes,
        p: np.ndarray,
        y: np.ndarray,
        p_cal: Optional[np.ndarray],
        title: str,
        color: str,
        cal_color: str,
    ) -> None:
        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(self._n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (p >= lo) & (p <= hi if i == self._n_bins - 1 else p < hi)
            n_b = mask.sum()
            bin_counts.append(n_b)
            if n_b == 0:
                bin_accs.append(None)
                bin_confs.append(None)
            else:
                bin_accs.append(float(y[mask].mean()))
                bin_confs.append(float(p[mask].mean()))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", zorder=5)

        # Shade gap
        valid = [(c, a) for c, a in zip(bin_confs, bin_accs) if c is not None and a is not None]
        if valid:
            confs_v, accs_v = zip(*valid)
            ax.fill_between(
                confs_v, confs_v, accs_v,
                alpha=0.15, color=color, label="Calibration gap",
            )

        # Uncalibrated bars
        bar_width = 1.0 / self._n_bins * 0.8
        for i, (conf, acc) in enumerate(zip(bin_confs, bin_accs)):
            if conf is None:
                continue
            ax.bar(
                bin_centers[i], acc, width=bar_width,
                color=color, alpha=0.7, edgecolor="white", linewidth=0.5,
            )

        # Calibrated overlay
        if p_cal is not None:
            cal_accs = []
            cal_confs = []
            for i in range(self._n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (p_cal >= lo) & (p_cal <= hi if i == self._n_bins - 1 else p_cal < hi)
                if mask.sum() == 0:
                    cal_accs.append(None)
                    cal_confs.append(None)
                else:
                    cal_accs.append(float(y[mask].mean()))
                    cal_confs.append(float(p_cal[mask].mean()))

            valid_cal = [(c, a) for c, a in zip(cal_confs, cal_accs) if c is not None and a is not None]
            if valid_cal:
                confs_c, accs_c = zip(*valid_cal)
                ax.plot(confs_c, accs_c, "o-", color=cal_color, linewidth=2,
                        markersize=5, label="Calibrated", zorder=6)

        # Confidence histogram (bottom)
        ax2 = ax.twinx()
        ax2.bar(
            bin_centers,
            [c / max(sum(bin_counts), 1) for c in bin_counts],
            width=bar_width * 0.6,
            color="gray", alpha=0.25, label="Sample density",
        )
        ax2.set_ylabel("Sample fraction", fontsize=9, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
        ax2.set_ylim(0, 1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
