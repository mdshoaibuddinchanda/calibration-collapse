"""
Uncertainty-band reliability diagram.

Shows mean reliability curve ± std shading across multiple seeds.
This is the key visualization for the paper: calibration instability
is shown as variance in the reliability curve, not just a single line.

Generates side-by-side:
  Left:  Global reliability with uncertainty bands
  Right: Minority-class reliability with uncertainty bands

The width of the shaded band directly visualizes calibration instability —
a core claim of the paper.
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

_STYLE = {
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


class UncertaintyReliabilityDiagram:
    """
    Generates reliability diagrams with ±std uncertainty bands from
    multi-seed bin data.

    The shaded band width is the key scientific signal: wide bands indicate
    calibration instability, narrow bands indicate stability.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._n_bins = n_bins

    def plot_from_multiseed_report(
        self,
        report,  # MultiSeedReport
        output_dir: Optional[Path] = None,
        compare_methods: Optional[list] = None,  # list of MultiSeedReport for comparison
    ) -> Optional[Path]:
        """
        Generate uncertainty-band reliability diagram from a MultiSeedReport.

        Parameters
        ----------
        report          : MultiSeedReport with reliability_data_per_seed
        compare_methods : optional list of additional MultiSeedReports to overlay
        """
        if not report.reliability_data_per_seed:
            logger.warning("No per-seed reliability data in report for %s", report.dataset)
            return None

        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            method_label = f"{report.model}+{report.resampler}+{report.calibrator}"
            fig.suptitle(
                f"Reliability Diagram (n={len(report.reliability_data_per_seed)} seeds) — "
                f"{report.dataset} | {method_label}",
                fontsize=12, fontweight="bold", y=1.01,
            )

            # Draw main method with uncertainty bands
            self._draw_uncertainty_band(
                ax=axes[0],
                reliability_data=report.reliability_data_per_seed,
                class_key="global",
                title="Global",
                color="#2196F3",
                label=method_label,
            )
            self._draw_uncertainty_band(
                ax=axes[1],
                reliability_data=report.reliability_data_per_seed,
                class_key="minority",
                title=f"Minority Class (class={report.calibrator})",
                color="#FF5722",
                label=method_label,
            )

            # Overlay comparison methods if provided
            comparison_colors = ["#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
            if compare_methods:
                for i, comp_report in enumerate(compare_methods[:4]):
                    if not comp_report.reliability_data_per_seed:
                        continue
                    comp_label = f"{comp_report.model}+{comp_report.resampler}+{comp_report.calibrator}"
                    color = comparison_colors[i % len(comparison_colors)]
                    self._draw_uncertainty_band(
                        ax=axes[0],
                        reliability_data=comp_report.reliability_data_per_seed,
                        class_key="global",
                        title="Global",
                        color=color,
                        label=comp_label,
                        alpha_band=0.08,
                        linewidth=1.5,
                    )
                    self._draw_uncertainty_band(
                        ax=axes[1],
                        reliability_data=comp_report.reliability_data_per_seed,
                        class_key="minority",
                        title="Minority",
                        color=color,
                        label=comp_label,
                        alpha_band=0.08,
                        linewidth=1.5,
                    )

            for ax in axes:
                ax.legend(fontsize=8, loc="upper left")

            plt.tight_layout()

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                safe = method_label.replace("+", "_")[:60]
                fname = f"{report.experiment_id}_{report.dataset}_{safe}_uncertainty_reliability.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("Uncertainty reliability diagram saved to %s", out_path)
                return out_path

            plt.close(fig)
            return None

    def plot_instability_comparison(
        self,
        reports: list,  # list of MultiSeedReport, one per method
        dataset: str,
        experiment_id: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Side-by-side comparison of calibration instability across methods.
        Each method gets its own uncertainty band on the minority reliability plot.
        This is the paper's key figure: showing that SMOTE widens the band.
        """
        if not reports:
            return None

        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 7))

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
                    label="Perfect calibration", zorder=10, alpha=0.7)

            for i, report in enumerate(reports):
                if not report.reliability_data_per_seed:
                    continue
                color = colors[i % len(colors)]
                label = f"{report.resampler}+{report.calibrator}"
                self._draw_uncertainty_band(
                    ax=ax,
                    reliability_data=report.reliability_data_per_seed,
                    class_key="minority",
                    title="",
                    color=color,
                    label=label,
                    alpha_band=0.12,
                    linewidth=2.0,
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted confidence P(y=1)", fontsize=12)
            ax.set_ylabel("Fraction of true minority samples", fontsize=12)
            ax.set_title(
                f"Minority-Class Calibration Instability — {dataset}\n"
                f"(Shaded band = ±1 std across {len(reports[0].seeds)} seeds)",
                fontsize=11,
            )
            ax.legend(fontsize=9, loc="upper left")

            plt.tight_layout()

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{experiment_id}_{dataset}_instability_comparison.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("Instability comparison saved to %s", out_path)
                return out_path

            plt.close(fig)
            return None

    # ------------------------------------------------------------------
    # Core drawing
    # ------------------------------------------------------------------

    def _draw_uncertainty_band(
        self,
        ax: plt.Axes,
        reliability_data: list[dict],
        class_key: str,
        title: str,
        color: str,
        label: str,
        alpha_band: float = 0.15,
        linewidth: float = 2.0,
    ) -> None:
        """
        Draw mean reliability curve with ±std shading from per-seed bin data.
        """
        # Collect acc values per bin across seeds
        n_bins = self._n_bins
        bin_accs_per_seed: list[list[Optional[float]]] = []
        bin_confs_per_seed: list[list[Optional[float]]] = []

        for seed_data in reliability_data:
            bin_data = seed_data.get("bin_data", {}).get(class_key, [])
            if not bin_data:
                continue
            accs = [b.get("acc") for b in bin_data]
            confs = [b.get("conf") for b in bin_data]
            bin_accs_per_seed.append(accs)
            bin_confs_per_seed.append(confs)

        if not bin_accs_per_seed:
            return

        # Compute mean and std per bin
        n_bins_actual = len(bin_accs_per_seed[0])
        mean_accs = []
        std_accs = []
        mean_confs = []

        for b in range(n_bins_actual):
            accs_b = [s[b] for s in bin_accs_per_seed if s[b] is not None]
            confs_b = [s[b] for s in bin_confs_per_seed if s[b] is not None]
            if accs_b:
                mean_accs.append(float(np.mean(accs_b)))
                std_accs.append(float(np.std(accs_b, ddof=1)) if len(accs_b) > 1 else 0.0)
                mean_confs.append(float(np.mean(confs_b)) if confs_b else float(b / n_bins_actual))
            else:
                mean_accs.append(None)
                std_accs.append(None)
                mean_confs.append(None)

        # Filter to non-None bins
        valid = [(c, a, s) for c, a, s in zip(mean_confs, mean_accs, std_accs)
                 if c is not None and a is not None]
        if not valid:
            return

        confs_v, accs_v, stds_v = zip(*valid)
        confs_arr = np.array(confs_v)
        accs_arr = np.array(accs_v)
        stds_arr = np.array(stds_v)

        # Perfect calibration line (only draw once)
        if title:
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
                    label="Perfect calibration", zorder=5, alpha=0.6)

        # Mean curve
        ax.plot(confs_arr, accs_arr, "o-", color=color, linewidth=linewidth,
                markersize=4, label=label, zorder=6, alpha=0.9)

        # ±std uncertainty band
        ax.fill_between(
            confs_arr,
            np.clip(accs_arr - stds_arr, 0, 1),
            np.clip(accs_arr + stds_arr, 0, 1),
            alpha=alpha_band, color=color,
        )

        if title:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted confidence")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(title, fontsize=11)
