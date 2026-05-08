
"""
Paper Output Generator — Clean Academic Version
================================================
Principles:
  - No text/boxes overlapping data lines
  - No inline value annotations (values belong in tables)
  - No arrows or decorative annotations on plots
  - Legends always outside the plot area (right side or bottom)
  - Journal-grade font sizes (11pt base)
  - Minimal ink, maximum clarity

Run from project root:
    python paper_output/generate_paper.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT  = Path(__file__).parent
OUT.mkdir(exist_ok=True)

MULTISEED_DIR = ROOT / "outputs" / "multiseed"
METRICS_CAL   = ROOT / "outputs" / "metrics" / "calibration"
ABLATION_DIR  = ROOT / "reports" / "ablations"
SYNTHETIC_DIR = ROOT / "datasets" / "synthetic"

# ── Journal-grade style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "font.family":           "DejaVu Sans",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#BBBBBB",
    "legend.borderpad":      0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,
    "grid.alpha":            0.20,
    "grid.linewidth":        0.5,
    "lines.linewidth":       1.6,
    "lines.markersize":      5,
    "errorbar.capsize":      2,
    "figure.constrained_layout.use": False,
})

# ── Colour palette (muted, print-safe) ───────────────────────────────────────
COLORS = {
    "lr_none":   "#3A6EA5",
    "lr_smote":  "#C0392B",
    "lr_cw":     "#27AE60",
    "lr_ts":     "#8E44AD",
    "lr_pcdm":   "#D4AC0D",
    "rf_none":   "#85C1E9",
    "rf_smote":  "#F1948A",
    "rf_pcdm":   "#82E0AA",
}
MARKERS = {
    "lr_none":"o", "lr_smote":"s", "lr_cw":"^",
    "lr_ts":"D",   "lr_pcdm":"*",
    "rf_none":"o", "rf_smote":"s", "rf_pcdm":"*",
}

DS_LABEL = {
    "pima":                     "Pima (IR=1.9)",
    "phoneme":                  "Phoneme (IR=2.4)",
    "default_credit_card_clients": "Default Credit Card Clients (IR≈11.6)",
    "extreme_imbalance_severe": "Extreme Imb. (IR=101)",
}
LEGACY_DATASET_ALIASES = {
    "default_credit_card_clients": "credit_card",
}
M_LABEL = {
    "logistic_regression+none+none":                 "LR + None",
    "logistic_regression+smote+none":                "LR + SMOTE",
    "logistic_regression+class_weight+none":         "LR + ClassWt",
    "logistic_regression+smote+temperature_scaling": "LR + SMOTE + TS",
    "logistic_regression+smote+per_class_adaptive":  "LR + SMOTE + PCDM",
    "random_forest+none+none":                       "RF + None",
    "random_forest+smote+none":                      "RF + SMOTE",
    "random_forest+class_weight+none":               "RF + ClassWt",
    "random_forest+smote+temperature_scaling":       "RF + SMOTE + TS",
    "random_forest+smote+per_class_adaptive":        "RF + SMOTE + PCDM",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_table():
    p = MULTISEED_DIR / "controlled_validation_paper_table.json"
    return json.load(open(p)) if p.exists() else {}

def gm(table, method, ds, metric):
    for key in (ds, LEGACY_DATASET_ALIASES.get(ds)):
        if not key:
            continue
        try:
            d = table[method][key][metric]
            return float(d["mean"]), float(d["std"])
        except (KeyError, TypeError):
            continue
    return float("nan"), float("nan")

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {stem}.pdf / .png")
    plt.close(fig)

def load_ece(dataset_name, suffix="logistic_regression_none_none"):
    candidates = [dataset_name]
    legacy = LEGACY_DATASET_ALIASES.get(dataset_name)
    if legacy:
        candidates.append(legacy)
    files = []
    for candidate in candidates:
        files = sorted(METRICS_CAL.glob(f"*{candidate}*{suffix}*calibration_metrics.json"))
        if files:
            break
    if not files:
        return float("nan")
    return float(json.load(open(files[0])).get("ece_minority", float("nan")))

def load_multiseed(ds, method_key):
    safe = method_key.replace("+", "_")
    candidates = [ds]
    legacy = LEGACY_DATASET_ALIASES.get(ds)
    if legacy:
        candidates.append(legacy)
    for candidate in candidates:
        fname = f"controlled_validation_{candidate}_{candidate}_{safe}_multiseed.json"
        p = MULTISEED_DIR / fname
        if p.exists():
            return json.load(open(p))
    return {}


# =============================================================================
# TABLES — CSV only
# =============================================================================

def make_tables(table):
    print("\n── Tables ──────────────────────────────────────────────────────")
    methods = [
        "logistic_regression+none+none",
        "logistic_regression+smote+none",
        "logistic_regression+class_weight+none",
        "logistic_regression+smote+temperature_scaling",
        "logistic_regression+smote+per_class_adaptive",
        "random_forest+none+none",
        "random_forest+smote+none",
        "random_forest+smote+per_class_adaptive",
    ]
    datasets_main = ["pima", "phoneme", "default_credit_card_clients"]

    # Table 1
    pd.DataFrame([
        ["Pima",               "Real",       768,   8,   1.9,  268,  "Mild imbalance baseline"],
        ["Phoneme",            "Real",      5404,   5,   2.4, 1586,  "Moderate imbalance"],
        ["Default Credit Card Clients", "Real", 30000, 23, 11.6, "",  "Real credit dataset"],
        ["Extreme Imb. (syn.)","Synthetic", 5000,  10, 101.0,   49,  "IR stress (controlled)"],
        ["Conf. Collapse (syn.)","Synthetic",5000, 12,  20.0,  238,  "Confidence instability"],
    ], columns=["Dataset","Type","Samples","Features","IR","Minority_n","Role"]
    ).to_csv(OUT / "table1_dataset_summary.csv", index=False)
    print("  Saved: table1_dataset_summary.csv")

    # Table 2 — ECE_minority
    rows = []
    for m in methods:
        row = {"Method": M_LABEL.get(m, m)}
        for ds in datasets_main:
            mn, sd = gm(table, m, ds, "ece_minority")
            row[f"{ds}_mean"] = round(mn, 4) if not np.isnan(mn) else ""
            row[f"{ds}_std"]  = round(sd, 4) if not np.isnan(sd) else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "table2_main_results.csv", index=False)
    print("  Saved: table2_main_results.csv")

    # Table 3 — Tradeoff
    rows = []
    for m in methods[:5]:
        row = {"Method": M_LABEL.get(m, m)}
        for ds in ["pima", "phoneme"]:
            rm, rs = gm(table, m, ds, "recall_minority")
            em, es = gm(table, m, ds, "ece_minority")
            row[f"{ds}_recall"] = f"{rm:.3f}±{rs:.3f}" if not np.isnan(rm) else ""
            row[f"{ds}_ece_min"] = f"{em:.3f}±{es:.3f}" if not np.isnan(em) else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "table3_tradeoff.csv", index=False)
    print("  Saved: table3_tradeoff.csv")

    # Table 4 — Stability
    rows = []
    for m in methods:
        for ds in ["pima", "phoneme"]:
            try:
                d = table[m][ds]["ece_minority"]
                rows.append({
                    "Method": M_LABEL.get(m, m), "Dataset": ds,
                    "Mean": round(d["mean"], 4), "Std": round(d["std"], 4),
                    "CV": round(d["cv"], 4) if d["cv"] == d["cv"] else "",
                    "CI_lower": round(d["ci_lower"], 4),
                    "CI_upper": round(d["ci_upper"], 4),
                    "Stable": d["is_stable"], "N_seeds": d["n_seeds"],
                })
            except (KeyError, TypeError):
                pass
    pd.DataFrame(rows).to_csv(OUT / "table4_stability.csv", index=False)
    print("  Saved: table4_stability.csv")

    # Table 5 — Ablation
    p = ABLATION_DIR / "exp001_smote_ratios_ablation_summary.csv"
    if p.exists():
        pd.read_csv(p).to_csv(OUT / "table5_ablation.csv", index=False)
        print("  Saved: table5_ablation.csv")

    # Appendix table
    pd.DataFrame([
        ["SMOTE ratio=0.5", "pima",    "LR/RF", 0.536, 0.5, "Current ratio >= requested"],
        ["SMOTE ratio=2.0", "pima",    "LR/RF", 0.536, 2.0, "Unstable near IR=1.9"],
        ["SMOTE ratio=2.0", "phoneme", "LR/RF", 0.416, 2.0, "Unstable near IR=2.4"],
    ], columns=["Config","Dataset","Model","Current_ratio","Requested_ratio","Reason"]
    ).to_csv(OUT / "appendix_table_skipped.csv", index=False)
    print("  Saved: appendix_table_skipped.csv")


# =============================================================================
# FIGURE 1 — Calibration Gap
# =============================================================================

def make_fig1(table):
    print("\n[Fig 1] Calibration Gap")
    methods_plot = [
        ("logistic_regression+none+none",                "LR+None",       "#3A6EA5"),
        ("logistic_regression+smote+none",               "LR+SMOTE",      "#C0392B"),
        ("logistic_regression+smote+per_class_adaptive", "LR+SMOTE+PCDM", "#D4AC0D"),
        ("random_forest+none+none",                      "RF+None",       "#85C1E9"),
        ("random_forest+smote+none",                     "RF+SMOTE",      "#F1948A"),
    ]
    datasets_plot = ["pima", "phoneme", "default_credit_card_clients"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.30, left=0.07, right=0.97)

    for ax, ds in zip(axes, datasets_plot):
        x = np.arange(len(methods_plot)) * 1.3   # wider spacing between groups
        w = 0.32
        g_m  = [gm(table, m, ds, "ece_global")[0]   for m, _, _ in methods_plot]
        g_s  = [gm(table, m, ds, "ece_global")[1]   for m, _, _ in methods_plot]
        mi_m = [gm(table, m, ds, "ece_minority")[0] for m, _, _ in methods_plot]
        mi_s = [gm(table, m, ds, "ece_minority")[1] for m, _, _ in methods_plot]

        ax.bar(x - w/2, g_m,  w, yerr=g_s,  capsize=2,
               color="#5B9BD5", alpha=0.75, label="ECE_global",
               error_kw={"elinewidth": 1.0, "ecolor": "#555"})
        ax.bar(x + w/2, mi_m, w, yerr=mi_s, capsize=2,
               color="#E67E22", alpha=0.75, label="ECE_minority",
               error_kw={"elinewidth": 1.0, "ecolor": "#555"})

        ax.set_title(DS_LABEL[ds], fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in methods_plot],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ECE  (↓ better)" if ax is axes[0] else "", fontsize=9)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.20)

    # Single legend below all subplots
    handles = [
        plt.Rectangle((0,0),1,1, color="#5B9BD5", alpha=0.75),
        plt.Rectangle((0,0),1,1, color="#E67E22", alpha=0.75),
    ]
    fig.legend(handles, ["ECE_global", "ECE_minority"],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), frameon=True)
    fig.suptitle(
        "Figure 1 — Global ECE vs Minority-Class ECE  (error bars = ±std, 5 seeds)",
        fontsize=10, fontweight="bold",
    )
    save(fig, "fig1_calibration_gap")


# =============================================================================
# FIGURE 2 — Reliability Diagrams
# =============================================================================

def _draw_reliability(ax, ds, class_key, methods_info):
    """Draw reliability curve with ±std band. No overlapping elements."""
    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--",
            label="Perfect calibration", zorder=10)

    for m_key, label, color in methods_info:
        report    = load_multiseed(ds, m_key)
        seed_data = report.get("reliability_data_per_seed", [])
        if not seed_data:
            continue
        per_a, per_c = [], []
        for sd in seed_data:
            bd = sd.get("bin_data", {}).get(class_key, [])
            if not bd:
                continue
            per_a.append([b.get("acc")  for b in bd])
            per_c.append([b.get("conf") for b in bd])
        if not per_a:
            continue
        n_b = len(per_a[0])
        ma, sa, mc = [], [], []
        for b in range(n_b):
            av = [s[b] for s in per_a if s[b] is not None]
            cv = [s[b] for s in per_c if s[b] is not None]
            if av:
                ma.append(float(np.mean(av)))
                sa.append(float(np.std(av, ddof=1)) if len(av) > 1 else 0.0)
                mc.append(float(np.mean(cv)) if cv else b / n_b)
        valid = [(c, a, s) for c, a, s in zip(mc, ma, sa)
                 if c is not None and a is not None]
        if not valid:
            continue
        cv_a, av_a, sv_a = (np.array(v) for v in zip(*valid))
        ax.plot(cv_a, av_a, "o-", color=color, lw=1.6, ms=4,
                label=label, zorder=6)
        ax.fill_between(cv_a,
                        np.clip(av_a - sv_a, 0, 1),
                        np.clip(av_a + sv_a, 0, 1),
                        alpha=0.10, color=color, zorder=5)

    # Clean axis limits with padding so nothing clips
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.18)


def make_fig2():
    print("\n[Fig 2] Reliability Diagrams with Uncertainty Bands")
    methods_info = [
        ("logistic_regression+none+none",                "LR + None",        "#3A6EA5"),
        ("logistic_regression+smote+none",               "LR + SMOTE",       "#C0392B"),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE + PCDM","#D4AC0D"),
    ]
    # Only global reliability — minority subplot pins at y=1.0 and adds no information
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.12, right=0.68)

    _draw_reliability(ax, "phoneme", "global", methods_info)
    ax.set_title("Global Reliability  (Phoneme)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean predicted confidence", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)

    # Legend to the right — outside the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center left", bbox_to_anchor=(0.70, 0.50),
               fontsize=9, frameon=True, title="Method", title_fontsize=9)

    fig.suptitle(
        "Figure 2 — Reliability Diagram with Uncertainty Bands\n"
        "(Phoneme, 5 seeds, shaded = ±1 std)",
        fontsize=10, fontweight="bold",
    )
    save(fig, "fig2_reliability_uncertainty")


# =============================================================================
# FIGURE 3 — Recall vs ECE Frontier
# =============================================================================

def make_fig3(table):
    print("\n[Fig 3] Recall vs ECE Frontier")
    # Separate LR and RF into two visual groups via shape
    # LR = filled markers, RF = open markers
    # Color = calibration method
    methods_plot = [
        ("logistic_regression+none+none",                "LR + None",       "#3A6EA5", "o", True),
        ("logistic_regression+smote+none",               "LR + SMOTE",      "#C0392B", "s", True),
        ("logistic_regression+class_weight+none",        "LR + ClassWt",    "#27AE60", "^", True),
        ("logistic_regression+smote+temperature_scaling","LR + SMOTE + TS", "#8E44AD", "D", True),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE+PCDM", "#D4AC0D", "*", True),
        ("random_forest+none+none",                      "RF + None",       "#3A6EA5", "o", False),
        ("random_forest+smote+none",                     "RF + SMOTE",      "#C0392B", "s", False),
        ("random_forest+smote+per_class_adaptive",       "RF + SMOTE+PCDM", "#D4AC0D", "*", False),
    ]
    # Jitter to separate overlapping points
    jitter = {
        "logistic_regression+none+none":                 ( 0.000,  0.000),
        "logistic_regression+smote+none":                ( 0.010,  0.000),
        "logistic_regression+class_weight+none":         (-0.010,  0.000),
        "logistic_regression+smote+temperature_scaling": ( 0.000,  0.010),
        "logistic_regression+smote+per_class_adaptive":  ( 0.000, -0.010),
        "random_forest+none+none":                       ( 0.015,  0.000),
        "random_forest+smote+none":                      (-0.015,  0.000),
        "random_forest+smote+per_class_adaptive":        ( 0.000,  0.015),
    }
    datasets_plot = ["pima", "phoneme", "default_credit_card_clients"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.30,
                        left=0.07, right=0.97)

    for ax, ds in zip(axes, datasets_plot):
        for m, lbl, color, mk, filled in methods_plot:
            rm, rs = gm(table, m, ds, "recall_minority")
            em, es = gm(table, m, ds, "ece_minority")
            if np.isnan(rm) or np.isnan(em):
                continue
            jx, jy = jitter.get(m, (0, 0))
            mfc = color if filled else "white"
            ax.errorbar(rm + jx, em + jy, xerr=rs, yerr=es,
                        fmt=mk, color=color, mfc=mfc, mec=color,
                        ms=6, capsize=2, elinewidth=0.9,
                        label=lbl, zorder=5, alpha=0.95)

        ax.set_title(DS_LABEL[ds], fontsize=10, fontweight="bold")
        ax.set_xlabel("Minority Recall  (↑)", fontsize=9)
        ax.set_ylabel("ECE_minority  (↓)" if ax is axes[0] else "", fontsize=9)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.02, None)
        ax.grid(alpha=0.18)

    # Legend at the bottom center — journal style
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, 0.01),
               frameon=True,
               title="Method  (filled = LR,  open = RF)",
               title_fontsize=8)
    fig.suptitle(
        "Figure 3 — Calibration-Recall Frontier  (error bars = ±std, 5 seeds)",
        fontsize=10, fontweight="bold",
    )
    save(fig, "fig3_recall_ece_frontier")


# =============================================================================
# FIGURE 4 — Severity Sweep
# =============================================================================

def make_fig4():
    print("\n[Fig 4] Severity Sweep")
    mechanisms = [
        ("extreme_imbalance",   "IR Sensitivity",      "#3A6EA5", "o"),
        ("boundary_overlap",    "Boundary Overlap",    "#C0392B", "s"),
        ("confidence_collapse", "Confidence Collapse", "#D4AC0D", "^"),
        ("noisy_minority",      "Label Noise",         "#27AE60", "D"),
    ]
    x_pos = np.arange(3)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.10, right=0.68)

    for mech, label, color, mk in mechanisms:
        ece_vals = [load_ece(f"{mech}_{sev}") for sev in ["mild","moderate","severe"]]
        valid_x  = [x_pos[i] for i, e in enumerate(ece_vals) if not np.isnan(e)]
        valid_y  = [e for e in ece_vals if not np.isnan(e)]
        if not valid_y:
            continue
        ax.plot(valid_x, valid_y, marker=mk, color=color,
                lw=1.6, ms=5, label=label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Mild", "Moderate", "Severe"], fontsize=9)
    ax.set_xlabel("Severity Level", fontsize=9)
    ax.set_ylabel("ECE_minority  (log scale, ↓ better)", fontsize=9)
    ax.set_title(
        "Figure 4 — Calibration Degradation Across Severity Levels\n"
        "(LR + no resampling, seed=42)",
        fontsize=10, fontweight="bold",
    )
    ax.set_yscale("log")                    # log scale fixes visual compression
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.grid(alpha=0.18, which="both")
    ax.set_ylim(bottom=1e-3)

    # Legend to the right — outside the plot
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.03, 1.0), borderaxespad=0,
              frameon=True, framealpha=0.9)
    save(fig, "fig4_severity_sweep")


# =============================================================================
# FIGURE 5 — Confidence Zone Sweep
# =============================================================================

def make_fig5():
    print("\n[Fig 5] Confidence Zone Sweep")
    zones = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Single method — LR + None (zone data only available for this method)
    methods_zone = [
        ("logistic_regression_none_none", "LR + None", "#3A6EA5", "o"),
    ]

    def zone_ece(zone, suffix):
        zone_str = f"{zone:.1f}".replace(".", "p")
        files = sorted(METRICS_CAL.glob(
            f"*confidence_zone_{zone_str}*{suffix}*calibration_metrics.json"))
        if not files:
            return float("nan")
        return float(json.load(open(files[0])).get("ece_minority", float("nan")))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.12, right=0.95)

    for suffix, label, color, mk in methods_zone:
        ece_vals = [zone_ece(z, suffix) for z in zones]
        valid = [(z, e) for z, e in zip(zones, ece_vals) if not np.isnan(e)]
        if not valid:
            continue
        zv, ev = zip(*valid)
        ax.plot(zv, ev, marker=mk, color=color, lw=1.6, ms=5, label=label)

    # Subtle shading for instability region — no text box on the plot
    ax.axvspan(0.25, 0.55, alpha=0.06, color="#C0392B", zorder=0)

    ax.set_xticks(zones)
    ax.set_xticklabels([str(z) for z in zones], fontsize=9)
    ax.set_xlabel("Confidence Zone  (true P(y=1|x) ≈ zone)", fontsize=9)
    ax.set_ylabel("ECE_minority  (↓ better)", fontsize=9)
    ax.set_title(
        "Figure 5 — Calibration Error Across Confidence Zones\n"
        "(shaded region = instability zone,  LR + no resampling)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(alpha=0.18)
    ax.set_ylim(0, None)
    # No legend — single-line plots do not need one
    save(fig, "fig5_confidence_zone")


# =============================================================================
# FIGURE 6 — Synthetic Benchmark Overview
# =============================================================================

def make_fig6():
    print("\n[Fig 6] Synthetic Benchmark Overview")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    benchmarks = [
        ("extreme_imbalance_severe",   "IR Sensitivity (IR=101)",        "#3A6EA5"),
        ("boundary_overlap_severe",    "Boundary Overlap (σ=0.8)",       "#C0392B"),
        ("confidence_collapse_severe", "Confidence Collapse (zone=0.15)","#D4AC0D"),
        ("noisy_minority_severe",      "Label Noise (30% flip)",         "#27AE60"),
        ("feature_corruption_severe",  "Feature Corruption (50% zero)",  "#8E44AD"),
        ("distribution_shift_severe",  "Covariate Shift (Δμ=3.0)",       "#E67E22"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.48, wspace=0.28,
                        top=0.88, bottom=0.10, left=0.06, right=0.97)

    for ax, (ds_name, title, color) in zip(axes.flatten(), benchmarks):
        csv_path = SYNTHETIC_DIR / f"{ds_name}_n5000_seed42.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, "Data not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")
            continue

        df = pd.read_csv(csv_path)
        skip = {"collapse_region","collapse_variance","noise_rate",
                "confidence_zone","split_hint","shift_magnitude","label"}
        feat_cols = [c for c in df.columns if c not in skip]
        if len(feat_cols) < 2:
            feat_cols = [c for c in df.columns if c.startswith("feature_")]

        X = df[feat_cols].values
        y = df["label"].values
        try:
            X_2d = PCA(n_components=2, random_state=42).fit_transform(
                       RobustScaler().fit_transform(X))
        except Exception:
            X_2d = X[:, :2]

        rng  = np.random.default_rng(42)
        idx0 = rng.choice(np.where(y == 0)[0], min(300, (y==0).sum()), replace=False)
        idx1 = rng.choice(np.where(y == 1)[0], min(80,  (y==1).sum()), replace=False)

        ax.scatter(X_2d[idx0, 0], X_2d[idx0, 1],
                   s=6, alpha=0.20, color="#AED6F1",
                   rasterized=True, zorder=2)
        ax.scatter(X_2d[idx1, 0], X_2d[idx1, 1],
                   s=14, alpha=0.80, color=color,
                   edgecolors="white", linewidths=0.3,
                   rasterized=True, zorder=3)

        n_min = int((y == 1).sum())
        ir    = int((y == 0).sum()) / max(n_min, 1)
        ax.set_title(f"{title}\n(n_min={n_min}, IR={ir:.0f})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=8)
        ax.set_ylabel("PC 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.12)

    # Single shared legend at the bottom
    legend_handles = [
        Line2D([0],[0], marker="o", color="w", mfc="#AED6F1",
               ms=6, label="Majority class"),
        Line2D([0],[0], marker="o", color="w", mfc="#888888",
               ms=6, label="Minority class (colour = mechanism)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, 0.02),
               frameon=True, framealpha=0.9,
               markerscale=0.8)
    fig.suptitle(
        "Figure 6 — Calibration Stress Test Suite  "
        "(PCA projection, severe severity, seed=42)",
        fontsize=11, fontweight="bold",
    )
    save(fig, "fig6_synthetic_benchmark")


# =============================================================================
# APPENDIX FIGURE A — Per-Dataset Reliability
# =============================================================================

def make_appendix_a():
    print("\n[Appendix A] Per-Dataset Reliability")
    methods_info = [
        ("logistic_regression+none+none",                "LR + None",        "#3A6EA5"),
        ("logistic_regression+smote+none",               "LR + SMOTE",       "#C0392B"),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE + PCDM","#D4AC0D"),
    ]
    datasets_plot = ["pima", "phoneme", "default_credit_card_clients"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.subplots_adjust(wspace=0.28, top=0.80, bottom=0.14,
                        left=0.07, right=0.72)

    for ax, ds in zip(axes, datasets_plot):
        _draw_reliability(ax, ds, "minority", methods_info)
        ax.set_title(DS_LABEL[ds], fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean predicted confidence", fontsize=9)
        ax.set_ylabel("Fraction of minority positives" if ax is axes[0] else "", fontsize=9)

    # Single legend to the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center left", bbox_to_anchor=(0.73, 0.50),
               fontsize=9, frameon=True, title="Method", title_fontsize=9)
    fig.suptitle(
        "Appendix Figure A — Minority-Class Reliability Diagrams  "
        "(5 seeds, shaded = ±1 std)",
        fontsize=11, fontweight="bold",
    )
    save(fig, "appendix_fig_a_reliability")


# =============================================================================
# APPENDIX FIGURE B — Seed Variance
# =============================================================================

def make_appendix_b(table):
    print("\n[Appendix B] Seed Variance")
    methods_var = [
        ("logistic_regression+none+none",                "LR + None",       "#3A6EA5"),
        ("logistic_regression+smote+none",               "LR + SMOTE",      "#C0392B"),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE+PCDM", "#D4AC0D"),
    ]
    datasets_var = ["pima", "phoneme"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.18,
                        left=0.09, right=0.97)

    for ax, ds in zip(axes, datasets_var):
        x      = np.arange(len(methods_var))
        means  = [gm(table, m, ds, "ece_minority")[0] for m, _, _ in methods_var]
        stds   = [gm(table, m, ds, "ece_minority")[1] for m, _, _ in methods_var]
        colors = [c for _, _, c in methods_var]
        labels = [lbl.replace("LR + ", "") for _, lbl, _ in methods_var]

        ax.bar(x, means, 0.50, color=colors, alpha=0.75,
               edgecolor="white", linewidth=0.8)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    capsize=4, elinewidth=1.2, ecolor="#444")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("ECE_minority" if ax is axes[0] else "", fontsize=9)
        ax.set_title(DS_LABEL[ds], fontsize=10, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Appendix Figure B — ECE_minority Variance Across 5 Seeds  "
        "(error bars = ±1 std)",
        fontsize=11, fontweight="bold",
    )
    save(fig, "appendix_fig_b_seed_variance")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Paper Output Generator — Academic Clean Version")
    print(f"  Output: {OUT.resolve()}")
    print("=" * 60)

    table = load_table()
    if not table:
        print("[ERROR] Paper table not found.")
        print("  Run: python scripts/_build_paper_table.py")
        sys.exit(1)

    make_tables(table)
    make_fig1(table)
    make_fig2()
    make_fig3(table)
    make_fig4()
    make_fig5()
    make_fig6()
    make_appendix_a()
    make_appendix_b(table)

    pdfs = sorted(OUT.glob("*.pdf"))
    pngs = sorted(OUT.glob("*.png"))
    csvs = sorted(OUT.glob("*.csv"))
    print("\n" + "=" * 60)
    print(f"  Done.  {len(pdfs)} PDFs  |  {len(pngs)} PNGs  |  {len(csvs)} CSVs")
    print("=" * 60)
    for f in pdfs:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
