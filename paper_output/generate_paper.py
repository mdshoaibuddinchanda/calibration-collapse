
"""
Paper Output Generator v2
==========================
Fixes applied vs v1:
  - Tables exported as CSV (not PDF tables)
  - Figures: larger fonts, no overlapping labels, tight layout
  - Fig4 severity sweep: uses newly generated mild/moderate data
  - Fig5 confidence zone: uses newly generated zone data
  - Fig6 synthetic: uses PCA projection for better scatter visibility
  - All figures: 300 DPI PDF + PNG

Run from project root:
    python paper_output/_gen_v2.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",
    "font.size":           14,
    "axes.titlesize":      16,
    "axes.labelsize":      14,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "legend.fontsize":     12,
    "legend.title_fontsize": 13,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      1.3,
    "grid.alpha":          0.35,
    "lines.linewidth":     2.4,
    "lines.markersize":    9,
})

# ── Palettes ─────────────────────────────────────────────────────────────────
C = {
    "lr_none":   "#4878CF",
    "lr_smote":  "#D65F5F",
    "lr_cw":     "#6ACC65",
    "lr_ts":     "#B47CC7",
    "lr_pcdm":   "#C4AD66",
    "rf_none":   "#77BEDB",
    "rf_smote":  "#F0A58F",
    "rf_pcdm":   "#8EBA42",
}
MK = {"lr_none":"o","lr_smote":"s","lr_cw":"^","lr_ts":"D","lr_pcdm":"*",
      "rf_none":"o","rf_smote":"s","rf_pcdm":"*"}

DS_LABEL = {
    "pima":                     "Pima (IR=1.9)",
    "phoneme":                  "Phoneme (IR=2.4)",
    "credit_card":              "Credit Card (IR=580)",
    "extreme_imbalance_severe": "Extreme Imb. (IR=101)",
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_table() -> dict:
    p = MULTISEED_DIR / "controlled_validation_paper_table.json"
    return json.load(open(p)) if p.exists() else {}

def gm(table, method, ds, metric):
    try:
        d = table[method][ds][metric]
        return float(d["mean"]), float(d["std"])
    except (KeyError, TypeError):
        return float("nan"), float("nan")

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {stem}.pdf / .png")
    plt.close(fig)

def load_ece(dataset_name, method_suffix="logistic_regression_none_none"):
    pat = f"*{dataset_name}*{method_suffix}*calibration_metrics.json"
    files = sorted(METRICS_CAL.glob(pat))
    if not files:
        return float("nan")
    d = json.load(open(files[0]))
    return float(d.get("ece_minority", float("nan")))

def load_multiseed(ds, method_key):
    safe  = method_key.replace("+", "_")
    fname = f"controlled_validation_{ds}_{ds}_{safe}_multiseed.json"
    p = MULTISEED_DIR / fname
    return json.load(open(p)) if p.exists() else {}


# =============================================================================
# TABLES — exported as CSV
# =============================================================================

def make_tables(table):
    print("\n── Tables ──────────────────────────────────────────────────────")

    # Table 1 — Dataset Summary
    df1 = pd.DataFrame([
        ["Pima",               "Real",      768,    8,  1.9,  268,  "Mild imbalance baseline"],
        ["Phoneme",            "Real",      5404,   5,  2.4,  1586, "Moderate imbalance"],
        ["Credit Card",        "Real/Proxy",28480, 29, 580,   49,   "Extreme imbalance"],
        ["Extreme Imb. (syn.)","Synthetic", 5000,  10, 101,   49,   "IR stress (controlled)"],
        ["Conf. Collapse (syn.)","Synthetic",5000, 12,  20,  238,   "Confidence instability"],
    ], columns=["Dataset","Type","Samples","Features","IR","Minority_n","Role"])
    df1.to_csv(OUT / "table1_dataset_summary.csv", index=False)
    print("  Saved: table1_dataset_summary.csv")

    # Table 2 — Main Results (ECE_minority)
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
    datasets_main = ["pima", "phoneme", "credit_card"]
    rows2 = []
    for m in methods:
        row = {"Method": M_LABEL.get(m, m)}
        for ds in datasets_main:
            mn, sd = gm(table, m, ds, "ece_minority")
            row[f"{DS_LABEL[ds]}_ECE_min_mean"] = round(mn, 4) if not np.isnan(mn) else ""
            row[f"{DS_LABEL[ds]}_ECE_min_std"]  = round(sd, 4) if not np.isnan(sd) else ""
        rows2.append(row)
    pd.DataFrame(rows2).to_csv(OUT / "table2_main_results.csv", index=False)
    print("  Saved: table2_main_results.csv")

    # Table 3 — Calibration-Recall Tradeoff
    rows3 = []
    for m in methods[:5]:
        row = {"Method": M_LABEL.get(m, m)}
        for ds in ["pima", "phoneme"]:
            rm, rs = gm(table, m, ds, "recall_minority")
            em, es = gm(table, m, ds, "ece_minority")
            row[f"{ds}_recall_mean"] = round(rm, 4) if not np.isnan(rm) else ""
            row[f"{ds}_recall_std"]  = round(rs, 4) if not np.isnan(rs) else ""
            row[f"{ds}_ECE_min_mean"]= round(em, 4) if not np.isnan(em) else ""
            row[f"{ds}_ECE_min_std"] = round(es, 4) if not np.isnan(es) else ""
        rows3.append(row)
    pd.DataFrame(rows3).to_csv(OUT / "table3_tradeoff.csv", index=False)
    print("  Saved: table3_tradeoff.csv")

    # Table 4 — Multi-Seed Stability
    rows4 = []
    for m in methods:
        for ds in ["pima", "phoneme"]:
            try:
                d = table[m][ds]["ece_minority"]
                rows4.append({
                    "Method":  M_LABEL.get(m, m),
                    "Dataset": DS_LABEL.get(ds, ds),
                    "Mean":    round(d["mean"], 4),
                    "Std":     round(d["std"],  4),
                    "CV":      round(d["cv"],   4) if d["cv"] == d["cv"] else "",
                    "CI_lower":round(d["ci_lower"], 4),
                    "CI_upper":round(d["ci_upper"], 4),
                    "Stable":  d["is_stable"],
                    "N_seeds": d["n_seeds"],
                })
            except (KeyError, TypeError):
                pass
    pd.DataFrame(rows4).to_csv(OUT / "table4_stability.csv", index=False)
    print("  Saved: table4_stability.csv")

    # Table 5 — Ablation
    df5 = pd.read_csv(ABLATION_DIR / "exp001_smote_ratios_ablation_summary.csv") \
          if (ABLATION_DIR / "exp001_smote_ratios_ablation_summary.csv").exists() \
          else pd.DataFrame()
    if not df5.empty:
        df5.to_csv(OUT / "table5_ablation.csv", index=False)
        print("  Saved: table5_ablation.csv")

    # Appendix Table — Skipped configs
    df_app = pd.DataFrame([
        ["SMOTE ratio=0.5", "pima",    "LR/RF", 0.536, 0.5, "Current ratio >= requested"],
        ["SMOTE ratio=2.0", "pima",    "LR/RF", 0.536, 2.0, "Unstable near IR=1.9"],
        ["SMOTE ratio=2.0", "phoneme", "LR/RF", 0.416, 2.0, "Unstable near IR=2.4"],
    ], columns=["Config","Dataset","Model","Current_ratio","Requested_ratio","Reason"])
    df_app.to_csv(OUT / "appendix_table_skipped.csv", index=False)
    print("  Saved: appendix_table_skipped.csv")


# =============================================================================
# FIGURE 1 — Calibration Gap (Global vs Minority ECE)
# =============================================================================

def make_fig1(table):
    print("\n[Fig 1] Calibration Gap")
    methods_plot = [
        ("logistic_regression+none+none",                "LR+None",       "#4878CF"),
        ("logistic_regression+smote+none",               "LR+SMOTE",      "#D65F5F"),
        ("logistic_regression+smote+per_class_adaptive", "LR+SMOTE+PCDM", "#C4AD66"),
        ("random_forest+none+none",                      "RF+None",       "#77BEDB"),
        ("random_forest+smote+none",                     "RF+SMOTE",      "#F0A58F"),
    ]
    datasets_plot = ["pima", "phoneme", "credit_card"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.22)

    for ax, ds in zip(axes, datasets_plot):
        x     = np.arange(len(methods_plot))
        w     = 0.36
        g_m   = [gm(table, m, ds, "ece_global")[0]   for m, _, _ in methods_plot]
        g_s   = [gm(table, m, ds, "ece_global")[1]   for m, _, _ in methods_plot]
        mi_m  = [gm(table, m, ds, "ece_minority")[0] for m, _, _ in methods_plot]
        mi_s  = [gm(table, m, ds, "ece_minority")[1] for m, _, _ in methods_plot]

        ax.bar(x - w/2, g_m,  w, yerr=g_s,  capsize=5,
               color="#5B9BD5", alpha=0.85, label="ECE_global",
               error_kw={"elinewidth":1.8, "ecolor":"#222"})
        ax.bar(x + w/2, mi_m, w, yerr=mi_s, capsize=5,
               color="#ED7D31", alpha=0.85, label="ECE_minority",
               error_kw={"elinewidth":1.8, "ecolor":"#222"})

        ax.set_title(DS_LABEL[ds], fontsize=15, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in methods_plot],
                           rotation=40, ha="right", fontsize=10)
        ax.set_ylabel("ECE  (↓ better)" if ax is axes[0] else "", fontsize=13)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=11, loc="upper right")
    fig.suptitle(
        "Figure 1 — Global ECE vs Minority-Class ECE\n"
        "Global ECE masks severe minority miscalibration  (error bars = ±std, 5 seeds)",
        fontsize=15, fontweight="bold",
    )
    save(fig, "fig1_calibration_gap")


# =============================================================================
# FIGURE 2 — Reliability Diagrams with Uncertainty Bands
# =============================================================================

def _reliability_bands(ax, ds, class_key, methods_info):
    ax.plot([0,1],[0,1],"k--",lw=1.8,label="Perfect",zorder=10,alpha=0.7)
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
                sa.append(float(np.std(av, ddof=1)) if len(av)>1 else 0.0)
                mc.append(float(np.mean(cv)) if cv else b/n_b)
        valid = [(c,a,s) for c,a,s in zip(mc,ma,sa) if c is not None and a is not None]
        if not valid:
            continue
        cv_a, av_a, sv_a = (np.array(v) for v in zip(*valid))
        ax.plot(cv_a, av_a, "o-", color=color, lw=2.4, ms=7, label=label, zorder=6)
        ax.fill_between(cv_a,
                        np.clip(av_a-sv_a,0,1),
                        np.clip(av_a+sv_a,0,1),
                        alpha=0.15, color=color)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(alpha=0.3)

def make_fig2():
    print("\n[Fig 2] Reliability Diagrams with Uncertainty Bands")
    methods_info = [
        ("logistic_regression+none+none",                "LR + None",        "#4878CF"),
        ("logistic_regression+smote+none",               "LR + SMOTE",       "#D65F5F"),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE + PCDM","#C4AD66"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.3, top=0.82, bottom=0.12)

    for ax_i, (ax, class_key, title) in enumerate(zip(
            axes,
            ["global", "minority"],
            ["Global Reliability", "Minority-Class Reliability"])):
        _reliability_bands(ax, "phoneme", class_key, methods_info)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Mean predicted confidence", fontsize=13)
        ax.set_ylabel("Fraction of positives" if ax_i==0 else "", fontsize=13)
        ax.legend(fontsize=11, loc="upper left")

    fig.suptitle(
        "Figure 2 — Reliability Diagrams with Uncertainty Bands  (Phoneme, 5 seeds)\n"
        "Shaded band = ±1 std across seeds",
        fontsize=15, fontweight="bold",
    )
    save(fig, "fig2_reliability_uncertainty")


# =============================================================================
# FIGURE 3 — Recall vs ECE Frontier
# =============================================================================

def make_fig3(table):
    print("\n[Fig 3] Recall vs ECE Frontier")
    methods_plot = [
        ("logistic_regression+none+none",                "LR+None",       "#4878CF","o"),
        ("logistic_regression+smote+none",               "LR+SMOTE",      "#D65F5F","s"),
        ("logistic_regression+class_weight+none",        "LR+ClassWt",    "#6ACC65","^"),
        ("logistic_regression+smote+temperature_scaling","LR+SMOTE+TS",   "#B47CC7","D"),
        ("logistic_regression+smote+per_class_adaptive", "LR+SMOTE+PCDM", "#C4AD66","*"),
        ("random_forest+none+none",                      "RF+None",       "#77BEDB","o"),
        ("random_forest+smote+none",                     "RF+SMOTE",      "#F0A58F","s"),
        ("random_forest+smote+per_class_adaptive",       "RF+SMOTE+PCDM", "#8EBA42","*"),
    ]
    datasets_plot = ["pima", "phoneme", "credit_card"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.35, top=0.82, bottom=0.18)

    for ax, ds in zip(axes, datasets_plot):
        for m, lbl, color, mk in methods_plot:
            rm, rs = gm(table, m, ds, "recall_minority")
            em, es = gm(table, m, ds, "ece_minority")
            if np.isnan(rm) or np.isnan(em):
                continue
            ax.errorbar(rm, em, xerr=rs, yerr=es,
                        fmt=mk, color=color, ms=10,
                        capsize=5, elinewidth=1.8,
                        label=lbl, zorder=5)

        ax.set_title(DS_LABEL[ds], fontsize=15, fontweight="bold")
        ax.set_xlabel("Minority Recall  (↑ better)", fontsize=13)
        ax.set_ylabel("ECE_minority  (↓ better)" if ax is axes[0] else "", fontsize=13)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.01, None)
        ax.grid(alpha=0.3)
        ax.annotate("Ideal\nregion", xy=(0.88, 0.015),
                    fontsize=9, color="#27AE60", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#EAFAF1", alpha=0.7))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.08),
               frameon=True, title="Method", title_fontsize=11)
    fig.suptitle(
        "Figure 3 — Calibration-Recall Frontier\n"
        "Each point = one method  (error bars = ±std, 5 seeds)",
        fontsize=15, fontweight="bold",
    )
    save(fig, "fig3_recall_ece_frontier")


# =============================================================================
# FIGURE 4 — Severity Sweep
# =============================================================================

def make_fig4():
    print("\n[Fig 4] Severity Sweep")
    mechanisms = [
        ("extreme_imbalance",   "IR Sensitivity",      "#4878CF"),
        ("boundary_overlap",    "Boundary Overlap",    "#D65F5F"),
        ("confidence_collapse", "Confidence Collapse", "#C4AD66"),
        ("noisy_minority",      "Label Noise",         "#6ACC65"),
    ]
    severity_levels = ["mild", "moderate", "severe"]
    x_pos = np.arange(3)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.82, bottom=0.12)

    for mech, label, color in mechanisms:
        ece_vals = []
        for sev in severity_levels:
            ds = f"{mech}_{sev}"
            ece = load_ece(ds)
            ece_vals.append(ece)

        valid_x = [x_pos[i] for i, e in enumerate(ece_vals) if not np.isnan(e)]
        valid_y = [e for e in ece_vals if not np.isnan(e)]
        if not valid_y:
            continue

        ax.plot(valid_x, valid_y, "o-", color=color,
                lw=2.4, ms=9, label=label)
        for xp, yp in zip(valid_x, valid_y):
            ax.annotate(f"{yp:.3f}", (xp, yp),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Mild", "Moderate", "Severe"], fontsize=13)
    ax.set_xlabel("Severity Level", fontsize=14)
    ax.set_ylabel("ECE_minority  (↓ better)", fontsize=14)
    ax.set_title(
        "Figure 4 — Calibration Degradation Across Severity Levels\n"
        "(LR + no resampling, seed=42)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)
    save(fig, "fig4_severity_sweep")


# =============================================================================
# FIGURE 5 — Confidence Zone Sweep
# =============================================================================

def make_fig5():
    print("\n[Fig 5] Confidence Zone Sweep")
    zones = [0.1, 0.3, 0.5, 0.7, 0.9]

    methods_zone = [
        ("logistic_regression_none_none",                "LR + None",        "#4878CF", "o"),
        ("logistic_regression_smote_none",               "LR + SMOTE",       "#D65F5F", "s"),
        ("logistic_regression_smote_per_class_adaptive", "LR + SMOTE + PCDM","#C4AD66", "*"),
    ]

    def zone_ece(zone, suffix):
        # Dataset names: confidence_zone_0p1, 0p3, 0p5, 0p7, 0p9
        zone_str = f"{zone:.1f}".replace(".", "p")
        pat = f"*confidence_zone_{zone_str}*{suffix}*calibration_metrics.json"
        files = sorted(METRICS_CAL.glob(pat))
        if not files:
            return float("nan")
        d = json.load(open(files[0]))
        return float(d.get("ece_minority", float("nan")))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.82, bottom=0.12)

    for suffix, label, color, mk in methods_zone:
        ece_vals = [zone_ece(z, suffix) for z in zones]
        valid = [(z, e) for z, e in zip(zones, ece_vals) if not np.isnan(e)]
        if not valid:
            continue
        zv, ev = zip(*valid)
        ax.plot(zv, ev, marker=mk, color=color,
                lw=2.4, ms=9, label=label)
        for zp, ep in zip(zv, ev):
            ax.annotate(f"{ep:.3f}", (zp, ep),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(zones)
    ax.set_xticklabels([str(z) for z in zones], fontsize=13)
    ax.set_xlabel("Confidence Zone  (true posterior P(y=1|x) ≈ zone)", fontsize=14)
    ax.set_ylabel("ECE_minority  (↓ better)", fontsize=14)
    ax.set_title(
        "Figure 5 — Calibration Error Across Confidence Zones\n"
        "Where does calibration instability concentrate?",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    # Shade instability region
    ax.axvspan(0.25, 0.55, alpha=0.07, color="#E74C3C")
    ax.text(0.4, ax.get_ylim()[1] * 0.92,
            "High-instability\nregion",
            fontsize=10, color="#C0392B", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", alpha=0.6))

    save(fig, "fig5_confidence_zone")


# =============================================================================
# FIGURE 6 — Synthetic Benchmark Overview (PCA scatter)
# =============================================================================

def make_fig6():
    print("\n[Fig 6] Synthetic Benchmark Overview")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    benchmarks = [
        ("extreme_imbalance_severe",   "IR Sensitivity\n(IR=101)",        "#4878CF"),
        ("boundary_overlap_severe",    "Boundary Overlap\n(σ=0.8)",       "#D65F5F"),
        ("confidence_collapse_severe", "Confidence Collapse\n(zone=0.15)","#C4AD66"),
        ("noisy_minority_severe",      "Label Noise\n(30% flip)",         "#6ACC65"),
        ("feature_corruption_severe",  "Feature Corruption\n(50% zero)",  "#B47CC7"),
        ("distribution_shift_severe",  "Covariate Shift\n(Δμ=3.0)",       "#F0A58F"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.90, bottom=0.06)

    for ax, (ds_name, title, color) in zip(axes.flatten(), benchmarks):
        csv_path = SYNTHETIC_DIR / f"{ds_name}_n5000_seed42.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, "Data not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        df = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns
                     if c.startswith("feature_") and c not in
                     ("collapse_region","collapse_variance","noise_rate",
                      "confidence_zone","split_hint","shift_magnitude")]

        if len(feat_cols) < 2:
            feat_cols = [c for c in df.columns if c.startswith("feature_")]

        X = df[feat_cols].values
        y = df["label"].values

        # PCA to 2D for clean scatter
        try:
            X_sc  = RobustScaler().fit_transform(X)
            X_2d  = PCA(n_components=2, random_state=42).fit_transform(X_sc)
        except Exception:
            X_2d = X[:, :2]

        rng   = np.random.default_rng(42)
        idx0  = rng.choice(np.where(y==0)[0], min(400, (y==0).sum()), replace=False)
        idx1  = rng.choice(np.where(y==1)[0], min(100, (y==1).sum()), replace=False)

        ax.scatter(X_2d[idx0,0], X_2d[idx0,1],
                   alpha=0.25, s=10, color="#AED6F1", label="Majority", rasterized=True)
        ax.scatter(X_2d[idx1,0], X_2d[idx1,1],
                   alpha=0.80, s=25, color=color, label="Minority",
                   edgecolors="white", linewidths=0.5, rasterized=True)

        n_min = int((y==1).sum())
        n_maj = int((y==0).sum())
        ir    = n_maj / max(n_min, 1)
        ax.set_title(f"{title}\nn_min={n_min}, IR={ir:.0f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.legend(fontsize=9, loc="upper right", markerscale=2)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Figure 6 — Calibration Stress Test Suite  (PCA projection, severe severity, seed=42)",
        fontsize=15, fontweight="bold",
    )
    save(fig, "fig6_synthetic_benchmark")


# =============================================================================
# APPENDIX FIGURE A — Per-Dataset Reliability
# =============================================================================

def make_appendix_a():
    print("\n[Appendix A] Per-Dataset Reliability")
    methods_info = [
        ("logistic_regression+none+none",                "LR + None",        "#4878CF"),
        ("logistic_regression+smote+none",               "LR + SMOTE",       "#D65F5F"),
        ("logistic_regression+smote+per_class_adaptive", "LR + SMOTE + PCDM","#C4AD66"),
    ]
    datasets_plot = ["pima", "phoneme", "credit_card"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.12)

    for ax, ds in zip(axes, datasets_plot):
        _reliability_bands(ax, ds, "minority", methods_info)
        ax.set_title(DS_LABEL[ds], fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean predicted confidence", fontsize=12)
        ax.set_ylabel("Fraction of minority positives" if ax is axes[0] else "", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")

    fig.suptitle(
        "Appendix Figure A — Minority-Class Reliability Diagrams  (5 seeds)\n"
        "Shaded band = ±1 std",
        fontsize=14, fontweight="bold",
    )
    save(fig, "appendix_fig_a_reliability")


# =============================================================================
# APPENDIX FIGURE B — Seed Variance
# =============================================================================

def make_appendix_b(table):
    print("\n[Appendix B] Seed Variance")
    methods_var = [
        ("logistic_regression+none+none",                "LR+None",       "#4878CF"),
        ("logistic_regression+smote+none",               "LR+SMOTE",      "#D65F5F"),
        ("logistic_regression+smote+per_class_adaptive", "LR+SMOTE+PCDM", "#C4AD66"),
    ]
    datasets_var = ["pima", "phoneme"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.15)

    for ax, ds in zip(axes, datasets_var):
        x      = np.arange(len(methods_var))
        means  = [gm(table, m, ds, "ece_minority")[0] for m, _, _ in methods_var]
        stds   = [gm(table, m, ds, "ece_minority")[1] for m, _, _ in methods_var]
        colors = [c for _, _, c in methods_var]
        labels = [lbl.replace("LR + ", "") for _, lbl, _ in methods_var]

        ax.bar(x, means, 0.55, color=colors, alpha=0.85,
               edgecolor="white", linewidth=1.2)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    capsize=7, elinewidth=2.2, ecolor="#333")

        for xi, (mn, sd) in enumerate(zip(means, stds)):
            if not np.isnan(mn):
                ax.text(xi, mn + sd + 0.012, f"σ={sd:.3f}",
                        ha="center", va="bottom", fontsize=10, color="#444")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel("ECE_minority" if ax is axes[0] else "", fontsize=13)
        ax.set_title(DS_LABEL[ds], fontsize=14, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Appendix Figure B — ECE_minority Variance Across 5 Seeds\n"
        "Error bars = ±1 std",
        fontsize=14, fontweight="bold",
    )
    save(fig, "appendix_fig_b_seed_variance")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Paper Output Generator v2")
    print(f"  Output: {OUT.resolve()}")
    print("=" * 60)

    table = load_table()
    if not table:
        print("[ERROR] Paper table not found. Run: python scripts/_build_paper_table.py")
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
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
