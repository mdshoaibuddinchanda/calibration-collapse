"""
Build the final paper table from all controlled_validation results.
Reads all multiseed JSON files and produces the cross-dataset summary.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
multiseed_dir = ROOT / "outputs" / "multiseed"

DATASETS = ["pima", "phoneme", "credit_card", "extreme_imbalance_severe"]
KEY_METRICS = ["ece_global", "ece_minority", "recall_minority", "f1_minority", "auc_roc", "brier_minority"]

# Load all controlled_validation results
table = {}  # method -> dataset -> metrics

for f in sorted(multiseed_dir.glob("controlled_validation_*.json")):
    if "summary" in f.name or "paper_table" in f.name:
        continue
    with open(f) as fh:
        data = json.load(fh)

    dataset = data.get("dataset", "unknown")
    model = data.get("model", "unknown")
    resampler = data.get("resampler", "unknown")
    calibrator = data.get("calibrator", "unknown")
    method = f"{model}+{resampler}+{calibrator}"
    failed = data.get("failed_seeds", [])
    n_seeds = len(data.get("seeds", []))
    n_ok = n_seeds - len(failed)

    if method not in table:
        table[method] = {}

    metrics = {}
    for k, v in data.get("metric_stats", {}).items():
        if k in KEY_METRICS:
            metrics[k] = {"mean": v["mean"], "std": v["std"],
                          "ci_lower": v["ci_lower"], "ci_upper": v["ci_upper"],
                          "cv": v["cv"], "is_stable": v["is_stable"],
                          "n_seeds": n_ok}
    table[method][dataset] = metrics

# Print paper table
print("\n" + "="*120)
print("PAPER TABLE: ECE_minority and recall_minority (mean ± std, 95% CI) — 5 seeds")
print("="*120)

# ECE_minority table
print("\n--- ECE_minority (↓ better) ---")
header = f"{'Method':<55}" + "".join(f"{d:<30}" for d in DATASETS)
print(header)
print("-" * (55 + 30*len(DATASETS)))

method_order = [
    "logistic_regression+none+none",
    "logistic_regression+smote+none",
    "logistic_regression+class_weight+none",
    "logistic_regression+smote+temperature_scaling",
    "logistic_regression+smote+per_class_adaptive",
    "random_forest+none+none",
    "random_forest+smote+none",
    "random_forest+class_weight+none",
    "random_forest+smote+temperature_scaling",
    "random_forest+smote+per_class_adaptive",
]

for method in method_order:
    if method not in table:
        continue
    row = f"{method:<55}"
    for ds in DATASETS:
        if ds in table[method] and "ece_minority" in table[method][ds]:
            m = table[method][ds]["ece_minority"]
            n = m.get("n_seeds", "?")
            cell = f"{m['mean']:.4f}±{m['std']:.4f} (n={n})"
            stable = "✓" if m.get("is_stable") else "⚠"
            row += f"{cell+stable:<30}"
        else:
            row += f"{'N/A':<30}"
    print(row)

# recall_minority table
print("\n--- recall_minority (↑ better) ---")
print(header)
print("-" * (55 + 30*len(DATASETS)))

for method in method_order:
    if method not in table:
        continue
    row = f"{method:<55}"
    for ds in DATASETS:
        if ds in table[method] and "recall_minority" in table[method][ds]:
            m = table[method][ds]["recall_minority"]
            n = m.get("n_seeds", "?")
            cell = f"{m['mean']:.4f}±{m['std']:.4f} (n={n})"
            stable = "✓" if m.get("is_stable") else "⚠"
            row += f"{cell+stable:<30}"
        else:
            row += f"{'N/A':<30}"
    print(row)

# AUC-ROC table
print("\n--- AUC-ROC (↑ better) ---")
print(header)
print("-" * (55 + 30*len(DATASETS)))

for method in method_order:
    if method not in table:
        continue
    row = f"{method:<55}"
    for ds in DATASETS:
        if ds in table[method] and "auc_roc" in table[method][ds]:
            m = table[method][ds]["auc_roc"]
            n = m.get("n_seeds", "?")
            cell = f"{m['mean']:.4f}±{m['std']:.4f} (n={n})"
            stable = "✓" if m.get("is_stable") else "⚠"
            row += f"{cell+stable:<30}"
        else:
            row += f"{'N/A':<30}"
    print(row)

# Save consolidated table
out_path = multiseed_dir / "controlled_validation_paper_table.json"
with open(out_path, "w") as fh:
    json.dump(table, fh, indent=2)
print(f"\nFull table saved to: {out_path}")
print(f"Total methods: {len(table)}")
print(f"Total datasets: {len(DATASETS)}")
