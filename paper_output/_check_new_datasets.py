"""Check the 7 new datasets — configs, imbalance ratios, and training status."""
import sys, json
import pandas as pd
from pathlib import Path
sys.path.insert(0, '.')
from src.data.registry import DatasetRegistry

ROOT = Path('.')
registry = DatasetRegistry(ROOT / 'configs' / 'datasets')
multiseed = ROOT / 'outputs' / 'multiseed'

new_datasets = [
    'bank_marketing', 'default_credit_card_clients', 'dry_bean',
    'give_me_some_credit', 'mammography', 'nsl_kdd', 'thyroid_disease'
]

print("=== NEW DATASET STATUS ===\n")
print(f"{'Dataset':<35} {'Registered':<12} {'Rows':<10} {'IR':<8} {'Min_n':<8} {'Multiseed'}")
print("-" * 90)

for ds in new_datasets:
    # Check registry
    try:
        cfg = registry.get(ds)
        registered = "YES"
        # Load and check
        try:
            df = pd.read_csv(ROOT / cfg.path)
            vc = df[cfg.target_column].value_counts()
            ir = vc.max() / vc.min()
            min_n = vc.min()
            rows = len(df)
        except Exception as e:
            rows, ir, min_n = "ERR", "ERR", "ERR"
    except KeyError:
        registered = "NO"
        rows, ir, min_n = "-", "-", "-"

    # Check multiseed results
    ms_files = list(multiseed.glob(f"controlled_validation_{ds}_*_multiseed.json"))
    ms_status = f"{len(ms_files)} files" if ms_files else "NOT RUN"

    print(f"  {ds:<33} {registered:<12} {str(rows):<10} {str(round(float(ir),1) if ir not in ('ERR','-') else ir):<8} {str(min_n):<8} {ms_status}")

print("\n=== FAILED RUNS (SMOTE ratio=2.0 — expected, not bugs) ===")
from src.experiment.tracker import ExperimentTracker
tracker = ExperimentTracker(ROOT / 'outputs' / 'experiments.db')
failed = tracker.query("SELECT run_id, dataset, model, resampler, calibrator FROM experiments WHERE status='failed' ORDER BY run_id")
print(f"  Total failed: {len(failed)}")
if len(failed) > 0:
    print("  All failed runs:")
    for _, row in failed.iterrows():
        print(f"    {row['run_id'][:70]}")

print("\n=== SKIPPED RUNS ===")
skipped = tracker.query("SELECT run_id FROM experiments WHERE status='skipped'")
print(f"  Total skipped: {len(skipped)}")
