"""Check which datasets have controlled_validation multiseed results."""
import sys, json
from pathlib import Path
sys.path.insert(0, '.')

ROOT = Path('.')
multiseed = ROOT / 'outputs' / 'multiseed'
metrics   = ROOT / 'outputs' / 'metrics' / 'calibration'
raw       = ROOT / 'datasets' / 'raw'

# 1. What raw datasets exist?
print("=== RAW DATASETS ===")
for f in sorted(raw.glob("*.csv")):
    if f.name == '.gitkeep':
        continue
    import pandas as pd
    df = pd.read_csv(f)
    print(f"  {f.name}: {len(df)} rows, {df.shape[1]} cols")

# 2. What multiseed results exist?
print("\n=== MULTISEED RESULTS (controlled_validation) ===")
datasets_with_results = set()
for f in sorted(multiseed.glob("controlled_validation_*_multiseed.json")):
    parts = f.name.split('_')
    # Extract dataset name from filename
    d = json.load(open(f))
    ds = d.get('dataset', '?')
    datasets_with_results.add(ds)

for ds in sorted(datasets_with_results):
    files = list(multiseed.glob(f"controlled_validation_{ds}_*_multiseed.json"))
    print(f"  {ds}: {len(files)} method files")

# 3. What datasets have calibration metrics?
print("\n=== DATASETS WITH CALIBRATION METRICS ===")
ds_in_metrics = set()
for f in metrics.glob("*calibration_metrics.json"):
    d = json.load(open(f))
    ds_in_metrics.add(d.get('dataset', '?'))
for ds in sorted(ds_in_metrics):
    count = len(list(metrics.glob(f"*{ds}*calibration_metrics.json")))
    print(f"  {ds}: {count} metric files")

# 4. Paper table coverage
print("\n=== PAPER TABLE COVERAGE ===")
pt = multiseed / 'controlled_validation_paper_table.json'
if pt.exists():
    table = json.load(open(pt))
    datasets_in_table = set()
    for method, ds_data in table.items():
        datasets_in_table.update(ds_data.keys())
    print(f"  Methods in table: {len(table)}")
    print(f"  Datasets in table: {sorted(datasets_in_table)}")
else:
    print("  Paper table NOT FOUND — run: python scripts/_build_paper_table.py")
