"""
Auto-generate individual YAML configs for every synthetic dataset
so the DatasetRegistry can load them like real datasets.
"""
import sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
syn_dir = ROOT / "datasets" / "synthetic"
cfg_dir = ROOT / "configs" / "datasets"

# Load the registry YAML to get metadata
with open(cfg_dir / "synthetic_registry.yaml") as f:
    registry = yaml.safe_load(f)

created = 0
for name, meta in registry.items():
    out_path = cfg_dir / f"{name}.yaml"
    if out_path.exists():
        continue
    cfg = {
        "name": name,
        "path": meta["path"],
        "target_column": "label",
        "positive_class": 1,
        "feature_columns": None,
        "imbalance_ratio": meta.get("imbalance_ratio"),
        "encoding": "utf-8",
        "separator": ",",
        "missing_value_strategy": "median",
        "expected_imbalance_range": [1.0, 200.0],
    }
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    created += 1

print(f"Created {created} synthetic dataset YAML configs in {cfg_dir}")
