"""
Generate the full Calibration Stress Test Suite.

Produces:
  - 3 severity levels × 6 modes = 18 severity-parameterized datasets
  - 5 confidence zone datasets (zones: 0.1, 0.3, 0.5, 0.7, 0.9)
  Total: 23 controlled benchmark datasets

Each dataset targets a specific calibration failure mechanism.
All outputs are deterministic (SHA256 verified).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import (
    SyntheticDataGenerator, SyntheticConfig, GenerationMode, CONFIDENCE_ZONES
)

ROOT = Path(__file__).parent.parent
out_dir = ROOT / "datasets" / "synthetic"
gen = SyntheticDataGenerator(output_dir=out_dir)

print("Generating Calibration Stress Test Suite...")
print("=" * 60)

# -----------------------------------------------------------------------
# Part 1: Severity sweeps — 3 levels × 6 modes = 18 datasets
# -----------------------------------------------------------------------
print("\n[1/2] Severity sweeps (mild / moderate / severe):")
severity_modes = [
    GenerationMode.EXTREME_IMBALANCE,
    GenerationMode.NOISY_MINORITY,
    GenerationMode.BOUNDARY_OVERLAP,
    GenerationMode.CONFIDENCE_COLLAPSE,
    GenerationMode.FEATURE_CORRUPTION,
    GenerationMode.DISTRIBUTION_SHIFT,
]

for mode in severity_modes:
    paths = gen.generate_severity_sweep(mode, n_samples=5000, seed=42)
    for p in paths:
        import pandas as pd
        df = pd.read_csv(p)
        vc = df["label"].value_counts()
        ir = vc.max() / vc.min()
        print(f"  {p.name}: rows={len(df)}, IR={ir:.1f}, minority_n={vc.min()}")

# -----------------------------------------------------------------------
# Part 2: Confidence zone sweep — 5 zones
# -----------------------------------------------------------------------
print("\n[2/2] Confidence zone sweep (zones: 0.1, 0.3, 0.5, 0.7, 0.9):")
zone_paths = gen.generate_confidence_zone_sweep(
    imbalance_ratio=20.0, n_samples=5000, seed=42
)
for p in zone_paths:
    import pandas as pd
    df = pd.read_csv(p)
    vc = df["label"].value_counts()
    ir = vc.max() / vc.min()
    zone = df["confidence_zone"].iloc[0]
    true_prob_mean = df["true_prob"].mean()
    print(f"  {p.name}: rows={len(df)}, IR={ir:.1f}, zone={zone:.1f}, mean_true_prob={true_prob_mean:.3f}")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
import os
all_csvs = sorted(f for f in os.listdir(out_dir) if f.endswith(".csv"))
print(f"\n{'='*60}")
print(f"Total synthetic datasets: {len(all_csvs)}")
print(f"Output directory: {out_dir}")
print("Done.")
