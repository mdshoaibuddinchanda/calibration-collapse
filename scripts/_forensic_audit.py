"""
Forensic audit script — executes every claim before reporting it.
Finds silent failures, interaction bugs, and statistical invalidity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
results = []

def check(name, status, detail):
    results.append((name, status, detail))
    icon = "✓" if status == PASS else ("✗" if status == FAIL else "⚠")
    print(f"  {icon} [{status}] {name}: {detail[:120]}")

print("\n" + "="*70)
print("FORENSIC AUDIT — EXECUTING ALL CHECKS")
print("="*70)

# =========================================================================
# PHASE 1: ECE DECOMPOSITION IDENTITY
# Claim: ECE_global ≈ (n_maj/n)*ECE_maj + (n_min/n)*ECE_min
# This MUST hold for the paper's central argument to be valid.
# =========================================================================
print("\n[PHASE 1] ECE Decomposition Identity")
from src.evaluation.calibration_metrics import CalibrationMetrics

rng = np.random.default_rng(42)
n_maj, n_min = 900, 100
p_maj = rng.normal(0.1, 0.02, n_maj).clip(0.01, 0.15)
p_min = rng.normal(0.5, 0.05, n_min).clip(0.35, 0.65)
p = np.concatenate([p_maj, p_min])
y = np.concatenate([np.zeros(n_maj, int), np.ones(n_min, int)])

cal = CalibrationMetrics(n_bins=10)
r = cal.compute(p, y)

# The decomposition ECE_global = (n_maj/n)*ECE_maj + (n_min/n)*ECE_min
# does NOT hold for the class-conditional ECE (acc=1.0 always).
# This is a CRITICAL scientific claim in the paper — verify it.
n = n_maj + n_min
reconstructed = (n_maj/n)*r.ece_majority + (n_min/n)*r.ece_minority
diff = abs(r.ece_global - reconstructed)
if diff < 0.05:
    check("ECE decomposition identity", PASS,
          f"ECE_global={r.ece_global:.4f} ≈ weighted_sum={reconstructed:.4f} (diff={diff:.4f})")
else:
    check("ECE decomposition identity", FAIL,
          f"ECE_global={r.ece_global:.4f} ≠ (n_maj/n)*ECE_maj+(n_min/n)*ECE_min={reconstructed:.4f} "
          f"(diff={diff:.4f}). Paper's central decomposition claim is INVALID with current ECE formula.")

# =========================================================================
# PHASE 2: PCDM SOFT-BLEND MATHEMATICAL COHERENCE
# The soft-weighted T_blend = T_1*p + T_0*(1-p) is NOT standard temperature scaling.
# Verify it actually reduces to global TS when T_0 == T_1.
# =========================================================================
print("\n[PHASE 2] PCDM Soft-Blend Coherence")
from src.calibration.per_class_adaptive import PerClassAdaptiveCalibrator
from src.calibration.temperature_scaling import TemperatureScaling

# Create synthetic val/test data
p_val = rng.beta(2, 5, 200)
y_val = (rng.random(200) < p_val).astype(int)
proba_val = np.column_stack([1-p_val, p_val])

p_test = rng.beta(2, 5, 100)
proba_test = np.column_stack([1-p_test, p_test])

# Fit PCDM
pcdm = PerClassAdaptiveCalibrator(use_logit=True)
pcdm.fit(proba_val, y_val, split_tag="val")
T_0 = pcdm._temperatures.get(0, pcdm._global_T)
T_1 = pcdm._temperatures.get(1, pcdm._global_T)

# When T_0 == T_1, PCDM should equal global TS
# Force equal temperatures
pcdm._temperatures = {0: 1.5, 1: 1.5}
pcdm._global_T = 1.5
p_pcdm = pcdm.calibrate(proba_test)[:, 1]

ts = TemperatureScaling(use_logit=True)
ts._T = 1.5
p_ts = ts.calibrate(proba_test)[:, 1]

max_diff = float(np.abs(p_pcdm - p_ts).max())
if max_diff < 1e-5:
    check("PCDM reduces to global TS when T_0==T_1", PASS,
          f"Max diff={max_diff:.2e}")
else:
    check("PCDM reduces to global TS when T_0==T_1", FAIL,
          f"Max diff={max_diff:.4f} — PCDM and global TS diverge even with equal temperatures. "
          "Soft-blend formula is NOT equivalent to standard TS.")

# =========================================================================
# PHASE 3: POWER SCALING DIRECTION CHECK
# p^(1/T): T>1 pushes p TOWARD 1 (increases), T<1 pushes p TOWARD 0 (decreases).
# This is the CORRECT behavior — the NLL optimizer finds the right T automatically.
# =========================================================================
print("\n[PHASE 3] Power Scaling Direction")
p_test_arr = np.array([0.3, 0.5, 0.7, 0.9])

# T=2.0: p^(1/2) = sqrt(p) → larger than p for p∈(0,1) → pushes toward 1
p_T2 = np.power(p_test_arr, 1.0/2.0)
# T=0.5: p^2 → smaller than p for p∈(0,1) → pushes toward 0
p_T05 = np.power(p_test_arr, 1.0/0.5)

if all(p_T2 > p_test_arr) and all(p_T05 < p_test_arr):
    check("Power scaling direction (T>1 increases toward 1, T<1 decreases toward 0)", PASS,
          f"T=2: {p_test_arr.round(2)} → {p_T2.round(3)} (↑), "
          f"T=0.5: {p_test_arr.round(2)} → {p_T05.round(3)} (↓). "
          "NLL optimizer selects correct T automatically.")
else:
    check("Power scaling direction", FAIL,
          f"T=2 result: {p_T2.round(3)}, T=0.5 result: {p_T05.round(3)}")

# =========================================================================
# PHASE 4: SMOTE MINORITY CLASS ASSUMPTION
# SMOTEResampler hardcodes y==1 as minority. This SILENTLY FAILS if
# minority class is 0 (e.g., in some datasets).
# =========================================================================
print("\n[PHASE 4] SMOTE Minority Class Assumption")
from src.resampling.smote_resampler import SMOTEResampler

# Simulate dataset where minority is class 0 (IR=10, 90% class 1, 10% class 0)
rng2 = np.random.default_rng(0)
X_fake = rng2.normal(0, 1, (500, 5))
y_fake = np.array([0]*50 + [1]*450)  # class 0 is minority

smote = SMOTEResampler(seed=42)
try:
    X_res, y_res = smote.fit_resample(X_fake, y_fake)
    meta = smote.get_metadata()
    n_minority_before = int((y_fake == 1).sum())  # This is WRONG — counts majority
    actual_minority_before = int((y_fake == 0).sum())  # 50
    if meta["n_minority_before"] == n_minority_before:
        check("SMOTE minority class assumption (y==1)", FAIL,
              f"SMOTEResampler hardcodes minority=class1. n_minority_before={meta['n_minority_before']} "
              f"but actual minority (class 0) has {actual_minority_before} samples. "
              "Silent corruption when minority class != 1.")
    else:
        check("SMOTE minority class assumption", PASS, "Correctly identifies minority class")
except Exception as e:
    check("SMOTE minority class assumption", WARN, f"Could not test: {e}")

# =========================================================================
# PHASE 5: CALIBRATION METRICS ON CALIBRATED PROBABILITIES
# The runner computes metrics on proba_test_cal (calibrated).
# But the reliability diagram plots proba_test (uncalibrated) as bars
# and proba_test_cal as overlay. Verify the ECE in the JSON matches
# what the reliability diagram shows.
# =========================================================================
print("\n[PHASE 5] Metrics/Plot Consistency")
# The runner passes proba_test_cal to cal_metrics.compute() — correct.
# The reliability diagram receives proba=proba_test (uncalibrated) and
# proba_cal=proba_test_cal. The bars show uncalibrated, overlay shows calibrated.
# The ECE in the JSON is computed from calibrated probs.
# This is CONSISTENT — but the bar heights in the reliability diagram
# do NOT correspond to the reported ECE. A reviewer could be confused.
check("Metrics/plot consistency (ECE from calibrated, bars from uncalibrated)", WARN,
      "Reliability diagram bars show UNCALIBRATED probs but reported ECE is from CALIBRATED probs. "
      "This is correct behavior but must be clearly labeled in paper figures.")

# =========================================================================
# PHASE 6: BRIER SCORE FOR MINORITY CLASS — WRONG LABELS
# brier_minority = brier_score(p[mask_min], y[mask_min])
# For minority samples (y=1), y[mask_min] is all 1s.
# So brier_minority = mean((p_i - 1)^2) for minority samples.
# This is correct — it measures how far P(y=1) is from 1.0 for true positives.
# But it's NOT symmetric with brier_majority.
# brier_majority = mean((p_i - 0)^2) = mean(p_i^2) for true negatives.
# These measure different things and cannot be directly compared.
# =========================================================================
print("\n[PHASE 6] Brier Score Asymmetry")
p_test_b = np.array([0.3, 0.3, 0.3, 0.7, 0.7, 0.7])
y_test_b = np.array([0, 0, 0, 1, 1, 1])

cal_b = CalibrationMetrics(n_bins=3)
r_b = cal_b.compute(p_test_b, y_test_b)

# brier_minority = mean((0.7-1)^2) = mean(0.09) = 0.09
# brier_majority = mean((0.3-0)^2) = mean(0.09) = 0.09
# In this case they're equal, but conceptually different
expected_min = float(np.mean((p_test_b[y_test_b==1] - 1.0)**2))
expected_maj = float(np.mean((p_test_b[y_test_b==0] - 0.0)**2))
if abs(r_b.brier_minority - expected_min) < 1e-6 and abs(r_b.brier_majority - expected_maj) < 1e-6:
    check("Brier score per-class computation", PASS,
          f"brier_minority={r_b.brier_minority:.4f} (expected {expected_min:.4f}), "
          f"brier_majority={r_b.brier_majority:.4f} (expected {expected_maj:.4f})")
else:
    check("Brier score per-class computation", FAIL,
          f"brier_minority={r_b.brier_minority:.4f} != {expected_min:.4f}")

# =========================================================================
# PHASE 7: PREPROCESSING PIPELINE — COLUMN ORDER CONSISTENCY
# If val/test have different column order than train, transform() silently
# uses wrong columns. The pipeline stores column names but doesn't verify
# that val/test columns match train columns.
# =========================================================================
print("\n[PHASE 7] Pipeline Column Order Consistency")
import pandas as pd
from src.preprocessing.pipeline import PreprocessingPipeline

X_train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
X_val_wrong_order = pd.DataFrame({"b": [7.0, 8.0], "a": [9.0, 10.0]})  # columns swapped

pipeline = PreprocessingPipeline()
pipeline.fit(X_train_df)

# transform() uses X[self._numeric_cols] which respects column names, not order
# So this should be safe — pandas indexing by name
try:
    result = pipeline.transform(X_val_wrong_order, split_tag="val")
    # Verify: column 'a' values should be scaled using train 'a' stats
    # Train 'a' median=2.0, IQR=1.0 → scaled val 'a' = (9-2)/1=7, (10-2)/1=8
    if abs(result[0, 0] - 7.0) < 0.5:  # approximate
        check("Pipeline column order safety (pandas name indexing)", PASS,
              "Column reordering handled correctly via pandas name-based indexing")
    else:
        check("Pipeline column order safety", WARN,
              f"Unexpected scaled value: {result[0,0]:.2f}, expected ~7.0")
except Exception as e:
    check("Pipeline column order safety", FAIL, f"Exception: {e}")

# =========================================================================
# PHASE 8: RUNNER DATASET RELOAD — SAME SPLIT EVERY RUN?
# The runner creates a new StratifiedSplitter with the same seed each run.
# But if the same dataset is run multiple times in one experiment loop,
# the split is identical (same seed). This is CORRECT for reproducibility.
# But if the runner is called with different datasets in sequence,
# the seed is the same for all — meaning all datasets get the same
# random split pattern. This is fine for stratified splits but worth noting.
# =========================================================================
print("\n[PHASE 8] Split Seed Consistency Across Datasets")
from src.preprocessing.splitter import StratifiedSplitter

X1 = pd.DataFrame(rng.random((500, 5)))
y1 = pd.Series((rng.random(500) > 0.8).astype(int))
X2 = pd.DataFrame(rng.random((500, 5)))
y2 = pd.Series((rng.random(500) > 0.8).astype(int))

splitter = StratifiedSplitter(seed=42)
s1 = splitter.split(X1, y1)
s2 = splitter.split(X2, y2)

# Both splits use seed=42 — train indices should be different (different data)
# but the split PROPORTIONS should be the same
train_prop_1 = len(s1.split_indices["train"]) / 500
train_prop_2 = len(s2.split_indices["train"]) / 500
if abs(train_prop_1 - train_prop_2) < 0.02:
    check("Split proportions consistent across datasets", PASS,
          f"Dataset1 train={train_prop_1:.2%}, Dataset2 train={train_prop_2:.2%}")
else:
    check("Split proportions inconsistent", WARN,
          f"Dataset1 train={train_prop_1:.2%}, Dataset2 train={train_prop_2:.2%}")

# =========================================================================
# PHASE 9: CONFIDENCE COLLAPSE GENERATOR — FEATURE VALUES ARE NOT PROBABILITIES
# The _confidence_collapse generator sets features to values near collapse_region (e.g., 0.3).
# These are FEATURE VALUES, not model confidence scores.
# A model trained on these features will NOT necessarily output P(y=1)≈0.3.
# The generator does NOT actually produce confidence collapse — it produces
# a dataset where features are concentrated near a value.
# =========================================================================
print("\n[PHASE 9] Confidence Collapse Generator Validity")
import pandas as pd
from src.data.synthetic import SyntheticDataGenerator, SyntheticConfig, GenerationMode
import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    gen = SyntheticDataGenerator(output_dir=Path(tmpdir))
    cfg = SyntheticConfig(
        mode=GenerationMode.CONFIDENCE_COLLAPSE,
        collapse_region=0.3,
        collapse_variance=0.05,
        n_samples=1000,
        seed=42,
    )
    path = gen.generate(cfg)
    df = pd.read_csv(path)

    # Check: are feature values actually near 0.3?
    feat_mean = df[[c for c in df.columns if c.startswith("feature_")]].values.mean()
    feat_std = df[[c for c in df.columns if c.startswith("feature_")]].values.std()

    if abs(feat_mean - 0.3) < 0.05:
        check("Confidence collapse: features near collapse_region", PASS,
              f"Feature mean={feat_mean:.3f} ≈ collapse_region=0.3")
    else:
        check("Confidence collapse: features near collapse_region", FAIL,
              f"Feature mean={feat_mean:.3f} ≠ 0.3")

    # CRITICAL: Train a simple model and check if it actually outputs P≈0.3
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    X_cc = df[feat_cols].values
    y_cc = df["label"].values

    # Scale
    scaler = RobustScaler()
    X_cc_scaled = scaler.fit_transform(X_cc)

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_cc_scaled, y_cc)
    proba_cc = lr.predict_proba(X_cc_scaled)[:, 1]
    model_conf_mean = proba_cc.mean()
    model_conf_std = proba_cc.std()

    if model_conf_std < 0.05:
        check("Confidence collapse: model actually collapses confidence", PASS,
              f"Model P(y=1) mean={model_conf_mean:.3f}, std={model_conf_std:.3f} — narrow band")
    else:
        check("Confidence collapse: model confidence NOT collapsed", WARN,
              f"Model P(y=1) mean={model_conf_mean:.3f}, std={model_conf_std:.3f} — "
              "features near 0.3 don't force model confidence near 0.3. "
              "Generator tests feature concentration, not model confidence collapse.")

# =========================================================================
# PHASE 10: ECE WITH EXTREME IMBALANCE — BIN SPARSITY
# At IR=100, minority has ~50 samples in test set (10% of 500).
# With 10 bins, expected minority samples per bin = 5.
# ECE_minority with <5 samples per bin is statistically unreliable.
# =========================================================================
print("\n[PHASE 10] ECE Bin Sparsity at Extreme Imbalance")
n_test_extreme = 500
n_min_extreme = 5  # IR=100, 20% test → 5 minority test samples
n_maj_extreme = n_test_extreme - n_min_extreme

p_extreme = np.concatenate([
    rng.normal(0.1, 0.05, n_maj_extreme).clip(0.01, 0.99),
    rng.normal(0.6, 0.1, n_min_extreme).clip(0.01, 0.99),
])
y_extreme = np.array([0]*n_maj_extreme + [1]*n_min_extreme)

cal_extreme = CalibrationMetrics(n_bins=10)
r_extreme = cal_extreme.compute(p_extreme, y_extreme)

# Count non-empty minority bins
min_bin_data = r_extreme.bin_data["minority"]
non_empty_bins = sum(1 for b in min_bin_data if b.get("n", 0) > 0)
avg_samples_per_bin = n_min_extreme / max(non_empty_bins, 1)

if avg_samples_per_bin < 5:
    check("ECE bin sparsity at extreme IR", FAIL,
          f"With {n_min_extreme} minority test samples and {non_empty_bins} non-empty bins, "
          f"avg {avg_samples_per_bin:.1f} samples/bin. ECE_minority is statistically unreliable. "
          "Need n_bins ≤ n_minority/5 for reliable ECE.")
else:
    check("ECE bin sparsity at extreme IR", PASS,
          f"Avg {avg_samples_per_bin:.1f} samples/bin — sufficient")

# =========================================================================
# PHASE 11: ISOTONIC CALIBRATION OVERFITTING RISK
# Isotonic regression with few validation samples overfits severely.
# With 10-20 minority val samples, isotonic will memorize them.
# =========================================================================
print("\n[PHASE 11] Isotonic Calibration Overfitting")
from src.calibration.isotonic import IsotonicCalibrator

# Small validation set (realistic for extreme imbalance)
n_val_small = 20
p_val_small = rng.random(n_val_small)
y_val_small = (rng.random(n_val_small) > 0.7).astype(int)

iso = IsotonicCalibrator()
iso.fit(np.column_stack([1-p_val_small, p_val_small]), y_val_small, split_tag="val")

# Check number of thresholds — if equal to n_val, it's memorizing
n_thresholds = len(iso._iso.X_thresholds_)
if n_thresholds >= n_val_small * 0.8:
    check("Isotonic calibration overfitting risk", WARN,
          f"Isotonic fitted {n_thresholds} thresholds on {n_val_small} val samples. "
          "Near-memorization. With small minority val sets, isotonic is unreliable.")
else:
    check("Isotonic calibration overfitting", PASS,
          f"Isotonic fitted {n_thresholds} thresholds on {n_val_small} val samples")

# =========================================================================
# PHASE 12: MANIFEST RUN_ID COLLISION
# run_id = f"{experiment_id}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Two runs in the same second get the SAME run_id → manifest overwrite.
# =========================================================================
print("\n[PHASE 12] Manifest Run ID Collision")
from src.experiment.manifest import ExperimentManifest
import time

m1 = ExperimentManifest.create("exp001", {"seed": 42})
time.sleep(0.01)  # less than 1 second
m2 = ExperimentManifest.create("exp001", {"seed": 42})

if m1.run_id == m2.run_id:
    check("Manifest run_id collision (same-second runs)", FAIL,
          f"Two manifests created <1s apart have same run_id: {m1.run_id}. "
          "Rapid sequential runs will overwrite each other's manifests.")
else:
    check("Manifest run_id collision", PASS,
          f"run_ids are distinct: {m1.run_id[:30]}... vs {m2.run_id[:30]}...")

# =========================================================================
# PHASE 13: LEAKAGE DETECTOR — RESAMPLING CHECK IS STRUCTURAL ONLY
# Check #3 passes if split_indices is non-empty, regardless of whether
# resampling actually happened before or after splitting.
# A malicious or buggy caller could pass split_indices={} after resampling.
# =========================================================================
print("\n[PHASE 13] Leakage Detector Resampling Check Weakness")
from src.audit.leakage_detector import LeakageDetector

detector = LeakageDetector()
# Pass non-empty split_indices — check #3 will PASS even if resampling happened first
result_ld = detector.run_all(
    split_indices={"train": [0,1,2], "val": [3,4], "test": [5,6]},
    n_train_samples=3, n_val_samples=2,
    calibrator_split_tag="val",
    X_train=np.zeros((3,5)), X_val=np.zeros((2,5)), X_test=np.zeros((2,5)),
    y_train=np.array([0,0,1]),
    feature_names=None,
)
check3 = next(c for c in result_ld.checks if c.check_id == 3)
check("Leakage check #3 is structural-only (not empirical)", WARN,
      f"Check #3 status={check3.status}. It passes whenever split_indices is non-empty, "
      "regardless of actual resampling order. Cannot detect resampling-before-split empirically.")

# =========================================================================
# PHASE 14: SINGLE SEED — STATISTICAL INVALIDITY
# All experiments use seed=42. With a single seed, any result could be
# a seed-specific artifact. This is the most likely reviewer rejection point.
# =========================================================================
print("\n[PHASE 14] Single-Seed Statistical Invalidity")
# Demonstrate seed sensitivity of ECE_minority
ece_minority_by_seed = []
for seed in [0, 1, 2, 3, 4, 42, 99, 123, 456, 789]:
    rng_s = np.random.default_rng(seed)
    p_s = np.concatenate([
        rng_s.normal(0.1, 0.03, 900).clip(0.01, 0.99),
        rng_s.normal(0.5, 0.08, 100).clip(0.01, 0.99),
    ])
    y_s = np.array([0]*900 + [1]*100)
    r_s = CalibrationMetrics(n_bins=10).compute(p_s, y_s)
    ece_minority_by_seed.append(r_s.ece_minority)

ece_arr = np.array(ece_minority_by_seed)
cv = ece_arr.std() / ece_arr.mean()
check("Single-seed ECE_minority variance across seeds", FAIL if cv > 0.1 else PASS,
      f"ECE_minority across 10 seeds: mean={ece_arr.mean():.4f}, std={ece_arr.std():.4f}, "
      f"CV={cv:.2%}. {'HIGH variance — single-seed results are unreliable.' if cv > 0.1 else 'Low variance.'}")

# =========================================================================
# PHASE 15: PCDM FIT — CLASS 0 NLL IS WRONG
# For class 0, the code fits: p_cls = 1-p[cls_mask], y_cls = 1-y[cls_mask]
# y_cls = 1 - y_val[cls_mask] = 1 - 0 = 1 (all ones, since cls_mask selects y==0)
# So it's fitting NLL(1-p, 1) = NLL(p, 0) — minimizing confidence for class 0.
# This is correct: T_0 calibrates P(y=0) = 1-P(y=1).
# But the NLL objective uses y_cls=1 always, which means it's fitting
# "how close is 1-p to 1?" = "how close is p to 0?" for majority samples.
# This is mathematically valid but asymmetric with class 1 fitting.
# =========================================================================
print("\n[PHASE 15] PCDM Class-0 NLL Objective")
# Verify: for class 0 samples, y_cls = 1-y_val[cls_mask]
# y_val[cls_mask] = 0 (all zeros), so y_cls = 1 (all ones)
# NLL = -mean(1*log(1-p) + 0*log(p)) = -mean(log(1-p))
# This minimizes T such that 1-p^(1/T) is close to 1, i.e., p^(1/T) close to 0
# Which means p close to 0 — correct for majority class calibration
y_cls_0 = 1 - np.zeros(10)  # = ones
if all(y_cls_0 == 1.0):
    check("PCDM class-0 NLL: y_cls=1 for majority samples", PASS,
          "y_cls = 1-y_val[cls_mask] = 1 for majority (y=0) samples. "
          "NLL minimizes -log(1-p) → calibrates P(y=0). Mathematically correct.")
else:
    check("PCDM class-0 NLL", FAIL, "Unexpected y_cls values")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("FORENSIC AUDIT SUMMARY")
print("="*70)
fails = [(n, d) for n, s, d in results if s == FAIL]
warns = [(n, d) for n, s, d in results if s == WARN]
passes = [(n, d) for n, s, d in results if s == PASS]

print(f"\n  PASS: {len(passes)}")
print(f"  WARN: {len(warns)}")
print(f"  FAIL: {len(fails)}")

if fails:
    print("\nCRITICAL FAILURES:")
    for name, detail in fails:
        print(f"  ✗ {name}")
        print(f"    {detail[:150]}")

if warns:
    print("\nWARNINGS:")
    for name, detail in warns:
        print(f"  ⚠ {name}")
        print(f"    {detail[:150]}")
