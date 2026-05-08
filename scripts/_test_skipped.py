"""Quick test: verify SMOTE feasibility check raises InvalidSamplingStrategyError."""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.resampling.smote_resampler import SMOTEResampler, InvalidSamplingStrategyError

# pima-like: IR=1.9, minority already > 50% of majority
rng = np.random.default_rng(42)
n_maj, n_min = 500, 268
X = rng.normal(0, 1, (n_maj + n_min, 8))
y = np.array([0]*n_maj + [1]*n_min)

# sampling_strategy=0.5 means "make minority 50% of majority" = 250 samples
# But minority already has 268 > 250 — infeasible
smote = SMOTEResampler(sampling_strategy=0.5, seed=42)
try:
    smote.fit_resample(X, y)
    print("ERROR: Should have raised InvalidSamplingStrategyError")
    sys.exit(1)
except InvalidSamplingStrategyError as e:
    print(f"PASS: Correctly raised InvalidSamplingStrategyError")
    print(f"  Message: {str(e)[:100]}")

# sampling_strategy=1.0 should work fine (balance classes)
smote2 = SMOTEResampler(sampling_strategy=1.0, seed=42)
X_res, y_res = smote2.fit_resample(X, y)
print(f"PASS: sampling_strategy=1.0 works. Resampled: {len(X_res)} samples")
print("All SKIPPED logic tests passed.")
