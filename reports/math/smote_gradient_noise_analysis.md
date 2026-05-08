# SMOTE Gradient Noise Analysis

## Mechanism

SMOTE generates synthetic minority samples by:
1. For each minority sample $x_i$, find its $k$ nearest minority neighbors
2. Randomly select one neighbor $x_j$
3. Generate synthetic sample: $x_{\text{syn}} = x_i + \lambda (x_j - x_i)$, where $\lambda \sim U[0,1]$

## Why This Causes Calibration Collapse

### Boundary Concentration

By construction, SMOTE generates samples *between* existing minority samples. When minority samples are near the class boundary (as is common in imbalanced datasets), synthetic samples are also near the boundary.

### Gradient Noise

During training, the model receives gradients from:
- Real majority samples: pushing the boundary away from majority region
- Real minority samples: pushing the boundary toward minority region  
- **Synthetic minority samples**: also pushing toward minority region, but from positions that may not reflect the true data distribution

The synthetic samples create a "gradient noise" effect: the model learns to be confident in regions where the true posterior probability is uncertain.

### Gradient Noise Risk Score

We quantify this risk as the variance of k-NN distances for synthetic samples:

$$\text{GNR} = \text{Var}\left(\{d(x_{\text{syn}}, \text{kNN}(x_{\text{syn}}))\}_{x_{\text{syn}} \in X_{\text{synthetic}}}\right)$$

High variance → synthetic samples are spread across the feature space (some near boundary, some far) → unstable gradient contributions → calibration collapse.

**Implementation:** `src/resampling/smote_resampler.py::_compute_gradient_noise_risk()`

| GNR Score | Risk Level | Interpretation |
|-----------|------------|----------------|
| < 0.1     | Low        | Synthetic samples are tightly clustered — stable gradients |
| 0.1–0.5   | Medium     | Some boundary samples — moderate calibration risk |
| > 0.5     | High       | Widely spread synthetic samples — high calibration collapse risk |
