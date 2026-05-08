# ECE Formulation and Per-Class Derivation

## 1. Global Expected Calibration Error (ECE)

### Definition

Given a classifier producing probability estimates $\hat{p}_i = P(Y=1 \mid X=x_i)$, we partition the unit interval $[0,1]$ into $B$ equal-width bins $\{B_b\}_{b=1}^{B}$.

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

where:
- $|B_b|$ = number of samples in bin $b$
- $n$ = total number of test samples
- $\text{acc}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \mathbf{1}[y_i = 1]$ — fraction of positives in bin
- $\text{conf}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \hat{p}_i$ — mean predicted confidence in bin

### Implementation Notes

- Bins are equal-width: $[0, 1/B), [1/B, 2/B), \ldots, [(B-1)/B, 1]$
- The last bin includes the right endpoint (samples with $\hat{p}_i = 1.0$)
- Empty bins contribute 0 to the sum
- Implementation: `src/evaluation/calibration_metrics.py::_ece_equal_width()`

---

## 2. Per-Class ECE — The Primary Research Metric

### Motivation

**Global ECE is structurally blind to minority miscalibration.**

*Proof by construction:* Consider a dataset with 900 majority samples and 100 minority samples. Suppose the model is perfectly calibrated for the majority class (ECE_majority = 0) but completely miscalibrated for the minority class (ECE_minority = 0.5).

$$\text{ECE}_{\text{global}} = \frac{900}{1000} \cdot 0 + \frac{100}{1000} \cdot 0.5 = 0.05$$

A global ECE of 0.05 would be reported as "well-calibrated" by any standard evaluation, while the minority class — the class that matters most in fraud detection, medical diagnosis, and rare event prediction — is severely miscalibrated.

### Definition

For class $c$, let $B_b^c$ denote the **true class-$c$ samples** in confidence bin $b$, and let $n_c = \sum_b |B_b^c|$ be the total number of class-$c$ samples.

$$\text{ECE}_c = \sum_{b=1}^{B} \frac{|B_b^c|}{n_c} \left| \text{acc}_c(B_b^c) - \text{conf}_c(B_b^c) \right|$$

where:
- $B_b^c$ = samples with **true label $c$** whose class-$c$ confidence falls in bin $b$
- $\text{conf}_c(B_b^c) = \frac{1}{|B_b^c|} \sum_{i \in B_b^c} P(\hat{Y}=c \mid x_i)$
  - For class 1 (minority): $\text{conf}_1 = P(y=1)$
  - For class 0 (majority): $\text{conf}_0 = 1 - P(y=1)$
- $\text{acc}_c(B_b^c) = 1.0$ always (since $B_b^c$ contains only true class-$c$ samples)

Therefore: $\text{ECE}_c = \sum_b \frac{|B_b^c|}{n_c} |1.0 - \text{conf}_c(B_b^c)|$

### Interpretation

$\text{ECE}_c$ measures: **for true class-$c$ samples, how far is the model's confidence from 1.0?**

A perfectly calibrated model assigns $P(y=c) = 1.0$ to all true class-$c$ samples (in the limit). In practice, a model that assigns $P(y=1) = 0.3$ to all minority samples has $\text{ECE}_{\text{minority}} = 0.7$ — severely miscalibrated for the minority class.

### Why This Formulation Is Correct

This is the **class-conditional calibration error** — the standard definition from the calibration literature (Guo et al. 2017, Kull et al. 2019). It answers: "Is the model's confidence for class $c$ well-calibrated specifically for true class-$c$ samples?"

**Why the "all samples" approach is wrong:** Binning all samples by $P(y=c)$ and measuring the fraction-of-$c$ in each bin is equivalent to global ECE when $c=1$ (since $P(y=1)$ is the same as the global confidence). It does not isolate minority miscalibration.

### Key Property

$$\text{ECE}_{\text{global}} \approx \frac{n_{\text{maj}}}{n} \cdot \text{ECE}_{\text{majority}} + \frac{n_{\text{min}}}{n} \cdot \text{ECE}_{\text{minority}}$$

This decomposition is what proves global ECE hides minority issues at high imbalance ratios.

---

## 3. Adaptive ECE (Equal-Mass Bins)

Standard ECE with equal-width bins can be unreliable when predictions are concentrated in a narrow range (common with overconfident models). Adaptive ECE uses equal-mass bins.

$$\text{ACE} = \frac{1}{B} \sum_{b=1}^{B} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

where each bin $B_b$ contains approximately $n/B$ samples (sorted by predicted confidence).

**Note:** ACE weights all bins equally regardless of sample count, unlike ECE which weights by bin size.

Implementation: `src/evaluation/calibration_metrics.py::_ece_equal_mass()`

---

## 4. Brier Score

$$\text{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

The Brier Score decomposes into:
$$\text{BS} = \underbrace{\text{Uncertainty}}_{\bar{y}(1-\bar{y})} - \underbrace{\text{Resolution}}_{\frac{1}{n}\sum_b |B_b|(\bar{y}_b - \bar{y})^2} + \underbrace{\text{Calibration}}_{\frac{1}{n}\sum_b |B_b|(\bar{y}_b - \bar{p}_b)^2}$$

Per-class Brier Score:
$$\text{BS}_c = \frac{1}{n_c} \sum_{i: y_i = c} (\hat{p}_i - y_i)^2$$

---

## 5. Reliability Diagram Construction

A reliability diagram plots $\text{conf}(B_b)$ (x-axis) vs $\text{acc}(B_b)$ (y-axis) for each bin $b$.

- **Perfect calibration**: all points lie on the diagonal $y = x$
- **Overconfidence**: points lie below the diagonal (model is more confident than accurate)
- **Underconfidence**: points lie above the diagonal

Our implementation always generates **side-by-side** diagrams:
1. Global reliability diagram (all samples)
2. Minority-class reliability diagram (minority samples only)

This is the key visualization that reveals minority miscalibration hidden by global metrics.
