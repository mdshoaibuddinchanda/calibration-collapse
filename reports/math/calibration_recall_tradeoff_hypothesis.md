# Calibration-Recall Tradeoff Hypothesis Register

Every claim in the paper traces to a hypothesis here.
Every hypothesis traces to an experiment.

---

## Hypothesis 1: Resampling Destroys Minority Calibration

**Formal claim:**
$$\mathbb{E}[\text{ECE}_{\text{minority}} \mid \text{SMOTE}] > \mathbb{E}[\text{ECE}_{\text{minority}} \mid \text{no resampling}]$$

**Mechanism:**
SMOTE generates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors. These synthetic samples:
1. Are concentrated near the class boundary (by construction — SMOTE targets the minority class)
2. Introduce gradient noise during training: the model receives conflicting signals from real and synthetic samples in the same region of feature space
3. Push the model's decision boundary, causing it to assign higher confidence to minority predictions
4. But this confidence is not calibrated — it reflects the density of synthetic samples, not the true posterior probability

**Gradient noise risk score:** `src/resampling/smote_resampler.py::_compute_gradient_noise_risk()`

**Falsification condition:**
If $\text{ECE}_{\text{minority}}(\text{SMOTE}) \leq \text{ECE}_{\text{minority}}(\text{no resampling})$ across all datasets and models, Hypothesis 1 is falsified.

**Measurement:**
- Metric: `ece_minority` from `src/evaluation/calibration_metrics.py`
- Experiment: `exp001_resampling_calibration.yaml`
- Comparison: `resampler=smote` vs `resampler=none`, all other factors held constant

---

## Hypothesis 2: Global ECE Hides Minority Miscalibration

**Formal claim:**
There exist experimental conditions where:
$$\text{ECE}_{\text{global}} < \epsilon \quad \text{but} \quad \text{ECE}_{\text{minority}} \gg \epsilon$$

for some threshold $\epsilon$ (e.g., $\epsilon = 0.05$).

**Mechanism:**
By construction (see `reports/math/ece_formulation.md`, Section 2):
$$\text{ECE}_{\text{global}} = \frac{n_{\text{majority}}}{n} \cdot \text{ECE}_{\text{majority}} + \frac{n_{\text{minority}}}{n} \cdot \text{ECE}_{\text{minority}}$$

With imbalance ratio $r = n_{\text{majority}} / n_{\text{minority}}$:
$$\text{ECE}_{\text{global}} \approx \frac{r}{r+1} \cdot \text{ECE}_{\text{majority}} + \frac{1}{r+1} \cdot \text{ECE}_{\text{minority}}$$

For $r = 100$: $\text{ECE}_{\text{global}} \approx 0.99 \cdot \text{ECE}_{\text{majority}} + 0.01 \cdot \text{ECE}_{\text{minority}}$

A minority ECE of 0.5 contributes only 0.005 to global ECE — invisible under standard reporting.

**Falsification condition:**
If $\text{ECE}_{\text{minority}} \approx \text{ECE}_{\text{global}}$ across all experiments, Hypothesis 2 is falsified.

**Measurement:**
- Metrics: `ece_global` and `ece_minority` from all experiments
- Visualization: Reliability diagrams (global vs minority side-by-side)
- Experiment: All experiments in `exp001` and `exp002`

---

## Hypothesis 3: Per-Class Temperature Scaling Reduces Minority ECE

**Formal claim:**
$$\mathbb{E}[\text{ECE}_{\text{minority}} \mid \text{PCDM}] < \mathbb{E}[\text{ECE}_{\text{minority}} \mid \text{global TS}]$$

where PCDM = Per-Class Confidence Drift Monitor (`src/calibration/per_class_adaptive.py`).

**Mechanism:**
Global temperature scaling finds a single $T^*$ that minimizes NLL across all validation samples. In imbalanced settings, this $T^*$ is dominated by the majority class (which contributes most samples to the NLL). The minority class may require a different temperature.

PCDM fits $T_k^*$ separately for each class $k$, allowing the minority class to have its own calibration correction.

**Falsification condition:**
If $\text{ECE}_{\text{minority}}(\text{PCDM}) \geq \text{ECE}_{\text{minority}}(\text{global TS})$ consistently, Hypothesis 3 is falsified.

**Measurement:**
- Metric: `ece_minority`
- Experiment: `exp002_posthoc_calibration.yaml`
- Comparison: `calibrator=per_class_adaptive` vs `calibrator=temperature_scaling`

---

## Hypothesis 4: Calibration-Recall Tradeoff Exists

**Formal claim:**
There is a negative correlation between minority recall and minority ECE across (model, resampler, calibrator) combinations:
$$\text{Corr}(\text{recall}_{\text{minority}}, \text{ECE}_{\text{minority}}) < 0$$

**Mechanism:**
Methods that improve minority recall (SMOTE, ADASYN, class weighting) do so by shifting the decision boundary toward the minority class. This shift increases recall but also increases the model's confidence in minority predictions — confidence that is not grounded in the true posterior probability.

**Visualization:** Calibration-Recall Frontier plot (`src/visualization/calibration_recall_frontier.py`)

**Falsification condition:**
If the frontier plot shows no tradeoff (points scattered randomly), Hypothesis 4 is falsified.

**Measurement:**
- Metrics: `recall_minority` and `ece_minority`
- Experiment: All experiments
- Visualization: `outputs/plots/frontier/`

---

## Hypothesis 5: Catastrophic Minority Forgetting in MLP Training

**Formal claim:**
For MLP models trained with SMOTE, $\text{ECE}_{\text{minority}}$ increases monotonically with training epochs (positive drift score).

**Mechanism:**
As training progresses, the MLP memorizes the synthetic minority samples (which are concentrated near the boundary). This memorization increases confidence in minority predictions without improving true calibration — a form of overfitting to synthetic data.

**Measurement:**
- Metric: `drift_score_minority` from `src/evaluation/drift_analysis.py`
- Positive slope = calibration worsening with training
- Experiment: `exp003_focal_loss_calibration.yaml` (MLP with epoch tracking)
