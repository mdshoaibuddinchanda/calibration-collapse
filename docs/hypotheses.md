# Research Hypotheses

See `reports/math/calibration_recall_tradeoff_hypothesis.md` for the full hypothesis register with formal claims, mechanisms, falsification conditions, and measurement plans.

## Quick Reference

| # | Hypothesis | Primary Metric | Experiment |
|---|-----------|----------------|------------|
| 1 | SMOTE increases ECE_minority | ece_minority | exp001 |
| 2 | Global ECE hides minority miscalibration | ece_global vs ece_minority | exp001, exp002 |
| 3 | Per-class TS reduces ECE_minority vs global TS | ece_minority | exp002 |
| 4 | Calibration-Recall tradeoff exists | recall_minority vs ece_minority | All |
| 5 | Catastrophic Minority Forgetting in MLP | drift_score_minority | exp003 |
