# Per-Class ECE Derivation

See `ece_formulation.md` Section 2 for the full derivation.

## Summary

Per-class ECE is necessary because global ECE is a weighted average dominated by the majority class:

$$\text{ECE}_{\text{global}} = \frac{n_{\text{maj}}}{n} \cdot \text{ECE}_{\text{maj}} + \frac{n_{\text{min}}}{n} \cdot \text{ECE}_{\text{min}}$$

With imbalance ratio $r$:

$$\text{ECE}_{\text{global}} = \frac{r}{r+1} \cdot \text{ECE}_{\text{maj}} + \frac{1}{r+1} \cdot \text{ECE}_{\text{min}}$$

| IR  | Weight of ECE_minority in global ECE |
|-----|--------------------------------------|
| 2   | 33.3%                                |
| 10  | 9.1%                                 |
| 50  | 1.96%                                |
| 100 | 0.99%                                |
| 500 | 0.20%                                |

At IR=100, a minority ECE of 0.5 contributes only 0.005 to global ECE — below the reporting threshold of most papers.
