# Methods Appendix: Entanglement Analysis (Paper 4)

---

## A. Data sources and scope
- Models: 11 model-domain runs (4 philosophy, 7 medical).
- Trials: 50 independent trials per model per condition.
- Positions: 30 prompts per conversation.
- Conditions: TRUE, COLD, SCRAMBLED.
- Unit of analysis: model × position mean response embeddings (aggregated across 50 trials).

Total pooled points for correlation analyses: N = 11 × 30 = 330 model-position points.
Each point represents aggregate statistics (mean, variance) computed over 50 trials.

---

## B. Response embeddings and variance
For each model, domain, and position, we embed responses (TRUE and COLD) using a fixed sentence embedding model (same as prior DRCI analyses). For each condition, we compute the variance of embeddings across trials.

Definitions:

```
Var_TRUE = variance of TRUE embeddings across trials
Var_COLD = variance of COLD embeddings across trials
Var_Ratio = Var_TRUE / Var_COLD
```

We use Var_Ratio as a proxy for context-induced predictability change.

---

## C. DRCI computation
We compute Delta RCI (DRCI) as:

```
DRCI = mean(RCI_TRUE) - mean(RCI_COLD)
```

Where:
- RCI_TRUE = mean self-similarity of each response to all other responses in TRUE condition
- RCI_COLD = mean self-similarity of each response to all other responses in COLD condition

For each trial and position, we compute the mean cosine similarity between each response and all other responses within the same condition. DRCI is then aggregated (mean) at the model × position level to align with Var_Ratio.

---

## D. MI proxy definition
We define an MI proxy derived from variance ratio:

```
MI_Proxy = 1 - Var_Ratio
```

This is monotonic with the direction of predictability change: positive values indicate reduced variance (more predictability) under context; negative values indicate increased variance (less predictability).

---

## E. Correlation analysis
We compute Pearson correlation between DRCI and MI_Proxy over pooled model-position points (N = 330). Reported statistics include r and p-value for the pooled test.

We also compute per-model correlations to examine consistency and directionality. If MI_Proxy is defined as Var_Ratio instead of 1 - Var_Ratio, the correlation sign will invert; therefore we report sign conventions explicitly with the proxy definition.

---

## F. Entanglement regimes
We categorize entanglement regimes by Var_Ratio and DRCI sign:

- Convergent: Var_Ratio < 1 and DRCI > 0
- Divergent: Var_Ratio > 1 and DRCI < 0
- Neutral: Var_Ratio approximately 1 and DRCI approximately 0

---

## G. Entanglement Stability Index (ESI)
We define a stability metric:

```
ESI = 1 / |1 - Var_Ratio|
```

Lower ESI indicates greater divergence from variance neutrality. Thresholds (e.g., ESI < 1) are treated as provisional, not normative.

---

## H. Multiple comparisons and robustness
- If per-model correlations are reported as a set, apply correction (e.g., Benjamini-Hochberg) to control false discovery rate.
- Sensitivity to embedding model should be validated by repeating the variance analysis with at least one alternative embedding model.

---

## I. Reproducibility checklist
- Fix random seed for any stochastic sampling.
- Use identical prompts and temperature settings across conditions.
- Report trial count per model and any missing trials.
- Document embedding model and version.
