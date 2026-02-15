# Llama Safety Anomaly (Paper 4)

This note documents the Llama medical P30 anomaly observed in the entanglement dataset.

---

## Summary

At medical position 30 (summarization), Llama 4 Scout shows extreme divergent entanglement (Var_Ratio >> 1) and negative DRCI, indicating unstable, highly unpredictable outputs under Type 2 prompts.

---

## Evidence (from entanglement_position_data.csv)

Source: `analysis/entanglement_position_data.csv` (model-position rows).

P30 values (medical domain):

| Model | Position | DRCI | Var_Ratio | ESI | TRUE self-sim | COLD self-sim | Cross-sim |
|-------|----------|------|----------|-----|---------------|---------------|-----------|
| Llama 4 Scout (Med) | 30 | -0.220632 | 7.463016 | 0.154727 | 0.745230 | 0.965862 | 0.024954 |
| Llama 4 Maverick (Med) | 30 | -0.152056 | 2.644340 | 0.608147 | 0.755472 | 0.907528 | 0.106847 |
| Gemini Flash (Med) | 30 | +0.227474 | 0.603515 | 2.522162 | 0.653747 | 0.426273 | 0.177272 |
| DeepSeek V3.1 (Med) | 30 | +0.208052 | 0.477634 | 1.914367 | 0.809764 | 0.601713 | 0.185581 |

ESI definition:

```
ESI = 1 / |1 - Var_Ratio|
```

---

## Interpretation

- Llama 4 Scout (Med) at P30 shows Var_Ratio = 7.463016 and ESI = 0.154727 (extreme divergence).
- Divergence occurs precisely when the task presupposes context (Type 2 summarization).
- COLD self-sim remains high while TRUE variance explodes, indicating intact baseline coherence with a breakdown in context coupling.

---

## Notes

- The mean variance summary file (`analysis/entanglement_variance_summary.csv`) reports model means and will not show the P30 spike. Use the position-level file above for anomaly verification.
- This finding is summarized in Paper 4 (Results/Discussion and Claims & Evidence tables) but documented here as a dedicated audit note for reviewers.
