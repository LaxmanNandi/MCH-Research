# Independence Test: ΔRCI vs RCI_COLD (INVALID TEST)

## Summary

**CONCLUSION: This test is INVALID** - ΔRCI and RCI_COLD show perfect negative correlation (r = -1.000, exact) because they are mathematically linked by construction in this dataset.

## Key Findings

1. **Pooled Correlation**: r = -1.0000 (exact), p = 0.0000e+00, N = 240
2. **Regression Analysis**: beta_RCI_COLD = -1.0000 (exact), p = 0.0000e+00
3. **Per-Domain**:
   - Philosophy: r = -1.0000 (exact), p = 0.0000e+00
   - Medical: r = -1.0000 (exact), p = 0.0000e+00
4. **Per-Model**: Median r = -1.0000, IQR = [-1.0000, -1.0000]
5. **PCA**: Loading correlation = 0.0000

## Interpretation

**This correlation is exact (r = -1.000), not empirical, due to dataset-specific property:**

In the analyzed JSON files, `alignments['true']` values are always 1.0 (self-similarity scores by construction). This makes:
- ΔRCI = RCI_TRUE - RCI_COLD
- ΔRCI = 1.0 - RCI_COLD (exact mathematical relationship in this dataset)

**Important caveats:**
- This r = -1.000 is **dataset/computation-specific**, not a universal property
- When RCI_TRUE is computed differently (e.g., as mean similarity to other responses), it ranges from 0.5 to 0.99, and the correlation is no longer -1.0
- This makes ΔRCI a **directional signal of entanglement** (positive = helpful, negative = harmful) but NOT an independent axis from RCI_COLD in this specific dataset

**For proper independence testing, see:** [independence_var_ratio_summary.md](independence_var_ratio_summary.md)

Variance Ratio is computed independently from embeddings and shows true independence from RCI_COLD (r = -0.048, p = 0.460).
