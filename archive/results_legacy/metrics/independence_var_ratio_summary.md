# Independence Test: RCI_COLD vs Variance Ratio

## Summary

**CONCLUSION: Independence supported** - Weak correlation (|r|=0.048) and non-significant regression (p=0.963)

## Data Source

- **Models**: 8 model-domain runs (4 philosophy + 4 medical)
- **Trials**: 50 trials per model per condition
- **Positions**: 30 conversation positions per trial
- **Data points**: N = 240 (8 model-domain runs x 30 positions)
  - Each data point aggregates metrics across 50 trials

## Key Findings

1. **Pooled Correlation**: r = -0.0479, p = 4.6020e-01, N = 240
2. **Regression Analysis**: beta_RCI_COLD = +0.0109, p = 9.6322e-01
3. **Per-Domain**:
   - Philosophy: r = +0.1792, p = 5.0171e-02
   - Medical: r = -0.1042, p = 2.5718e-01
4. **Per-Model**: Median r = 0.1855, IQR = [0.0213, 0.3520]
5. **PCA**: Loading correlation = 0.0000

## Interpretation

This test examines whether RCI_COLD (base capability) and Variance Ratio (entanglement measure) are independent dimensions of AI cognition.

**Key distinction:**
- **ΔRCI**: Directional entanglement signal (ΔRCI = 1 - RCI_COLD when RCI_TRUE ≈ 1.0, exact mathematical relationship)
- **Variance Ratio**: Independent entanglement axis computed from embedding variance (Var_TRUE / Var_COLD)

Unlike ΔRCI (which = 1 - RCI_COLD by construction), Variance Ratio is computed independently from embeddings and is NOT mathematically linked to RCI_COLD.

**Result: Independence supported** - negligible correlation and perfect PCA orthogonality validate dual-axis model of AI cognition.
