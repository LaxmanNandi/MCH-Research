# Paper 6: Conservation Law for Context Sensitivity

**Status**: DRAFT COMPLETE
**Title**: *Context Sensitivity and Output Variance Obey a Conservation Law Across Large Language Model Architectures*

## Overview
Capstone paper of the MCH Research Program. Discovers that the product of context sensitivity (ΔRCI) and output variance (Var_Ratio) is approximately constant within a domain, across all architectures. This conservation law unifies findings from all five prior papers under a single quantitative constraint.

## Key Findings
1. **Conservation law**: ΔRCI × Var_Ratio ≈ K(domain)
   - Medical K = 0.429 (CV = 0.170, N = 8)
   - Philosophy K = 0.301 (CV = 0.166, N = 6)

2. **Domain constants differ significantly**:
   - Mann-Whitney U = 46, p = 0.003
   - Welch's t = 3.91, p = 0.002
   - Cohen's d = 2.00 (very large)

3. **Information budget interpretation**: Models operate under a domain-specific budget. Context sensitivity and output variance trade off within this fixed budget.

4. **Safety taxonomy integration**: The four classes from Paper 5 represent different allocation strategies within the same budget.

5. **MI-based test**: Negative result — KSG estimator failed in high-dimensional space. Conservation law established via direct product test.

## Dataset
- **Configurations**: 14 model-domain runs (8 Medical, 6 Philosophy)
- **Models**: 11 unique architectures from 8 vendors
- **Data source**: Paper 2 standardized dataset (models with embedding-based Var_Ratio)
- **Location**: `/data/paper6/` (conservation product CSV + MI verification)

## Contents
- `Paper6_Draft.md`: Complete manuscript (v1.0)
- `Paper6_Definition.md`: Paper definition and scope
- `figures/`: All Paper 6 figures (4 main + variant renderings)

## Main Figures
1. Conservation law with domain hyperbolas (14 model-domain runs)
2. Product distribution by domain (within-domain clustering)
3. Domain constants comparison (K_med vs K_phil with 95% CI)
4. Safety taxonomy overlay on conservation law

## Scripts
- `scripts/analysis/paper6_conservation_law.py` — MI-based conservation test
- `scripts/analysis/paper6_conservation_product.py` — Direct product test
- `scripts/analysis/paper6_figures.py` — Figure generation
- `scripts/analysis/paper6_verify.py` — Statistical verification

## Related Documents
- Paper 4 (entanglement): Provides ΔRCI ~ VRI correlation that conservation law quantifies
- Paper 5 (safety): Taxonomy maps onto hyperbolic constraint

---

**Status**: Ready for submission
