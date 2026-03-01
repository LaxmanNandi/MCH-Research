# Paper 5: Stochastic Incompleteness

**Status**: ✅ PUBLISHED — Preprints.org (February 28, 2026)
**DOI**: 10.20944/preprints202602.2034.v1
**Title**: *Stochastic Incompleteness: A Predictability Taxonomy for Clinical AI Deployment*

## Overview
Extension of Paper 4's Llama anomaly into a comprehensive four-class predictability taxonomy. Demonstrates that accuracy alone is insufficient for deployment assessment — output predictability (Var_Ratio) is required as a second dimension.

## Key Findings
1. **Four-class taxonomy** based on Var_Ratio × Accuracy:
   - **IDEAL** (DeepSeek V3.1, Kimi K2, Ministral 14B, Mistral Small): High accuracy, convergent outputs
   - **EMPTY** (Gemini Flash): Low clinical detail (16% accuracy), convergent
   - **DIVERGENT** (Llama Scout, Llama Maverick): High trial-to-trial variance (2.6-7.5), correlates with incomplete task coverage
   - **RICH** (Qwen3 235B): Moderate variance (1.5), high accuracy (95%)

2. **Stochastic incompleteness**: Models produce factually accurate but randomly incomplete clinical summaries — the core safety finding
3. **Accuracy-VR independence**: Pearson r=-0.24, p=0.56 (N=8) — the two dimensions are statistically independent
4. **Deployment flowchart**: Two-dimensional assessment for clinical AI screening

## Dataset
- **Models**: 8 medical models with response text at P30
- **Data source**: Paper 2 standardized dataset + P30 accuracy verification
- **Location**: `/data/paper5/` (accuracy data), `/data/medical/` (model responses)

## Contents
- `Paper5_Final_Corrected.tex`: Published manuscript (LaTeX source)
- `figures/`: 6 publication figures (fig1.png through fig6.png)
- `v1_submission/`: First submission attempt (declined, low external refs)
- `archive/`: Legacy drafts, old figures, and dropped content

## Main Figures
1. **Safety matrix**: Var_Ratio × Accuracy quadrant plot with 8 models
2. **Trial variability**: Score distributions + per-element detection rates
3. **Embedding archetypes**: Response structure comparison across classes
4. **Single-metric failure**: Why accuracy alone is insufficient
5. **Position-level VR**: Var_Ratio curves across 30 positions
6. **Deployment flowchart**: Two-dimensional clinical AI screening decision tree

## Related Documents
- Parent study: `papers/paper4_entanglement/` (Llama anomaly source)
- Conservation constraint: `papers/paper6_conservation/` (taxonomy maps onto hyperbola)

---

**Published**: Preprints.org (ID: 200695) — DOI: 10.20944/preprints202602.2034.v1
