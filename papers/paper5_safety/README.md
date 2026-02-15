# Paper 5: Stochastic Incompleteness in LLM Summarization

**Status**: DRAFT COMPLETE
**Title**: *Stochastic Incompleteness in LLM Summarization: A Predictability Taxonomy for Clinical AI Deployment*

## Overview
Extension of Paper 4's Llama anomaly into a comprehensive four-class predictability taxonomy. Demonstrates that accuracy alone is insufficient for deployment assessment — output predictability (Var_Ratio) is required as a second dimension.

## Key Findings
1. **Four-class taxonomy** based on Var_Ratio x Accuracy:
   - **IDEAL** (DeepSeek V3.1, Kimi K2, Ministral 14B, Mistral Small): High accuracy, convergent outputs
   - **EMPTY** (Gemini Flash): Low clinical detail (16% accuracy), convergent
   - **DIVERGENT** (Llama Scout, Llama Maverick): High trial-to-trial variance (2.6-7.5), correlates with incomplete task coverage
   - **RICH** (Qwen3 235B): Moderate variance (1.5), high accuracy (95%)

2. **Accuracy verification**: Cross-model P30 medical accuracy assessed against clinical ground truth
3. **Var_Ratio insufficiency**: Low Var_Ratio does not guarantee high accuracy (Gemini Flash counterexample)
4. **Deployment flowchart**: Two-dimensional assessment for clinical AI screening

## Dataset
- **Models**: 8 medical models with response text at P30
- **Data source**: Paper 2 standardized dataset + P30 accuracy verification
- **Location**: `/data/paper5/` (accuracy data), `/data/medical/` (model responses)

## Contents
- `Paper5_Definition.md`: Complete paper definition with methods and results
- `Llama_Safety_Note.md`: Detailed Llama anomaly documentation
- `figures/`: All Paper 5 figures (6 files)

## Main Figures
1. Predictability matrix (Var_Ratio x Accuracy quadrant plot)
2. Llama variability visualization
3. Archetype embedding comparison
4. One-dimension failure demonstration
5. Position-level Var_Ratio curves
6. Deployment decision flowchart

## Related Documents
- Parent study: `papers/paper4_entanglement/` (Llama anomaly source)
- Conservation constraint: `papers/paper6_conservation/` (taxonomy maps onto hyperbola)

---

**Status**: Ready for submission
