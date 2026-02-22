# Paper 4: Entanglement Mechanism

**Status**: ✅ SUBMITTED TO PREPRINTS.ORG (ID: 199894)
**Submitted**: February 22, 2026
**Title**: *Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models*

## Overview
Extension of Paper 2 providing information-theoretic interpretation of ΔRCI. Demonstrates that context effects represent predictability modulation via variance ratio, validating ΔRCI as a Variance Reduction Index (VRI).

## Key Findings
1. **Entanglement validation**: ΔRCI ~ VRI (r=0.76, p=2.37×10⁻⁶⁸, N=360)
2. **Bidirectional entanglement**:
   - Convergent: Var_Ratio < 1, ΔRCI > 0 (context reduces variance)
   - Divergent: Var_Ratio > 1, ΔRCI < 0 (context increases variance)

3. **Llama safety anomaly** (Medical P30):
   - Llama 4 Maverick: Var_Ratio = 2.64, ΔRCI = -0.15
   - Llama 4 Scout: Var_Ratio = 7.46, ΔRCI = -0.22
   - Identifies safety-critical divergence class

4. **Domain architecture differences**:
   - Medical: Var_Ratio ~ 1.20 (variance-increasing)
   - Philosophy: Var_Ratio ~ 1.01 (variance-neutral)

5. **Variance sufficiency**: Simple surrogate works (no k-NN needed)

## Dataset
- **Models**: 12 (subset of Paper 2 with response text)
  - Philosophy: 4 closed (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
  - Medical: 8 (7 open + Gemini Flash closed)
- **Data points**: 360 (12 models × 30 positions)
- **Data source**: Paper 2 standardized dataset
- **Location**: `/data/` (shared, no duplication)

## Contents
- `figures/`: All Paper 4 figures (main + supplementary)
- `MODEL_LIST.md`: 12-model subset details
- `Paper4_Results.md`: Complete results and discussion

## Main Figures
1. ΔRCI vs VRI entanglement validation (r=0.76)
2. Multi-panel entanglement analysis (regime map, position patterns, domain comparison)
3. Llama safety anomaly at medical P30 (divergent variance signatures)
4. Independence test: RCI vs Variance Ratio

## Supplementary Figures
- S1: Gaussian assumption verification (r=0.76, p=2.37×10⁻⁶⁸)
- S2: Trial-level convergence analysis (11/12 models non-significant)
- S3: Model-level ΔRCI comparison (Medical: 0.361, Philosophy: 0.305)
- S4: Lost in Conversation experimental validation (6-panel analysis)

## Safety Implications
- Var_Ratio > 3 or ESI < 1: Red flag for safety-critical tasks
- Llama models show extreme instability at task enablement (P30)
- Convergent models (DeepSeek, Kimi K2, Gemini) stabilize under context

## Related Documents
- Parent study: `papers/paper2_standardized/`
- Companion analysis: `papers/paper3_cross_domain/`
- Safety application: `papers/paper5_safety/` (extends Llama anomaly into four-class deployment taxonomy)
- Safety note: `docs/Llama_Safety_Anomaly.md`

## Submission Files

All submitted materials available in:
- **v1_submission/final_submitted/**: Final .tex files submitted to Preprints.org
- **Paper4_All_Figures.zip**: All 8 figures (2.4 MB)
- **Paper4_Supplementary_Complete.zip**: Combined supplementary package (765 KB)
- **SUBMISSION_STATUS.md**: Complete submission record
- **FINAL_SUBMISSION_PACKAGE.md**: Full verification checklist

## Verification Reports
- **FINAL_SUPPLEMENTARY_VERIFICATION.md**: 50+ statistics verified
- **CORRELATION_CALCULATION_EXPLAINED.md**: Pearson r and p calculation details
- **SUBMISSION_READY.md**: Pre-submission verification summary

---

**Status**: ✅ Successfully submitted to Preprints.org (ID: 199894)
**All figures regenerated and verified**: February 21-22, 2026
**Zero errors remaining**
