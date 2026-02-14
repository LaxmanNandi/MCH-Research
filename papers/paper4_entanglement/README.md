# Paper 4: Entanglement Mechanism

**Status**: DRAFT COMPLETE
**Title**: *Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling*

## Overview
Extension of Paper 2 providing information-theoretic interpretation of ΔRCI. Demonstrates that context effects represent predictability modulation via variance ratio, validating ΔRCI as a Variance Reduction Index (VRI).

## Key Findings
1. **Entanglement validation**: ΔRCI ~ VRI (r=0.76, p=1.5×10⁻⁶², N=330)
2. **Bidirectional entanglement**:
   - Convergent: Var_Ratio < 1, ΔRCI > 0 (context reduces variance)
   - Divergent: Var_Ratio > 1, ΔRCI < 0 (context increases variance)

3. **Llama safety anomaly** (Medical P30):
   - Llama 4 Maverick: Var_Ratio = 2.64, ΔRCI = -0.15
   - Llama 4 Scout: Var_Ratio = 7.46, ΔRCI = -0.22
   - Identifies safety-critical divergence class

4. **Domain architecture differences**:
   - Medical: Var_Ratio ~ 1.23 (variance-increasing)
   - Philosophy: Var_Ratio ~ 1.01 (variance-neutral)

5. **Variance sufficiency**: Simple surrogate works (no k-NN needed)

## Dataset
- **Models**: 11 (subset of Paper 2 with response text)
  - Philosophy: 4 closed (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
  - Medical: 7 (6 open + Gemini Flash closed)
- **Data points**: 330 (11 models × 30 positions)
- **Data source**: Paper 2 standardized dataset
- **Location**: `/data/` (shared, no duplication)

## Contents
- `figures/`: All Paper 4 figures (main + supplementary)
- `MODEL_LIST.md`: 11-model subset details
- `Paper4_Results.md`: Complete results and discussion

## Main Figures
1. ΔRCI vs VRI entanglement validation (r=0.76)
2. Multi-panel entanglement analysis (regime map, position patterns, domain comparison)
3. Llama safety anomaly at medical P30 (divergent variance signatures)
4. Independence test: RCI vs Variance Ratio

## Supplementary Figures
- S1: Gaussian assumption verification
- S2: Trial-level convergence analysis
- S3: Model-level ΔRCI comparison

## Safety Implications
- Var_Ratio > 3 or ESI < 1: Red flag for safety-critical tasks
- Llama models show extreme instability at task enablement (P30)
- Convergent models (DeepSeek, Gemini) stabilize under context

## Related Documents
- Parent study: `papers/paper2_standardized/`
- Companion analysis: `papers/paper3_cross_domain/`
- Safety application: `papers/paper5_safety/` (extends Llama anomaly into four-class deployment taxonomy)
- Safety note: `docs/Llama_Safety_Anomaly.md`

---

**Status**: Ready for submission
