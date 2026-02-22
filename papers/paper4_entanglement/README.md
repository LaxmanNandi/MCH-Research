# Paper 4: Engagement as Entanglement

**Status**: ✅ Submitted to Preprints.org (ID: 199894)
**Date**: February 22, 2026

## Manuscript Files

- **Paper4_Manuscript.tex** - Main manuscript (standalone version)
- **Paper4_Supplementary.tex** - Supplementary materials

## Figures

**Main Figures (4)**:
1. entanglement_validation.png - ΔRCI vs VRI correlation (r=0.76, p=2.37×10⁻⁶⁸)
2. fig4_entanglement_multipanel.png - 6-panel entanglement analysis
3. fig5_independence_rci_var.png - Independence verification
4. fig7_llama_safety_anomaly.png - Llama P30 divergence

**Supplementary Figures (4)**:
- figure6_gaussian_verification.png (S1) - Correlation verification
- figure8_trial_convergence.png (S2) - Trial-level convergence
- figure9_model_comparison.png (S3) - Model ΔRCI comparison
- lost_in_conversation_tests.png (S4) - 6-panel validation

All figures at 300 DPI.

## Data

**This paper uses data from Paper 2 (no duplication).**

Data location: `/data/`
- Medical models: `/data/medical/`
- Philosophy models: `/data/philosophy/`

**12 models analyzed** (subset of Paper 2 with response text):
- Philosophy (4): GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash
- Medical (8): DeepSeek V3.1, Gemini Flash, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B

**360 observations** = 12 models × 30 positions

## Key Findings

1. **Entanglement validation**: ΔRCI ~ VRI (r=0.76, p=2.37×10⁻⁶⁸)
2. **Bidirectional coupling**: Convergent (Var_Ratio < 1) and Divergent (Var_Ratio > 1)
3. **Llama safety anomaly**: Extreme instability at medical P30 (Var_Ratio up to 7.46)
4. **Domain differences**: Medical ~1.20, Philosophy ~1.01

## Archive

All submission documentation, verification reports, and old drafts are in `/archive/`.

---

**GitHub**: https://github.com/LaxmanNandi/MCH-Research
**Preprints.org**: https://www.preprints.org/manuscript/202602.XXXXX (pending approval)
