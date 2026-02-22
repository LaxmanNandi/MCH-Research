# Git Repository Update Summary - Paper 4 Submission

**Date**: February 22, 2026
**Commits Pushed**: 2
**Branch**: master

---

## ✅ Successfully Pushed to GitHub

**Repository**: https://github.com/LaxmanNandi/MCH-Research

### Commit 1: Paper 4 Submission Files
**Hash**: 6aa7c75
**Message**: Paper 4 submitted to Preprints.org (ID: 199894)

**Files Added** (33 files, 4989 insertions):
- v1_submission/final_submitted/Paper4_Manuscript_Overleaf.tex
- v1_submission/final_submitted/Paper4_Supplementary.tex
- Paper4_All_Figures.zip (all 8 figures)
- Paper4_Supplementary_Complete.zip (combined package)
- SUBMISSION_STATUS.md
- FINAL_SUBMISSION_PACKAGE.md
- FINAL_SUPPLEMENTARY_VERIFICATION.md
- CORRELATION_CALCULATION_EXPLAINED.md
- SUBMISSION_READY.md
- archive/old_figures_pre_submission/ (3 old figures)
- figures/lost_in_conversation_tests.png (NEW)
- scripts/validate/regenerate_figure8_convergence.py
- scripts/validate/regenerate_figure_s3*.py (4 versions)
- scripts/validate/paper3_verify.py

**Files Updated** (8 figures):
1. entanglement_validation.png - **p-value corrected**
2. fig4_entanglement_multipanel.png
3. fig5_independence_rci_var.png
4. fig7_llama_safety_anomaly.png
5. figure6_gaussian_verification.png - **REGENERATED**
6. figure8_trial_convergence.png - **REGENERATED**
7. figure9_model_comparison.png - **REGENERATED**
8. README.md (updated submission status)

---

### Commit 2: Main README Update
**Hash**: dd30084
**Message**: Update main README with Paper 4 submission status

**Changes**:
- Updated Paper 4 status from "Draft complete" to "Submitted (Preprints.org ID: 199894)"
- Corrected entanglement statistics: p=2.37×10⁻⁶⁸, N=360
- Added Qwen3 235B philosophy backup data (10 trials)
- Replaced Mistral Small 24B philosophy with metrics-only version

---

## Repository Structure After Update

```
papers/paper4_entanglement/
├── README.md ✅ Updated with submission status
├── SUBMISSION_STATUS.md ✅ NEW - Complete submission record
├── FINAL_SUBMISSION_PACKAGE.md ✅ NEW - Full verification checklist
├── FINAL_SUPPLEMENTARY_VERIFICATION.md ✅ NEW - 50+ statistics verified
├── CORRELATION_CALCULATION_EXPLAINED.md ✅ NEW - r and p calculation details
├── SUBMISSION_READY.md ✅ NEW - Pre-submission summary
├── Paper4_All_Figures.zip ✅ NEW - All 8 figures (2.4 MB)
├── Paper4_Supplementary_Complete.zip ✅ NEW - Combined package (765 KB)
│
├── figures/
│   ├── FIGURE_INVENTORY.md ✅ NEW - Complete figure documentation
│   ├── entanglement_validation.png ✅ UPDATED (p-value corrected)
│   ├── fig4_entanglement_multipanel.png ✅ UPDATED
│   ├── fig5_independence_rci_var.png ✅ UPDATED
│   ├── fig7_llama_safety_anomaly.png ✅ UPDATED
│   ├── figure6_gaussian_verification.png ✅ REGENERATED (r=0.76)
│   ├── figure8_trial_convergence.png ✅ REGENERATED (12 models)
│   ├── figure9_model_comparison.png ✅ REGENERATED (correct means)
│   ├── lost_in_conversation_tests.png ✅ NEW (Figure S4)
│   └── [5 older figures kept for reference]
│
├── v1_submission/
│   ├── final_submitted/ ✅ NEW
│   │   ├── Paper4_Manuscript_Overleaf.tex
│   │   └── Paper4_Supplementary.tex
│   ├── Paper4_v1.tex
│   ├── Paper4_v1_Hybrid.tex
│   ├── Paper4_Supplementary.tex
│   └── README_SUBMISSION.md
│
└── archive/
    └── old_figures_pre_submission/ ✅ NEW
        ├── figure6_gaussian_verification.png (old r-value)
        ├── figure8_trial_convergence.png (wrong models)
        └── figure9_model_comparison.png (wrong values)
```

---

## Verification Scripts Added

All scripts added to `scripts/validate/`:
1. **regenerate_figure8_convergence.py** - Figure S2 with 12 Paper 3 models
2. **regenerate_figure_s3.py** - Initial Figure S3 attempt
3. **regenerate_figure_s3_corrected.py** - Fixed Gemini Flash value
4. **regenerate_figure_s3_final.py** - Final version with correct calculations
5. **regenerate_figure_s3_from_json.py** - Direct from JSON (authoritative)
6. **paper3_verify.py** - Paper 3 model verification

---

## Key Statistics Now Correct Across Repository

### Main Correlation
- **r** = 0.76 (correlation coefficient)
- **p** = 2.37×10⁻⁶⁸ (p-value) ✅ CORRECTED
- **N** = 360 observations (12 models × 30 positions)

### Model Assignments
- **12 Paper 3 models** (4 Philosophy, 8 Medical) ✅ VERIFIED
- Philosophy: GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash
- Medical: DeepSeek V3.1, Gemini Flash, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B

### Domain Means (Figure S3)
- **Medical**: 0.361 ✅ CORRECTED (was 0.291)
- **Philosophy**: 0.305 ✅ VERIFIED

### Lost in Conversation (Figure S4)
- 10/12 models (83%) increasing Var_Ratio ✅
- 9/12 models (75%) divergent ✅
- 9/12 models (75%) fail to recover ✅
- Llama 4 Scout P30: Var_Ratio = 7.46, ESI = 0.15 ✅

---

## Documentation Files Hierarchy

**Submission Records**:
1. SUBMISSION_STATUS.md - Official submission record
2. papers/SUBMISSION_STATUS.md - Cross-paper submission tracker

**Verification Reports**:
1. FINAL_SUBMISSION_PACKAGE.md - Complete checklist (all files ready)
2. FINAL_SUPPLEMENTARY_VERIFICATION.md - 50+ statistics verified
3. CORRELATION_CALCULATION_EXPLAINED.md - How r and p were calculated
4. SUBMISSION_READY.md - Pre-submission verification
5. figures/FIGURE_INVENTORY.md - Complete figure documentation

---

## Changes Summary

### Files Added: 33
- 8 documentation files
- 2 ZIP archives
- 3 archived old figures
- 1 new figure (lost_in_conversation_tests.png)
- 6 validation scripts
- 2 final submitted .tex files
- 4 draft .tex files
- 7 other supporting files

### Files Modified: 9
- 8 figures updated/regenerated
- 1 README updated

### Total Changes: 4,989 insertions, 7 deletions

---

## GitHub Status

✅ **All changes successfully pushed to GitHub**
- Branch: master
- Remote: origin
- Commits ahead: 0 (fully synced)

**View on GitHub**:
- Main repo: https://github.com/LaxmanNandi/MCH-Research
- Paper 4 folder: https://github.com/LaxmanNandi/MCH-Research/tree/master/papers/paper4_entanglement

---

## What's Next

Paper 4 is now:
- ✅ Submitted to Preprints.org (ID: 199894)
- ✅ All files archived in repository
- ✅ All documentation committed
- ✅ All figures verified and updated
- ✅ Zero errors remaining
- ✅ Fully synced with GitHub

**Next steps**:
- Monitor Preprints.org for approval notification
- Paper 4 will join Papers 2 and 3 in the public record
- Continue work on Papers 5, 6, and 7

---

**Prepared by**: Claude Code (Anthropic)
**Date**: February 22, 2026, 11:00 AM IST
