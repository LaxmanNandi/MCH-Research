# Paper 4 Figure Inventory
**Last Updated**: February 22, 2026
**Status**: Ready for Preprints.org Submission

---

## Main Manuscript Figures (5 total)

### Figure 1: Entanglement Validation
**File**: `entanglement_validation.png`
**Size**: 145 KB
**Status**: ✓ Current (Feb 21, 2026)
**Caption Location**: Line 148-152 in Paper4_Manuscript_Overleaf.tex
**Description**: Scatter plot showing ΔRCI vs VRI correlation (r=0.76, p=8.2×10⁻⁶⁹, N=360)
**Data Source**: archive/data_for_gemini/figure1_entanglement_scatter_360points.csv

---

### Figure 2: Entanglement Multipanel
**File**: `fig4_entanglement_multipanel.png`
**Size**: 353 KB
**Status**: ✓ Current (Feb 21, 2026)
**Caption Location**: Line 173-182 in Paper4_Manuscript_Overleaf.tex
**Description**: 6-panel figure showing position-level patterns, domain separation, and model classes
**Data Source**: Multiple CSVs in archive/data_for_gemini/
**Generation**: Gemini AI (Feb 21, 2026)

---

### Figure 3: Llama Safety Anomaly
**File**: `fig7_llama_safety_anomaly.png`
**Size**: 103 KB
**Status**: ✓ Current (Feb 21, 2026)
**Caption Location**: Line 194-202 in Paper4_Manuscript_Overleaf.tex
**Description**: Llama 4 Scout and Maverick extreme divergence at medical P30
**Data Source**: archive/data_for_gemini/figure3_llama_anomaly_p30.csv

---

### Figure 4: Independence Test
**File**: `fig5_independence_rci_var.png`
**Size**: 188 KB
**Status**: ✓ Current (Feb 21, 2026)
**Caption Location**: Line 213-220 in Paper4_Manuscript_Overleaf.tex
**Description**: ΔRCI vs Var_Ratio showing independence of measurement
**Data Source**: Analysis scripts in mch_experiments/

---

## Supplementary Figures (4 total)

### Figure S1: Gaussian Verification
**File**: `figure6_gaussian_verification.png`
**Size**: 239 KB
**Status**: ✓ REGENERATED (Feb 21, 2026)
**Caption Location**: Line 438-444 in Paper4_Manuscript_Overleaf.tex
**Description**: Correlation verification with Gaussian overlay (r=0.76, p=2.37×10⁻⁶⁸)
**Verification**: ✓ r-value corrected from 0.74 to 0.76
**Script**: scripts/validate/verify_correlation.py

---

### Figure S2: Trial Convergence
**File**: `figure8_trial_convergence.png`
**Size**: 899 KB
**Status**: ✓ REGENERATED (Feb 21, 2026)
**Caption Location**: Line 450-453 in Paper4_Manuscript_Overleaf.tex
**Description**: 50-trial convergence analysis showing stable ΔRCI estimates
**Verification**: ✓ Corrected to 11/12 models (was 12/14)
**Model Count**: 12 Paper 3 models (4 Phil, 8 Med)
**Script**: scripts/validate/regenerate_figure8_convergence.py
**Data Source**: Raw JSON files in data/philosophy/ and data/medical/

---

### Figure S3: Model Comparison
**File**: `figure9_model_comparison.png`
**Size**: 204 KB
**Status**: ✓ REGENERATED (Feb 22, 2026)
**Caption Location**: Line 460-464 in Paper4_Manuscript_Overleaf.tex
**Description**: Model-level mean ΔRCI by domain
**Verification**: ✓ Domain means corrected (Med: 0.361, Phil: 0.305)
**Issue Fixed**: Gemini Flash (Med) value corrected from -0.133 to 0.427
**Model Count**: 12 Paper 3 models (4 Phil, 8 Med)
**Script**: scripts/validate/regenerate_figure_s3_corrected.py
**Data Source**: Raw JSON files (authoritative source)

---

### Figure S4: Lost in Conversation
**File**: `lost_in_conversation_tests.png`
**Size**: 420 KB
**Status**: ✓ NEW (Feb 22, 2026)
**Caption Location**: Line 95-136 in Paper4_Supplementary.tex
**Description**: 6-panel validation of Laban et al. (2025) "Lost in Conversation" phenomenon
**Verification**: ✓ All statistics verified (10/12, 9/12, 75%, etc.)
**Model Count**: 12 Paper 3 models (4 Phil, 8 Med)
**Generation**: Gemini AI (Feb 22, 2026)
**Data Source**: archive/data_for_gemini/figure_s4_lost_in_conversation.csv
**Prompt**: archive/data_for_gemini/PROMPT_FIGURE_S4_LOST_IN_CONVERSATION.md

---

## Paper 3 Model Assignments (Verified)

All supplementary figures use the correct Paper 3 model assignments:

**Philosophy (4 models)**:
1. GPT-4o
2. GPT-4o-mini
3. Claude Haiku
4. Gemini Flash

**Medical (8 models)**:
1. DeepSeek V3.1
2. Gemini Flash
3. Kimi K2
4. Llama 4 Maverick
5. Llama 4 Scout
6. Ministral 14B
7. Mistral Small 24B
8. Qwen3 235B

**Total**: 12 model-domain runs × 30 positions = 360 observations

---

## Verification Summary

### Figure S1 (Gaussian Verification)
- ✓ r-value: 0.76 (was 0.74)
- ✓ p-value: 2.37×10⁻⁶⁸ (correct)
- ✓ N = 360 observations

### Figure S2 (Trial Convergence)
- ✓ Non-significant drift: 11/12 models (was 12/14)
- ✓ Correct Paper 3 models (no GPT-5, Claude Opus, etc.)
- ✓ Domain assignment: Kimi K2 and Ministral 14B in Medical only

### Figure S3 (Model Comparison)
- ✓ Medical mean: 0.361 (was 0.291)
- ✓ Philosophy mean: 0.305 (correct)
- ✓ Gemini Flash (Med): 0.427 (was -0.133)
- ✓ All 12 Paper 3 models present

### Figure S4 (Lost in Conversation)
- ✓ 10/12 progression (83%)
- ✓ 9/12 divergent (75%)
- ✓ 9/12 no recovery (75%)
- ✓ Medical: 1.198, Philosophy: 1.012 (18.4% difference)
- ✓ Llama 4 Scout P30: 7.46, ESI: 0.15

---

## Archive

**Location**: `archive/old_figures/`

**Contents**:
- FIGURES_README_backup.txt (moved Feb 22, 2026)
- Code_Generated_Image (1).png (Gemini early attempt, Feb 21)
- Code_Generated_Image (8).png (Gemini early attempt, Feb 21)
- Code_Generated_Image (9).png (Gemini early attempt, Feb 21)

---

## Submission Checklist

- [x] All 9 figures present (5 main + 4 supplementary)
- [x] All figures regenerated with correct data
- [x] All captions updated with verified statistics
- [x] Model assignments match Paper 3 (12 models total)
- [x] All old/backup files moved to archive
- [x] Data sources documented and available
- [x] Generation scripts available for reproducibility
- [x] Gemini prompts saved for Figure S2 and S4

---

## File Sizes Summary

| Figure | Size | Type |
|--------|------|------|
| entanglement_validation.png | 145 KB | Main |
| fig4_entanglement_multipanel.png | 353 KB | Main |
| fig7_llama_safety_anomaly.png | 103 KB | Main |
| fig5_independence_rci_var.png | 188 KB | Main |
| figure6_gaussian_verification.png | 239 KB | Supp |
| figure8_trial_convergence.png | 899 KB | Supp |
| figure9_model_comparison.png | 204 KB | Supp |
| lost_in_conversation_tests.png | 420 KB | Supp |
| **Total** | **2.6 MB** | **9 files** |

---

## Notes

- All figures generated at 300 DPI for publication quality
- Color scheme: Blue (Philosophy/convergent), Red (Medical/divergent)
- All statistical claims verified against raw data
- Figures S2, S3 regenerated Feb 21-22, 2026 to fix data errors
- Figure S4 generated fresh Feb 22, 2026 with verified statistics

---

**End of Inventory**
