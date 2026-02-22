# Paper 4: Engagement as Entanglement
## Submission Package - Ready for Preprints.org

**Date**: February 22, 2026
**Status**: ✓ ALL VERIFICATIONS COMPLETE
**Author**: Dr. Laxman M M, MBBS

---

## 📁 Folder Structure

```
Paper4_Preprint_Submission/
├── main_manuscript/
│   ├── Paper4_Manuscript.tex (Standalone version)
│   ├── Paper4_Manuscript_Overleaf.tex (Overleaf-ready)
│   ├── Paper4_Manuscript.pdf (Compiled PDF)
│   └── FINAL_CHECK_REPORT.md
├── supplementary/
│   ├── Paper4_Supplementary.tex
│   └── Paper4_Supplementary.pdf
├── figures/ (9 PNG files, 2.6 MB total)
│   ├── entanglement_validation.png (Main Fig 1)
│   ├── fig4_entanglement_multipanel.png (Main Fig 2)
│   ├── fig7_llama_safety_anomaly.png (Main Fig 3)
│   ├── fig5_independence_rci_var.png (Main Fig 4)
│   ├── figure6_gaussian_verification.png (Supp Fig S1) ✓ REGENERATED
│   ├── figure8_trial_convergence.png (Supp Fig S2) ✓ REGENERATED
│   ├── figure9_model_comparison.png (Supp Fig S3) ✓ REGENERATED
│   ├── lost_in_conversation_tests.png (Supp Fig S4) ✓ NEW
│   ├── FIGURES_README.txt
│   └── FIGURE_INVENTORY.md (Complete verification)
├── archive/
│   ├── data_for_gemini/ (CSV files + Gemini prompts)
│   ├── old_figures/ (Backup files moved here)
│   └── README.md
└── SUBMISSION_READY.md (This file)
```

---

## ✓ Verification Checklist

### Main Manuscript
- [x] Paper4_Manuscript_Overleaf.tex - Overleaf-ready version
- [x] Paper4_Manuscript.tex - Standalone version
- [x] Paper4_Manuscript.pdf - Compiled and verified
- [x] All 5 main figures referenced correctly
- [x] Kaiser citation typo fixed (Line 406)
- [x] All statistics verified against data

### Supplementary Materials
- [x] Paper4_Supplementary.tex - Complete with 4 figures
- [x] Paper4_Supplementary.pdf - Compiled
- [x] Table S1: Complete model-position data (360 rows)
- [x] Table S2: ESI classification and recovery rates
- [x] All 4 supplementary figures verified

### Figures (All Current)
- [x] Figure 1: Entanglement validation (r=0.76, N=360)
- [x] Figure 2: Multipanel (6 panels, Gemini-generated)
- [x] Figure 3: Llama anomaly (P30 divergence)
- [x] Figure 4: Independence test
- [x] **Figure S1**: Gaussian verification ✓ REGENERATED
  - r-value corrected: 0.74 → 0.76
  - p-value verified: 2.37×10⁻⁶⁸
- [x] **Figure S2**: Trial convergence ✓ REGENERATED
  - Model count corrected: 12/14 → 11/12
  - Paper 3 models only (12 total)
  - Kimi K2, Ministral 14B in Medical (not Philosophy)
- [x] **Figure S3**: Model comparison ✓ REGENERATED
  - Medical mean: 0.291 → 0.361
  - Philosophy mean: 0.305 (unchanged)
  - Gemini Flash (Med): -0.133 → 0.427
- [x] **Figure S4**: Lost in Conversation ✓ NEW
  - All statistics verified: 10/12, 9/12, 75%
  - Llama trajectory correct: P30=7.46, ESI=0.15

### Data Verification
- [x] All 12 Paper 3 models correctly assigned
  - Philosophy: 4 models (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
  - Medical: 8 models (DeepSeek, Gemini Flash, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B)
- [x] Total: 360 observations (12 models × 30 positions)
- [x] All caption statistics match raw data
- [x] Response text verified for all models

### Archive Organization
- [x] Old verification files moved to archive/old_figures/
- [x] Backup readme moved to archive
- [x] Old Gemini images archived
- [x] Data CSVs organized in archive/data_for_gemini/
- [x] Gemini prompts saved for reproducibility

---

## 📊 Key Statistics (Verified)

### Paper Scope
- **Models**: 12 (4 Philosophy, 8 Medical)
- **Observations**: 360 (12 × 30 positions)
- **Trials per model**: 50
- **Total responses**: ~54,000

### Main Correlation
- **r**: 0.76 (ΔRCI vs VRI)
- **p**: 8.2 × 10⁻⁶⁹
- **N**: 360

### Figure S2 (Trial Convergence)
- **Non-significant drift**: 11/12 models (p > 0.05)
- **One significant**: Llama 4 Maverick (p=0.043)

### Figure S3 (Model Comparison)
- **Medical mean**: 0.361
- **Philosophy mean**: 0.305
- **Models**: 12 total

### Figure S4 (Lost in Conversation)
- **Progression**: 10/12 models (83%) increasing
- **Divergent**: 9/12 models (75%)
- **No recovery**: 9/12 models (75%)
- **Domain difference**: 18.4% (Medical > Philosophy)
- **Llama 4 Scout P30**: Var_Ratio = 7.46, ESI = 0.15

---

## 🔧 Fixes Applied (Feb 21-22, 2026)

### Figure S1 Issues Fixed
1. ✓ r-value: 0.74 → 0.76 (calculation verified)
2. ✓ p-value: 2.37×10⁻⁶⁸ (verified correct)
3. ✓ Regenerated with correct statistics

### Figure S2 Issues Fixed
1. ✓ Model count: Used trial_level_drci.csv with 24 runs → Used raw JSON with 12 Paper 3 models
2. ✓ Domain assignments: Kimi K2, Ministral 14B moved from Philosophy → Medical
3. ✓ Caption: "12/14 models" → "11/12 models"
4. ✓ Removed non-Paper 3 models (Claude Opus, GPT-5-2, etc.)

### Figure S3 Issues Fixed
1. ✓ Data source: paper3_correct_models.csv → Raw JSON files
2. ✓ Gemini Flash (Medical): -0.133 → 0.427 (correct value)
3. ✓ Medical mean: 0.291 → 0.361
4. ✓ Caption updated with correct values

### Figure S4 Added
1. ✓ New 6-panel figure generated via Gemini
2. ✓ All statistics verified against raw data
3. ✓ Prompt saved for reproducibility

---

## 📝 Files for Preprints.org Upload

### Required Files
1. **Main PDF**: `main_manuscript/Paper4_Manuscript.pdf`
2. **Supplementary PDF**: `supplementary/Paper4_Supplementary.pdf`
3. **Figures** (9 PNG files from `figures/` folder):
   - entanglement_validation.png
   - fig4_entanglement_multipanel.png
   - fig5_independence_rci_var.png
   - fig7_llama_safety_anomaly.png
   - figure6_gaussian_verification.png
   - figure8_trial_convergence.png
   - figure9_model_comparison.png
   - lost_in_conversation_tests.png

### Optional (Can include)
4. **Source files**: LaTeX files for transparency
5. **Data files**: CSVs in `archive/data_for_gemini/`

---

## 🎯 Submission Metadata

**Title**: Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models

**Author**: Dr. Laxman M M, MBBS
Government Duty Medical Officer, PHC Manchi, Karnataka, India

**Keywords**: large language models, context sensitivity, variance reduction, entanglement, multi-turn dialogue, clinical AI safety, lost in conversation

**Abstract**: (Lines 47-77 in Paper4_Manuscript_Overleaf.tex)

**Subject Areas**:
- Artificial Intelligence
- Natural Language Processing
- Human-Computer Interaction
- Clinical AI Safety

---

## 🔗 Related Papers

1. **Paper 2** (Published): Preprints.org ID 198986 - "Scaling Context Sensitivity"
2. **Paper 3** (Submitted): Preprints.org ID 199272 - "Domain-Specific Temporal Dynamics"
3. **Paper 4** (This submission): "Engagement as Entanglement"

---

## ✅ Final Pre-Submission Checks

- [x] All figures at 300 DPI
- [x] All statistics verified against raw data
- [x] No typos in manuscript (Kaiser citation fixed)
- [x] All captions accurate
- [x] Model assignments match Paper 3
- [x] References complete and formatted
- [x] Supplementary materials complete
- [x] Archive organized
- [x] Old files removed from active folders
- [x] Inventory documentation created

---

## 📧 Contact

**Dr. Laxman M M, MBBS**
Primary Health Centre Manchi, Bantwal Taluk
Dakshina Kannada, Karnataka, India
DNB General Medicine Resident (2026), KC General Hospital, Bangalore

GitHub: https://github.com/LaxmanNandi/MCH-Research

---

**Status**: ✅ READY FOR SUBMISSION

All verifications complete. All figures regenerated with correct data. All statistics verified. Archive organized. Submission package is ready for upload to Preprints.org.

---

**Last Updated**: February 22, 2026, 12:30 AM IST
