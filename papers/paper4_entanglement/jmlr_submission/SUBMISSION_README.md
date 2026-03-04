# Paper 4: TMLR Submission Package

**Title**: Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models

**Submission Date**: February 25, 2026
**Journal**: Transactions on Machine Learning Research (TMLR)
**Status**: Ready for submission

---

## Submission Package Contents

### Main Files
- ✅ **Paper4_TMLR_Final.tex** (23 KB) - Main manuscript source
- ✅ **Paper4_TMLR_Submission_with_Figures.pdf** (886 KB) - Compiled PDF with embedded figures
- ✅ **tmlr.sty** (6.5 KB) - TMLR style file

### Figures (4 main figures)
- ✅ **entanglement_validation.png** (154 KB) - Figure 1: ΔRCI vs VRI correlation
- ✅ **fig4_entanglement_multipanel.png** (353 KB) - Figure 2: Multi-panel entanglement analysis
- ✅ **fig7_llama_safety_anomaly.png** (103 KB) - Figure 3: Llama divergence anomaly
- ✅ **fig5_independence_rci_var.png** (188 KB) - Figure 4: RCI vs Variance Ratio

All figures at 300 DPI, PNG format.

---

## Key Findings

1. **ΔRCI ~ VRI correlation**: r = 0.76, p = 2.37×10⁻⁶⁸ (N=360)
2. **Bidirectional entanglement**: Convergent (Var_Ratio < 1) vs Divergent (Var_Ratio > 1)
3. **Llama anomaly**: Extreme divergence at medical P30 (Var_Ratio up to 7.46)
4. **ESI metric**: Predicts multi-turn instability (ESI < 1.0 = unstable)
5. **Domain patterns**: Medical Var_Ratio ≈ 1.20, Philosophy ≈ 1.01

---

## TMLR Submission Checklist

### Format Requirements
- ✅ Anonymous submission (no author info in tex)
- ✅ Uses `\usepackage{tmlr}` style file
- ✅ Broader Impact Statement included (line 289)
- ✅ Reproducibility Statement included (line 296)
- ✅ Acknowledgments section included
- ✅ References properly formatted

### Content Requirements
- ✅ Abstract: concise summary of contribution
- ✅ Introduction: clear problem statement and contribution
- ✅ Methods: fully described and reproducible
- ✅ Results: 4 main findings with statistical tests
- ✅ Discussion: limitations and scope appropriately matched
- ✅ Conclusion: summarizes key findings
- ✅ Data availability: GitHub repository linked

### Technical Requirements
- ✅ Compiles with `pdflatex Paper4_TMLR_Final.tex`
- ✅ All figures referenced correctly
- ✅ No overfull hboxes or major warnings
- ✅ PDF includes all figures and renders correctly

---

## TMLR Submission Process

### 1. OpenReview Submission Portal
URL: https://openreview.net/group?id=TMLR

### 2. Required Information
- **Title**: (already in tex file)
- **Abstract**: (already in tex file)
- **Keywords**: large language models, context sensitivity, entanglement, variance reduction, multi-turn conversation, output stability, medical AI, model evaluation
- **TL;DR**: "We explain why LLMs get lost in conversation through variance-based entanglement framework, introducing ESI metric to predict multi-turn instability."

### 3. Upload Files
Upload to OpenReview:
1. **Main PDF**: Paper4_TMLR_Submission_with_Figures.pdf
2. **Source files** (as .zip):
   - Paper4_TMLR_Final.tex
   - tmlr.sty
   - figures/ (all 4 .png files)

### 4. Supplementary Materials
- Not required for initial submission
- Can reference GitHub repository for full data/code

### 5. Author Information
**After acceptance**, update lines 36-40 with:
```latex
\author{\name Dr. Laxman M M, MBBS \\
\addr Government Duty Medical Officer, PHC Manchi \\
\addr Bantwal Taluk, Dakshina Kannada, Karnataka, India \\
\addr DNB General Medicine Resident (2026), KC General Hospital, Bangalore \\
\addr \texttt{barlax5377@gmail.com}
}
```

---

## Key Differences from Preprints Version

### Content Changes
1. ✅ Added connection to Liu et al. "Lost in the Middle" paper
2. ✅ Removed Paper 5 and Paper 6 forward references (they're not published yet)
3. ✅ Shortened to ~350 lines (vs 430 in preprints version)
4. ✅ Removed supplementary figures (not needed for TMLR)
5. ✅ Anonymized author information

### Format Changes
1. ✅ Uses TMLR style file instead of article class
2. ✅ Added Broader Impact Statement
3. ✅ Added Reproducibility Statement
4. ✅ Inline bibliography (no separate .bib file needed)

---

## Post-Submission

### Expected Timeline
- **Initial review**: 4-8 weeks
- **Revisions** (if requested): typically one round
- **Final decision**: 2-4 months total

### Review Criteria (TMLR)
- ✅ Technical correctness
- ✅ Clarity of presentation
- ✅ Reproducibility
- ✅ Significance of contribution (empirical findings valued)

### If Accepted
- Update author information in tex
- Add OpenReview URL
- Generate camera-ready PDF with `\usepackage[accepted]{tmlr}`
- Update repository README with publication info

### If Revisions Requested
- Address reviewer comments point-by-point
- Update manuscript accordingly
- Resubmit with response letter

---

## Data Availability

**GitHub Repository**: https://github.com/LaxmanNandi/MCH-Research

Contains:
- Raw data (12 models × 50 trials × 30 positions × 3 conditions)
- Response text (required for variance computation)
- Analysis scripts
- All figures (source and generated)
- Position-level data (N=360 observations)

---

## Related Papers (MCH Research Program)

- **Paper 1**: Context Curves Behavior (Preprints.org 202601.1881.v2)
- **Paper 2**: Standardized Benchmark (Preprints.org 202602.1114.v2)
- **Paper 3**: Temporal Dynamics (Preprints.org ID: 199272)
- **Paper 4**: **THIS PAPER** (Entanglement, submitted to TMLR)
- **Paper 5**: Safety Taxonomy (in preparation)
- **Paper 6**: Conservation Constraint (in preparation)
- **Paper 7**: Context Utilization Depth (pilot complete)

---

## Notes

- **Anonymous submission**: Current version is anonymized per TMLR requirements
- **Figures embedded**: PDF includes all figures, but source .png files also included for reviewer access
- **Compilation tested**: Successfully compiles with pdflatex + tmlr.sty
- **No errors**: Clean compilation, no warnings

---

**Prepared by**: Dr. Laxman M M (assisted by Claude Code)
**Date**: February 25, 2026
**Location**: C:\Users\barla\mch_experiments\papers\paper4_entanglement\tmlr_submission\
