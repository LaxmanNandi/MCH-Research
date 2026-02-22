# Paper 4: Engagement as Entanglement
## Submission Package (Local Repository)

**Location**: `c:/Users/barla/mch_experiments/papers/paper4_entanglement/v1_submission/`
**Updated**: February 21, 2026
**Status**: ✅ READY FOR SUBMISSION

---

## FILES IN THIS FOLDER

### Main Manuscript
**File**: `Paper4_v1_Hybrid.tex`
- Complete LaTeX manuscript with all sections
- ✅ Future Directions subsection added (Section 4.7)
- ✅ Supplementary Materials section added
- **COMPILE THIS for main PDF**

### Supplementary Materials
**File**: `Paper4_Supplementary.tex`
- 8-page comprehensive supplement
- ✅ Figure S4: "Lost in Conversation" experimental validation
- ✅ Complete methods, tables, discussion
- **COMPILE THIS for supplementary PDF**

### Key Figure (Generated)
**File**: `../../analysis/lost_in_conversation_tests.png`
- Referenced in Supplementary as Figure S4
- 6-panel validation figure
- Located in analysis folder

---

## WHAT'S INCLUDED

### Main Manuscript Enhancements (Feb 21)
1. **Future Directions** (Section 4.7)
   - CUD experiments "in preparation"
   - Domain expansion (legal, code, creative)
   - Longitudinal tracking
   - Intervention testing

2. **Supplementary Materials Section**
   - Lists all supplementary figures/tables
   - GitHub repository links

### Supplementary Materials (NEW)
1. **Figure S1**: Position-level ΔRCI and VRI heatmaps
2. **Figure S2**: Vendor-specific entanglement signatures
3. **Figure S3**: ESI distribution across model classes
4. **Figure S4**: "Lost in Conversation" experimental validation
   - 6-panel analysis
   - 5 experiments (Progression, Classification, Recovery, Domain, Llama)
   - 83% show increasing Var_Ratio
   - 75% fail to recover
5. **Table S1**: Complete model-position data (360 obs)
6. **Table S2**: ESI classification and recovery rates
7. **Table S3**: Detailed progression statistics

### Dataset Integrity ✅
- Uses ONLY published Paper 2 dataset (N=360, 12 models)
- NO incomplete Paper 7 pilot data included
- "Lost in Conversation" analysis uses same 360 observations

---

## COMPILATION

```bash
cd c:/Users/barla/mch_experiments/papers/paper4_entanglement/v1_submission

# Main manuscript
pdflatex Paper4_v1_Hybrid.tex
pdflatex Paper4_v1_Hybrid.tex  # Run twice

# Supplementary
pdflatex Paper4_Supplementary.tex
pdflatex Paper4_Supplementary.tex  # Run twice
```

---

## SUBMISSION CHECKLIST

### Before Submitting
- [ ] Compile both LaTeX files to PDF
- [ ] Review both PDFs for formatting
- [ ] Verify all figures are referenced correctly
- [ ] Check bibliography formatting

### Files to Submit
- [ ] Paper4_v1_Hybrid.pdf (main manuscript)
- [ ] Paper4_Supplementary.pdf (supplement)
- [ ] All figure files from `figures/` folder

### Metadata
- **Title**: Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models
- **Author**: Dr. Laxman M M, MBBS
- **Keywords**: LLMs, Context Sensitivity, Entanglement, Variance Reduction, Multi-turn Dialogue, AI Safety

---

## KEY FINDINGS

1. **r = 0.76** correlation between ΔRCI and VRI (p = 8.2×10⁻⁶⁹, N=360)
2. **Bidirectional entanglement**: Convergent (Var_Ratio < 1) vs Divergent (Var_Ratio > 1)
3. **ESI metric**: Predicts multi-turn instability (ESI < 1.0 = high risk)
4. **"Lost in Conversation"**: 75% of models fail to recover after divergence
5. **Llama anomaly**: Var_Ratio up to 7.46 at medical P30

---

## RELATED PAPERS

- **Paper 1**: DOI 10.20944/preprints202601.1881.v2
- **Paper 2**: DOI 10.20944/preprints202602.1114.v2
- **Paper 3**: Preprints.org ID 199272 (submitted)
- **Paper 5**: In preparation (safety taxonomy)
- **Paper 6**: In preparation (conservation law)
- **Paper 7**: Pilot in progress (CUD)

---

## REPOSITORY

**GitHub**: https://github.com/LaxmanNandi/MCH-Research
**Data**: All raw data, scripts, and analysis available in repository

**Ready for Preprints.org submission!** 🚀
