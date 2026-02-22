# MCH Research Program — Submission Status
**Updated**: February 21, 2026

## Paper 4: Engagement as Entanglement
**Status**: ✅ SUBMISSION-READY

### Main Manuscript
- File: `papers/paper4_entanglement/v1_submission/Paper4_v1_Hybrid.tex`
- **NEW**: Future Directions subsection added (Section 4.7)
  - Mentions CUD experiments in progress
  - Domain expansion plans
  - Longitudinal tracking proposals
- **NEW**: Supplementary Materials section added
  - References 4 figures + 2 tables
  - GitHub repository links

### Supplementary Materials
- File: `papers/paper4_entanglement/v1_submission/Paper4_Supplementary.tex`
- **NEW**: Complete 8-page supplement created
- **Figure S4**: "Lost in Conversation" experimental validation
  - 6-panel analysis using 360 observations from Paper 2 dataset
  - 5 experiments: Progression, Classification, Recovery, Domain, Llama trajectory
  - Results: 83% show increasing Var_Ratio, 75% fail to recover
  - Validates Laban et al. (2025) findings
- **Methods**: Detailed experimental design for all tests
- **Tables**: ESI classification, recovery rates, progression statistics

### Key Additions
1. **Engagement-as-entanglement framing** throughout Discussion
2. **ESI (Entanglement Stability Index)** as predictive tool
3. **"Lost in Conversation" mechanism** explained via divergent entanglement
4. **Future work** mentions Paper 7 CUD concept without revealing incomplete data

### Dataset Integrity
✅ Uses ONLY published Paper 2 dataset (N=360, 12 models)
✅ No mixing with Paper 7 pilot data
✅ All statistics verified against repository data

### Ready for:
- Preprints.org submission
- Journal submission (Nature Machine Intelligence, JMLR, ICML/NeurIPS)

---

## Paper 5: Stochastic Incompleteness
**Status**: 📝 MARKDOWN COMPLETE, LATEX PENDING

### Content Ready
- File: `papers/paper5_safety/Paper5_Draft_v1.md` (291 lines)
- Four-class taxonomy: IDEAL / EMPTY / DIVERGENT / RICH
- Stochastic incompleteness failure mode
- 16-element clinical rubric validation
- 8 medical models at P30

### Needs LaTeX Conversion
- Template: Match Paper 4 structure
- Figures: 6 main figures ready in `/figures/`
- Tables: Accuracy verification, classification, element hit rates
- Future Directions: Add CUD mention

### Next Steps
1. Convert markdown to LaTeX
2. Add Future Directions subsection
3. Verify all figures compile
4. Submit to clinical AI journal

---

## Paper 6: Conservation Constraint
**Status**: 📝 MARKDOWN COMPLETE, LATEX PENDING

### Content Ready
- File: `papers/paper6_conservation/Paper6_Draft_v1.md` (334 lines)
- Conservation law: ΔRCI × Var_Ratio ≈ K(domain)
- Medical K=0.429, Philosophy K=0.301
- Mann-Whitney p=0.003, Cohen's d=2.06
- 14 model-domain configurations

### Needs LaTeX Conversion
- Template: Match Paper 4 structure
- Figures: 4 main figures in `/figures/`
- Tables: Conservation products, domain statistics
- Future Directions: Add CUD depth prediction

### Next Steps
1. Convert markdown to LaTeX
2. Add Future Directions subsection
3. Cross-reference Papers 4 & 5
4. Submit to ML theory journal

---

## Paper 7: Context Utilization Depth (CUD)
**Status**: 🔬 PILOT INCOMPLETE — DO NOT SUBMIT YET

### Pilot Progress
| Model | Medical P30 | Philosophy P15 | Status |
|-------|------------|----------------|--------|
| DeepSeek V3.1 | 50/50 | 50/50 | ✅ COMPLETE |
| Gemini Flash | 50/50 | 50/50 | ✅ COMPLETE |
| Llama 4 Maverick | 50/50 | 39/50 (11 lost) | ⚠️ Needs rerun |
| Qwen3 235B | ~29/50 | 0/50 | 🔄 IN PROGRESS |

### Not Ready Because:
- Qwen3 235B incomplete (21 trials remaining)
- Maverick philosophy needs 11-trial rerun
- Only 4 pilot models (target: 8-12 for publication)

### Mentioned In Papers 4-6:
✅ Future Directions sections reference CUD "in preparation"
❌ NO pilot data included in Papers 4-6 submissions

---

## Submission Strategy

### Recommended Timeline

**Week 1 (Feb 21-28)**
1. ✅ Paper 4: Ready to submit NOW
2. 📝 Papers 5 & 6: Create LaTeX versions

**Week 2 (Mar 1-7)**
3. 🚀 Submit Papers 4, 5, 6 in parallel
   - Paper 4 → Nature MI / JMLR
   - Paper 5 → Clinical AI journal (npj Digital Medicine)
   - Paper 6 → ML theory journal (JMLR / TMLR)

**March-May**
4. 🔬 Complete Paper 7 pilot
5. 📊 Analyze full CUD results
6. 🚀 Submit Paper 7 in May 2026

### Cross-Referencing
- Papers 4, 5, 6: Reference each other as "submitted" or "in preparation"
- All cite Paper 2 (published) for dataset
- Paper 7: Cited as "in preparation" in Future Directions only

---

## Key Decisions Made

✅ **Dataset integrity**: Papers 4-6 use ONLY published N=360 dataset
✅ **"Lost in Conversation"**: Added to Paper 4 Supplementary (uses same 360 data)
✅ **Future work**: CUD mentioned conceptually, no incomplete data revealed
✅ **Parallel submission**: All three papers ready for simultaneous submission

---

## Files Created/Modified Today

### Paper 4
- `Paper4_v1_Hybrid.tex` — Added Future Directions subsection (lines 288-302)
- `Paper4_v1_Hybrid.tex` — Added Supplementary Materials section (after Data Availability)
- `Paper4_Supplementary.tex` — **NEW FILE** (8 pages, complete supplement)

### Analysis
- `scripts/analysis/test_lost_in_conversation.py` — Experimental validation script
- `analysis/lost_in_conversation_tests.png` — 6-panel figure (Figure S4)
- `analysis/lost_in_conversation_summary.csv` — Aggregate statistics
- `analysis/lost_in_conversation_progression.csv` — Model trajectories

### Status Tracking
- `papers/SUBMISSION_STATUS.md` — **THIS FILE**

---

## Next Actions Required

### Option A: Full LaTeX Conversion (Recommended)
Create complete LaTeX manuscripts for Papers 5 & 6 matching Paper 4 template:
- Convert all sections from markdown
- Add Future Directions subsections
- Include all figures/tables
- **Estimated time**: 2-3 hours

### Option B: Quick LaTeX Stubs
Create minimal LaTeX scaffolds:
- User completes detailed content later
- Provides structure and formatting
- **Estimated time**: 30 minutes

### Option C: Submit Paper 4 First
- Submit Paper 4 immediately to Preprints.org
- Work on Papers 5 & 6 conversion afterward
- **Advantage**: Get Paper 4 priority established

---

## Recommendation

**Proceed with Option A** — Create full LaTeX versions of Papers 5 & 6 now while all context is fresh. This allows parallel submission of all three papers by end of February 2026.

**User should**: Confirm approval to proceed with full LaTeX conversion, or select alternative option.
