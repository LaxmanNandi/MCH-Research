# Paper 5: Safety Taxonomy for Clinical AI Deployment
## Predictability as a Safety Metric: When Correct Isn't Safe

**Version:** 1.0  
**Date:** February 14, 2026  
**Status:** Defined  

---

## Abstract

This document defines Paper 5 of the MCH Research Program, which extends the Variance Ratio Index (VRI) framework to clinical deployment decisions. Cross-model accuracy verification reveals that Var_Ratio alone does not predict task completeness—instead, a four-class behavioral taxonomy emerges that maps AI systems to deployment recommendations.

---

## 1. Background

### Research Program Context

| Paper | Question | Status |
|-------|----------|--------|
| Paper 1 | Does context matter? | Published |
| Paper 2 | How does it vary? | Published |
| Paper 3 | What patterns exist? | Ready |
| Paper 4 | Why does it happen? | Ready |
| **Paper 5** | **How do we deploy safely?** | **Defined** |

### Motivation

Paper 4 identified the Llama anomaly (Var_Ratio = 7.46 at P30 Medical). This paper extends that finding by:
1. Verifying medical accuracy across all models
2. Testing the hypothesis: "Var_Ratio predicts completeness"
3. Developing a deployment framework based on findings

---

## 2. Methods

### Experimental Parameters (Identical Across Models)

| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| Max tokens | 1024 |
| Trials per model | 50 |
| Prompt | P30 Medical ("Summarize this case...") |
| Context | 29-exchange STEMI case history |

### Accuracy Rubric

16-element scoring spanning the full case journey:

| Phase | Elements |
|-------|----------|
| Presentation (P1-P5) | Chest pain, age 52 male, STEMI diagnosis |
| Diagnostics (P6-P10) | Troponin elevated, EF 45%, LAD occlusion |
| Intervention (P11-P15) | PCI performed, RV involvement, hypotension management |
| Complications (P16-P20) | Murmur/MR day 2 |
| Recovery (P21-P29) | Cardiac rehab, return to work, follow-up plan |

---

## 3. Results

### 3.1 Cross-Model Accuracy Table

| Model | Var_Ratio | Mean Acc | Std | Range | Perfect Scores |
|-------|-----------|----------|-----|-------|----------------|
| DeepSeek V3.1 | 0.48 | 83% (13.2/16) | 1.18 | 10-16 | 1/50 |
| Gemini Flash | 0.60 | 13% (2.0/16) | 2.90 | 0-16 | 1/50 |
| Ministral 14B | 0.75 | 89% (14.3/16) | 2.00 | 6-16 | 13/50 |
| Mistral Small 24B | 1.02 | 86% (13.8/16) | 1.33 | 10-16 | 5/50 |
| Qwen3 235B | 1.45 | 95% (15.2/16) | 0.96 | 12-16 | 22/50 |
| Llama 4 Maverick | 2.64 | 47% (7.5/16) | 1.40 | 5-12 | 0/50 |
| Llama 4 Scout | 7.46 | 54% (8.7/16) | 1.82 | 5-13 | 0/50 |

### 3.2 Correlation Analysis

| Correlation | Pearson r | p-value | Spearman ρ | p-value |
|-------------|-----------|---------|------------|---------|
| Var_Ratio vs Accuracy | -0.21 | 0.65 | 0.00 | 1.00 |
| Var_Ratio vs Std | -0.01 | 0.99 | -0.07 | 0.88 |
| Var_Ratio vs Range | -0.12 | 0.79 | -0.13 | 0.79 |

**Finding:** No significant linear correlation. The relationship is categorical, not continuous.

---

## 4. The 2x2 Deployment Matrix

```
                        ACCURACY
                    Low         High
                ┌───────────┬───────────┐
           Low  │  CLASS 2  │  CLASS 1  │
  Var_Ratio     │  (Empty)  │  (Ideal)  │
                ├───────────┼───────────┤
           High │  CLASS 3  │  CLASS 4  │
                │(Dangerous)│  (Rich)   │
                └───────────┴───────────┘
```

---

## 5. Four Behavioral Classes

### Class 1: IDEAL
| Attribute | Value |
|-----------|-------|
| Var_Ratio | < 1.2 |
| Accuracy | High (83-95%) |
| Models | Qwen3, Ministral, Mistral, DeepSeek |
| Behavior | Consistent AND comprehensive |
| Recommendation | **Deploy** |

### Class 2: EMPTY (Safety Filter Pathology)
| Attribute | Value |
|-----------|-------|
| Var_Ratio | Low (0.60) |
| Accuracy | Low (13%) |
| Models | Gemini Flash |
| Behavior | Convergent toward empty/refused responses |
| Recommendation | **Reconfigure safety filters** |

### Class 3: DANGEROUS (The Llama Anomaly)
| Attribute | Value |
|-----------|-------|
| Var_Ratio | High (2.64-7.46) |
| Accuracy | Low-Medium (47-54%) |
| Models | Llama Scout, Llama Maverick |
| Behavior | Unpredictable AND incomplete |
| Recommendation | **Do not deploy** |

### Class 4: RICH (Under Investigation)
| Attribute | Value |
|-----------|-------|
| Var_Ratio | Mild (1.2-2.0) |
| Accuracy | High (95%) |
| Models | Qwen3 (edge case) |
| Behavior | Diverse surface forms, consistent accuracy |
| Recommendation | **Investigate further** |

---

## 6. Var_Ratio Continuous Scale

| Range | Classification | Interpretation |
|-------|----------------|----------------|
| < 0.8 | Overly Rigid | Risk of empty convergence |
| 0.8-1.2 | Ideal | Stable and complete |
| 1.2-2.0 | Rich Divergence | Potentially beneficial variation |
| > 2.0 | Dangerous | Coherence breakdown |

---

## 7. The Llama Anomaly: Detailed Analysis

### Accuracy Breakdown

**Consistently Present (Core Facts):**

| Element | Llama Scout |
|---------|-------------|
| STEMI diagnosis | 96% |
| PCI performed | 100% |
| Chest pain | 96% |
| RV involvement | 86% |

**Stochastically Dropped (Critical Details):**

| Element | Llama Scout | DeepSeek V3.1 |
|---------|-------------|---------------|
| LAD occlusion | 22% | 100% |
| Hypotension management | 50% | — |
| Cardiac rehab | 16% | — |
| Follow-up plan | 38% | — |

### Key Observation

- No hallucinations detected
- Errors are omissions, not fabrications
- Same case, same prompt: Trial 27 captures 5/16 elements; Trial 11 captures 13/16 elements

### Characterization

> "Llama produces summaries that are factually sound but randomly incomplete—critical case details are included or omitted stochastically across trials."

---

## 8. Methodological Considerations

### Controlled experimental conditions
All models operated under identical constraints (max_tokens: 1024, temperature: 0.7, identical prompt and context). Accuracy differences across models (DeepSeek 83% vs Llama 54%) reflect architectural variation rather than infrastructural limits.

### Accuracy alone as an evaluation metric
Standard accuracy benchmarks would rate Llama's 54% as moderate but not disqualifying. The Var_Ratio dimension (7.46 for Llama Scout) captures extreme trial-to-trial inconsistency that accuracy metrics do not surface, motivating the two-dimensional framework.

### Semantic vs surface variance
Var_Ratio is computed from 384-dimensional sentence embeddings capturing semantic content. Across Llama trials, different clinical priorities are emphasized or omitted stochastically---this reflects semantic variance in information selection, not stylistic or formatting differences.

---

## 9. Contributions

1. **Falsification of simple hypothesis:** Var_Ratio does not linearly predict accuracy
2. **Four-class taxonomy:** Behavioral classification for deployment decisions
3. **2x2 deployment matrix:** Visual framework for safety assessment
4. **Unique detection capability:** Only VRI identifies Class 3 (Dangerous) models
5. **Continuous scale:** Var_Ratio thresholds for risk stratification

---

## 10. Proposed Paper Structure

| Section | Content |
|---------|---------|
| Introduction | Accuracy is necessary but not sufficient |
| Methods | Cross-model accuracy verification protocol |
| Results | 2x2 matrix and four-class taxonomy |
| The Gemini Anomaly | Safety overcorrection failure mode |
| The Llama Danger | Unique VRI detection of coherence breakdown |
| The Qwen3 Question | Rich divergence hypothesis |
| Deployment Framework | Quadrant-based recommendations |
| Discussion | Integration with MCH framework |
| Conclusion | Predictability as a safety dimension |

---

## 10a. Supplementary Analysis: Continuous vs Categorical

We tested whether Var_Ratio exhibits a U-shaped (or inverted-U) relationship with perfect scores, as proposed by the hypothesis that mild divergence (Var_Ratio 1.2-2.0) is optimal for summarization tasks.

### Results

| Analysis | Quadratic R² | F-test p | Significant? |
|----------|-------------|----------|--------------|
| Full dataset (N=8) | 0.11 | 0.72 | No |
| Excluding Gemini (N=7) | 0.14 | 0.89 | No |

The quadratic model is not significantly better than the linear model in either case. The apparent peak (Qwen3 at VR=1.45, 23/50 perfect) is driven by a single data point; within-zone variance is large (e.g., Kimi K2: 13 vs Mistral Small: 1 at nearly identical Var_Ratios).

### Best Predictor of Perfect Scores

| Metric | Spearman rho | p-value |
|--------|-------------|---------|
| Mean accuracy | 0.89-0.99 | < 0.003 |
| Score std | -0.70-0.76 | 0.049-0.054 |
| Var_Ratio | -0.39-0.49 | n.s. |

### Conclusion

The four-class behavioral taxonomy captures the structure in the data better than any continuous model. The relationship between Var_Ratio and task performance is **categorical, not continuous**: models cluster into distinct behavioral classes rather than following a smooth function. This validates the 2x2 deployment matrix as the appropriate framework.

---

## 11. Title Options

**Option A (Comprehensive):**
> Predictability as a Safety Metric: A Two-Dimensional Framework for Clinical AI Deployment

**Option B (Focused):**
> When Correct Isn't Safe: Detecting Deployment Risk with Variance Ratio Index

**Option C (Hybrid, Recommended):**
> Predictability as a Safety Metric: When Correct Isn't Safe—A Two-Dimensional Framework for Clinical AI Deployment

---

## 12. Integration with MCH Framework

### Dimensional Extension

| Dimension | Metric | Paper |
|-----------|--------|-------|
| Intelligence | RCI_COLD | Papers 1-2 |
| Entanglement | Var_Ratio / ΔRCI | Papers 3-4 |
| **Completeness** | **Task Accuracy** | **Paper 5** |

### Research Program Arc

```
Paper 1: Existence proof (ΔRCI validated)
    ↓
Paper 2: Methodology (standardized benchmark)
    ↓
Paper 3: Architectures (Type 1 vs Type 2)
    ↓
Paper 4: Theory (information-theoretic grounding)
    ↓
Paper 5: Application (deployment safety framework)
```

---

## 13. Summary

| Finding | Implication |
|---------|-------------|
| Correlation fails (r = -0.21) | Relationship is categorical |
| Four classes emerge | Taxonomy replaces linear model |
| Gemini: convergent but empty | Low Var_Ratio ≠ safe |
| Llama: divergent and incomplete | Only VRI detects this class |
| Qwen3: divergent but accurate | Mild divergence may indicate richness |

**Core Contribution:**
> A two-dimensional framework (Var_Ratio × Accuracy) that classifies AI systems into four behavioral classes with distinct deployment recommendations.

---

## References

- Paper 1: Context Curves Behavior (DOI: 10.20944/preprints202601.1881.v2)
- Paper 2: Scaling Context Sensitivity (DOI: 10.20944/preprints202602.1114.v1)
- Paper 3: Cognitive Architectures (in preparation)
- Paper 4: Entanglement Theory (in preparation)

---

## Appendix: Raw Data Location

- Cross-model accuracy results: `data/paper5/accuracy_verification/`
- Llama trial-level analysis: `data/paper5/llama_deep_dive/`
- Figures: `docs/figures/paper5/`

---

**Document Version:** 1.0  
**Last Updated:** February 14, 2026
