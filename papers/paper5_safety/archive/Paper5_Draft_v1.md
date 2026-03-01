# Stochastic Incompleteness in LLM Summarization: A Predictability Taxonomy for Clinical AI Deployment

**Dr. Laxman M M, MBBS**
Government Duty Medical Officer, PHC Manchi, Karnataka, India
DNB General Medicine Resident (2026), KC General Hospital, Bangalore

---

## Abstract

Standard accuracy benchmarks evaluate whether a language model produces correct outputs but not whether it produces them consistently. We demonstrate that accuracy and output predictability are independent dimensions by analyzing 8 medical LLMs at a clinical summarization position (P30, STEMI case history). Cross-model accuracy verification against a 16-element clinical rubric reveals no significant correlation between Variance Ratio (Var_Ratio = Var_TRUE / Var_COLD) and task accuracy (Pearson r = −0.24, p = 0.56, N = 8). This independence yields a four-class behavioral taxonomy: **IDEAL** (convergent, accurate; 4 models), **EMPTY** (convergent, inaccurate; Gemini Flash, 16% accuracy despite low variance), **DIVERGENT** (high variance, incomplete; Llama 4 Scout Var_Ratio = 7.46, Llama 4 Maverick Var_Ratio = 2.64), and **RICH** (moderate variance, highly accurate; Qwen3 235B, 95% accuracy). The DIVERGENT class exhibits a novel failure mode — stochastic incompleteness — in which summaries are factually accurate but randomly incomplete: identical prompts produce summaries covering 5/16 to 14/16 clinical elements across trials, with zero hallucinations. This failure mode is invisible to standard benchmarks. The EMPTY class demonstrates a complementary blind spot: convergent outputs that lack clinical content. We propose a two-dimensional deployment framework (Var_Ratio × Accuracy) as a minimum requirement for clinical AI assessment.

---

## 1. Introduction

The deployment of language models in clinical settings requires that outputs be both accurate and predictable. A model that produces correct summaries on average but varies wildly across trials poses a risk: any individual output may omit critical information, with no indication that information is missing.

Prior work in the MCH Research Program established that context sensitivity (ΔRCI) varies across architectures (Papers 1–2), follows task-specific temporal patterns (Paper 3), and can be characterized through variance analysis (Paper 4). Paper 4 identified an anomaly: two Llama models showed extreme output variance (Var_Ratio up to 7.46) at the medical summarization position, suggesting a potential safety concern.

This paper extends that observation into a systematic framework. We ask: does output variance predict task accuracy? If not, what is the structure of the Var_Ratio–accuracy relationship?

We find that the relationship is categorical, not continuous. Var_Ratio and accuracy are statistically independent (r = −0.24, p = 0.56), and four distinct behavioral classes emerge. Each class represents a different failure mode or success pattern that is invisible when only one dimension is assessed. In particular, we identify **stochastic incompleteness** — a failure mode in which outputs are factually sound but randomly incomplete — as a clinically relevant risk that standard accuracy benchmarks do not detect.

---

## 2. Methods

### 2.1 Experimental Design

We analyzed 8 medical LLMs at position 30 (P30) of a 29-exchange STEMI case history. P30 used a summarization prompt ("Summarize this case...") that requires integration of accumulated clinical information. Each model was evaluated for 50 independent trials under identical conditions: temperature = 0.7, max_tokens = 1024, embedding model = all-MiniLM-L6-v2 (384-dimensional).

### 2.2 Models

| Model | Vendor | Architecture | Parameters |
|-------|--------|-------------|-----------|
| DeepSeek V3.1 | DeepSeek | Dense/MoE | 671B (37B active) |
| Gemini Flash | Google | Undisclosed | Undisclosed |
| Ministral 14B | Mistral | Dense | 14B |
| Kimi K2 | Moonshot | MoE | ~1T (32B active) |
| Mistral Small 24B | Mistral | Dense | 24B |
| Qwen3 235B | Alibaba | MoE | 235B (22B active) |
| Llama 4 Maverick | Meta | MoE | 400B (17B active) |
| Llama 4 Scout | Meta | MoE | 109B (17B active) |

### 2.3 Accuracy Rubric

We developed a 16-element clinical rubric spanning the full STEMI case trajectory:

| Phase | Elements |
|-------|----------|
| Presentation (P1–P5) | STEMI diagnosis, age 52 male, chest pain |
| Diagnostics (P6–P10) | Troponin elevated, ECG ST-elevation, LAD occlusion, EF 45% |
| Intervention (P11–P15) | PCI performed, RV involvement, hypotension management |
| Complications (P16–P20) | New murmur, mitral regurgitation (day 2) |
| Recovery (P21–P29) | Secondary prevention, cardiac rehab, lifestyle modification, return to work, follow-up |

Each element was scored as present (1) or absent (0) using regex pattern matching against the response text. Accuracy = (elements present) / 16. All 50 TRUE-condition P30 responses were scored per model.

### 2.4 Variance Ratio

Var_Ratio at P30 was computed from embedding-level analysis (Paper 4): Var_Ratio = Var(TRUE embeddings) / Var(COLD embeddings) across 50 trials. Values are position-specific (P30 only) rather than model-level averages.

### 2.5 Statistical Analysis

The primary test is the Pearson correlation between Var_Ratio and mean accuracy (%) across 8 models. We also computed Spearman rank correlation and tested quadratic models to assess whether the relationship is continuous, monotonic, or categorical.

---

## 3. Results

### 3.1 Cross-model accuracy

**Table 1.** Cross-model P30 medical accuracy and variance.

| Model | Var_Ratio | Mean Acc (%) | Std | Range | Perfect (16/16) | Class |
|-------|-----------|-------------|-----|-------|-----------------|-------|
| DeepSeek V3.1 | 0.48 | 82.8 | 1.23 | 10–16 | 2/50 | IDEAL |
| Gemini Flash | 0.60 | 15.8 | 2.88 | 0–16 | 1/50 | EMPTY |
| Ministral 14B | 0.75 | 90.2 | 1.71 | 7–16 | 10/50 | IDEAL |
| Kimi K2 | 0.97 | 91.9 | 1.04 | 11–16 | 13/50 | IDEAL |
| Mistral Small 24B | 1.02 | 82.6 | 1.25 | 10–16 | 1/50 | IDEAL |
| Qwen3 235B | 1.45 | 95.2 | 0.86 | 13–16 | 23/50 | RICH |
| Llama 4 Maverick | 2.64 | 46.8 | 1.33 | 5–11 | 0/50 | DIVERGENT |
| Llama 4 Scout | 7.46 | 55.4 | 1.78 | 5–14 | 0/50 | DIVERGENT |

### 3.2 Accuracy and Var_Ratio are independent

**Table 2.** Correlation analysis.

| Metric pair | Pearson r | p-value | Spearman ρ | p-value |
|-------------|-----------|---------|------------|---------|
| Var_Ratio vs Accuracy | −0.24 | 0.56 | −0.07 | 0.87 |

No significant linear or monotonic correlation exists between Var_Ratio and accuracy. A quadratic model (testing for a U-shaped or inverted-U relationship) also failed to reach significance (R² = 0.11, F-test p = 0.72). The relationship between Var_Ratio and task performance is **categorical, not continuous**: models cluster into distinct behavioral classes rather than following a smooth function.

![Figure 4: One-dimension failure](figures/fig4_one_dimension_failure.png)

*Figure 4.* Why neither dimension alone is sufficient. Left: Var_Ratio alone cannot distinguish IDEAL from EMPTY (both low variance, but Gemini Flash has 16% accuracy). Right: Accuracy alone cannot distinguish IDEAL from DIVERGENT at the individual-trial level (a single Llama trial may score adequately, but the next may not).

---

### 3.3 Four-class predictability taxonomy

The independence of Var_Ratio and accuracy yields a 2×2 taxonomy:

```
                        ACCURACY
                    Low         High
                ┌───────────┬───────────┐
           Low  │  EMPTY    │  IDEAL    │
  Var_Ratio     │           │           │
                ├───────────┼───────────┤
           High │ DIVERGENT │  RICH     │
                │           │           │
                └───────────┴───────────┘
```

![Figure 1: Predictability matrix](figures/fig1_safety_matrix.png)

*Figure 1.* The 2×2 predictability matrix. Each quadrant represents a distinct behavioral class with different deployment implications.

**Class 1: IDEAL** (DeepSeek V3.1, Kimi K2, Ministral 14B, Mistral Small 24B). Low Var_Ratio (< 1.2) with high accuracy (83–92%). Convergent and comprehensive — outputs are both predictable and clinically complete. These models are suitable for deployment with standard monitoring.

**Class 2: EMPTY** (Gemini Flash). Low Var_Ratio (0.60) with low accuracy (15.8%). Convergent toward empty or refused responses. The model consistently produces abbreviated outputs that lack clinical detail. This class is invisible to variance-based assessment alone, since low Var_Ratio would normally suggest stability.

**Class 3: DIVERGENT** (Llama 4 Scout, Llama 4 Maverick). High Var_Ratio (2.64–7.46) with low-to-moderate accuracy (47–55%). High trial-to-trial variance correlates with incomplete task coverage. This class exhibits stochastic incompleteness (see Section 3.4).

**Class 4: RICH** (Qwen3 235B). Moderate Var_Ratio (1.45) with high accuracy (95.2%, 23/50 perfect scores). Diverse surface forms with consistent semantic accuracy. This model produces varied summaries that nonetheless cover clinical elements comprehensively.

---

### 3.4 Stochastic incompleteness: a novel failure mode

The DIVERGENT class exhibits a failure pattern we term **stochastic incompleteness**: summaries are factually accurate but randomly incomplete across trials.

**Element-level analysis** of Llama P30 TRUE-condition responses:

**Table 3.** Clinical element hit rates: DIVERGENT vs IDEAL reference.

| Element | Llama Scout | Llama Maverick | DeepSeek V3.1 (IDEAL) |
|---------|-------------|----------------|----------------------|
| STEMI diagnosis | 94% | 100% | 100% |
| PCI performed | 100% | 100% | 100% |
| Age 52 male | 100% | 100% | 100% |
| Chest pain | 96% | 58% | 100% |
| LAD occlusion | 22% | 6% | 100% |
| EF 45% | 4% | 2% | 34% |
| Cardiac rehab | 16% | 22% | 94% |
| Return to work | 16% | 0% | 64% |
| Follow-up plan | 38% | 34% | 54% |

Key observations:

1. **Zero hallucinations** detected across 100 Llama trials (50 Scout + 50 Maverick). All errors are omissions, not fabrications.
2. **Core facts are preserved**: STEMI diagnosis, PCI, and patient demographics are consistently present (>94%).
3. **Critical details are stochastically omitted**: LAD occlusion, ejection fraction, rehabilitation plan, and follow-up are present in some trials and absent in others. Llama Scout Trial 27 captures 5/16 elements; Trial 11 captures 13/16 — same model, same prompt, same context.
4. **COLD condition baselines** are near-zero: Scout 1.0/16, Maverick 1.1/16. The models cannot generate clinical summaries without context.

This failure mode is clinically concerning: a physician relying on such a summary would receive an accurate but arbitrarily partial picture, with no indication that information is missing. The output reads as a complete summary — it simply omits different elements each time.

![Figure 2: Llama variability](figures/fig2_llama_variability.png)

*Figure 2.* Trial-level variability in Llama models. Clinical element coverage varies dramatically across trials under identical conditions.

![Figure 3: Embedding archetypes](figures/fig3_archetypes_embedding.png)

*Figure 3.* Response distribution archetypes in embedding space. IDEAL models cluster tightly; DIVERGENT models show wide dispersion; EMPTY models cluster tightly but in a low-content region.

---

### 3.5 Var_Ratio thresholds and continuous assessment

While the four-class taxonomy provides categorical guidance, Var_Ratio also supports continuous risk stratification:

| Var_Ratio Range | Classification | Interpretation |
|----------------|---------------|----------------|
| < 0.8 | Overly rigid | Risk of empty convergence (EMPTY class) |
| 0.8–1.2 | Stable | Predictable and complete (IDEAL class) |
| 1.2–2.0 | Moderate divergence | Potentially beneficial variation (RICH class) |
| > 2.0 | High divergence | Stochastic incompleteness risk (DIVERGENT class) |

These thresholds are empirically motivated from the current 8-model sample and should be validated on additional models and tasks before adoption as standards.

![Figure 5: Position-level Var_Ratio](figures/fig5_position_var_ratio.png)

*Figure 5.* Position-level Var_Ratio curves for three archetypal models (IDEAL, DIVERGENT, EMPTY), showing how variance patterns differ across the full conversation.

---

## 4. Discussion

### 4.1 Accuracy is necessary but not sufficient

The central finding is that accuracy and output predictability are independent dimensions. A model with acceptable aggregate accuracy (Llama Scout, 55%) can produce individual summaries that range from clinically adequate (13/16 elements) to dangerously incomplete (5/16 elements). Conversely, a model with highly predictable outputs (Gemini Flash, Var_Ratio = 0.60) can be consistently wrong (15.8% accuracy). Neither dimension alone captures the full deployment risk.

### 4.2 Stochastic incompleteness as a failure mode

Standard hallucination detection focuses on fabricated content. Stochastic incompleteness is a complementary failure mode: content is accurate but arbitrarily partial. This is arguably more dangerous than hallucination in clinical settings, because omissions are harder to detect than fabrications. A hallucinated lab value can be flagged as inconsistent; a missing follow-up plan simply does not appear in the summary.

The term "stochastic incompleteness" captures the core mechanism: each trial samples a different subset of the available clinical information, with no systematic pattern to the omissions. The result is a model that is correct but unreliable — factually sound on any given element it includes, but unpredictable in which elements it includes.

### 4.3 The RICH class: beneficial divergence

Qwen3 235B achieves the highest accuracy (95.2%) with moderate variance (Var_Ratio = 1.45). This suggests that some output diversity may be compatible with — or even beneficial for — task completeness, provided the model consistently covers the required elements despite surface-level variation. The RICH class is small (N = 1) and should be studied further, but it provides a counterexample to the assumption that lower variance is always better.

### 4.4 Deployment framework

We propose a minimum two-dimensional assessment for clinical AI deployment:

1. **Accuracy assessment**: Score against a task-specific rubric across multiple trials.
2. **Predictability assessment**: Compute Var_Ratio across those same trials.
3. **Classify**: Map into the four-class taxonomy.
4. **Decide**: IDEAL → deploy with monitoring; EMPTY → reconfigure; DIVERGENT → do not deploy in current form; RICH → investigate further.

![Figure 6: Deployment flowchart](figures/fig6_deployment_flowchart.png)

*Figure 6.* Deployment decision framework. Both accuracy and Var_Ratio are required; neither alone is sufficient.

### 4.5 Limitations

**Sample size.** Eight models in a single domain (medical P30). The taxonomy may not generalize to other tasks or positions.

**Accuracy rubric.** The 16-element rubric is specific to the STEMI case. Different clinical scenarios would require different rubrics, and the element granularity affects accuracy scores.

**Var_Ratio computation.** P30-specific Var_Ratio values differ from model-level averages. The taxonomy is valid at P30 but may not apply at other positions.

**Class boundaries.** The IDEAL/RICH boundary (Var_Ratio = 1.2) and DIVERGENT threshold (Var_Ratio > 2.0) are empirically motivated from 8 models. Larger samples may refine these boundaries.

**Single embedding model.** Var_Ratio depends on the all-MiniLM-L6-v2 embedding space.

---

## 5. Conclusion

We demonstrate that accuracy and output predictability are independent dimensions for clinical AI assessment (r = −0.24, p = 0.56). This independence yields a four-class behavioral taxonomy — IDEAL, EMPTY, DIVERGENT, RICH — that captures failure modes invisible to either dimension alone.

The DIVERGENT class exhibits stochastic incompleteness: summaries that are factually sound but randomly incomplete, with zero hallucinations and no warning of missing information. This failure mode is undetectable by standard accuracy benchmarks, which report adequate mean performance while masking extreme trial-to-trial variability.

The EMPTY class demonstrates a complementary blind spot: convergent outputs that standard variance metrics would classify as stable but that lack clinical content entirely.

We propose that any deployment assessment for clinical AI should evaluate both accuracy (via task-specific rubrics) and predictability (via Var_Ratio or equivalent) across multiple independent trials. Single-trial evaluation is insufficient for safety-critical applications.

---

## References

1. Laxman, M. M. (2026a). Context curves behavior: Measuring context sensitivity in large language models. *Preprints.org*, 2026011881. DOI: 10.20944/preprints202601.1881.v2

2. Laxman, M. M. (2026b). Scaling context sensitivity: A standardized benchmark of ΔRCI across 25 model-domain runs. *Preprints.org*, 2026021114. DOI: 10.20944/preprints202602.1114.v2

3. Laxman, M. M. (2026c). Domain-specific temporal dynamics of context sensitivity in large language models. In preparation.

4. Laxman, M. M. (2026d). Engagement as entanglement: Variance signatures of bidirectional context coupling in large language models. In preparation.

5. Laxman, M. M. (2026e). An empirical conservation constraint on context sensitivity and output variance: Evidence across LLM architectures. In preparation.

6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

---

## Figure List

### Main Figures
1. **Figure 1:** Predictability matrix (Var_Ratio × Accuracy quadrant plot).
2. **Figure 2:** Llama trial-level variability and clinical element analysis.
3. **Figure 3:** Response distribution archetypes in embedding space.
4. **Figure 4:** One-dimension failure demonstration (why neither metric alone suffices).
5. **Figure 5:** Position-level Var_Ratio curves for three archetypes.
6. **Figure 6:** Deployment decision flowchart.

### Main Tables
1. **Table 1:** Cross-model P30 accuracy and variance with class assignments.
2. **Table 2:** Correlation analysis (Var_Ratio vs accuracy).
3. **Table 3:** Clinical element hit rates for DIVERGENT vs IDEAL models.

---

## Data Availability

- Cross-model accuracy results: `data/paper5/accuracy_verification/`
- Llama trial-level analysis: `data/paper5/llama_deep_dive/`
- Model response data: `data/medical/` (shared with Paper 2)

Repository: https://github.com/LaxmanNandi/MCH-Research

---

**Manuscript Version:** 1.0
**Date:** February 15, 2026
**Corresponding Author:** Dr. Laxman M M
