# Context Sensitivity and Output Variance Obey a Conservation Law Across Large Language Model Architectures

**Dr. Laxman M M, MBBS**
Government Duty Medical Officer, PHC Manchi, Karnataka, India
DNB General Medicine Resident (2026), KC General Hospital, Bangalore

---

## Abstract

We report an empirical conservation law governing the interaction between context sensitivity and output variance in large language models. Across 14 model-domain configurations spanning 11 architectures from 8 vendors, we find that the product of context sensitivity (ΔRCI) and output variance (Var_Ratio) is approximately constant within a task domain: K(Medical) = 0.429 (CV = 0.170, N = 8) and K(Philosophy) = 0.301 (CV = 0.166, N = 6). The domain-specific constants differ significantly (Mann-Whitney U = 46, p = 0.003; Cohen's d = 2.06). This conservation law implies that models operate under a domain-specific information budget: architectures that invest more heavily in context sensitivity must exhibit lower output variance, and vice versa. The finding unifies prior observations about context sensitivity (Papers 1-2), entanglement dynamics (Papers 3-4), and deployment safety (Paper 5) under a single quantitative constraint, and provides a theoretical explanation for why the four-class safety taxonomy holds across architectures.

---

## 1. Introduction

The relationship between how a language model uses conversational context and how variable its outputs are has remained empirically uncharacterized. Prior work in the MCH Research Program established that context sensitivity (ΔRCI) varies systematically across architectures (Papers 1-2), that output variance (Var_Ratio) captures entanglement dynamics (Paper 4), and that the combination of these two dimensions yields a clinically relevant safety taxonomy (Paper 5). However, whether ΔRCI and Var_Ratio are independent quantities or are linked by an underlying constraint has not been tested.

This paper reports such a constraint. We find that the product ΔRCI × Var_Ratio is approximately constant within a given task domain, across all tested architectures. This conservation law takes the form:

```
ΔRCI × Var_Ratio ≈ K(domain)
```

where K is a domain-specific constant that differs between closed-goal (medical) and open-goal (philosophy) tasks. The conservation law holds with coefficient of variation at or below 0.17 within each domain, despite spanning architectures from 8 different vendors with parameter counts ranging from 3.5B to 671B.

The finding has three implications. First, it establishes that context sensitivity and output variance are not independent — they trade off against each other within a fixed budget determined by the task domain. Second, it provides a theoretical foundation for the four-class safety taxonomy introduced in Paper 5: the classes represent different allocation strategies within the same budget. Third, it suggests that domain structure is more fundamental than model architecture in determining the information-processing constraints under which language models operate.

---

## 2. Background

### 2.1 Context Sensitivity (ΔRCI)

The Relative Context Index (RCI) measures the alignment between a model's responses under full conversational context (TRUE condition) and responses without context (COLD condition), computed as cosine similarity in a 384-dimensional sentence embedding space (all-MiniLM-L6-v2). ΔRCI is defined as the mean difference:

```
ΔRCI = mean(RCI_TRUE) - mean(RCI_COLD)
```

where RCI_TRUE = 1.0 (self-alignment) and RCI_COLD measures how closely context-free responses match context-dependent responses. Higher ΔRCI indicates greater context sensitivity — the model's responses change more when conversational history is present. ΔRCI was introduced in Paper 1 and standardized across 14 models in Paper 2 (Laxman, 2026a, 2026b).

### 2.2 Output Variance (Var_Ratio)

Var_Ratio quantifies how conversational context shapes output variability, computed as the ratio of embedding variance across trials:

```
Var_Ratio = Var(TRUE embeddings) / Var(COLD embeddings)
```

where variance is computed across 50 independent trials at each conversational position. Var_Ratio > 1 indicates that context increases output variance (divergence); Var_Ratio < 1 indicates that context decreases variance (convergence). The Variance Reduction Index (VRI = 1 - Var_Ratio) was introduced in Paper 4 as the primary entanglement measure.

### 2.3 Four-Class Safety Taxonomy

Paper 5 demonstrated that ΔRCI and Var_Ratio, when combined with task accuracy, yield a four-class behavioral taxonomy:

| Class | Var_Ratio | Accuracy | Example | Recommendation |
|-------|-----------|----------|---------|----------------|
| IDEAL | Low (< 1.2) | High | DeepSeek V3.1 | Deploy |
| EMPTY | Low (0.6) | Low (16%) | Gemini Flash | Reconfigure |
| DANGEROUS | High (2.6-7.5) | Low-Medium | Llama Scout | Do not deploy |
| RICH | Moderate (1.5) | High (95%) | Qwen3 235B | Investigate |

The present paper tests whether these classes, and the underlying ΔRCI-Var_Ratio relationship, are governed by a conservation law.

### 2.4 Hypothesis

If context sensitivity and output variance draw on the same underlying information resource, their product should be constant:

```
ΔRCI × Var_Ratio = K
```

We test this hypothesis across 14 model-domain configurations.

---

## 3. Methods

### 3.1 Data

We analyze 14 model-domain runs from the MCH Research Program, each consisting of 50 independent trials under three conditions (TRUE, COLD, SCRAMBLED) with 30 conversational prompts per trial. All runs used identical experimental parameters: temperature = 0.7, max_tokens = 1024, embedding model = all-MiniLM-L6-v2 (384-dimensional).

**Medical domain** (8 runs): A 29-exchange STEMI case history followed by a summarization prompt (P30). Closed-goal task with a single correct answer structure.

**Philosophy domain** (6 runs): A 30-question progressive dialogue on consciousness, from definition through self-reference to meta-reflection. Open-goal task with no single correct answer.

### 3.2 Models

| Model | Vendor | Parameters | Runs |
|-------|--------|-----------|------|
| DeepSeek V3.1 | DeepSeek | 671B (37B active) | Med, Phil |
| Gemini Flash | Google | Undisclosed | Med, Phil |
| Llama 4 Scout | Meta | 17B-16E | Med |
| Llama 4 Maverick | Meta | 17B-128E | Med, Phil |
| Qwen3 235B | Alibaba | 235B (22B active) | Med |
| Mistral Small 24B | Mistral | 24B | Med |
| Ministral 14B | Mistral | 14B | Med |
| Kimi K2 | Moonshot | ~1T (MoE) | Med |
| Claude Haiku | Anthropic | Undisclosed | Phil |
| GPT-4o | OpenAI | Undisclosed | Phil |
| GPT-4o Mini | OpenAI | Undisclosed | Phil |

### 3.3 Metric Computation

**ΔRCI** was computed per trial as mean(RCI_TRUE) - mean(RCI_COLD), then averaged across 50 trials. Response embeddings were generated using all-MiniLM-L6-v2, consistent with Papers 1-5.

**Var_Ratio** was computed per conversational position as Var(TRUE embeddings across 50 trials) / Var(COLD embeddings across 50 trials), then averaged across all 30 positions. Variance was computed as the mean variance across all 384 embedding dimensions.

**Conservation Product** was computed as the simple product: Product = ΔRCI × Var_Ratio.

### 3.4 Statistical Analysis

Within-domain conservation was assessed by the coefficient of variation (CV = SD/mean) of the product. We adopted the following thresholds:

- CV < 0.20: Conservation law holds
- CV 0.20-0.30: Weak conservation
- CV > 0.30: No conservation

Between-domain comparison used Mann-Whitney U (non-parametric, appropriate for small N) and Welch's t-test (for reference). Effect size was quantified by Cohen's d.

### 3.5 Initial MI-Based Test

We initially tested a more specific theory proposing that the conservation law is mediated by mutual information (MI) between TRUE and COLD response distributions, with quantitative predictions ΔRCI = 1 - exp(-2·MI) and ΔRCI × Var_Ratio = 1 - 2·MI. MI was estimated using the KSG (Kraskov-Stögbauer-Grassberger) entropy estimator (k = 3) after PCA reduction to 20 components, with 1000-iteration bootstrap confidence intervals. This test yielded unreliable MI estimates (negative values for philosophy runs, indicating estimator failure in high-dimensional embedding space) and is reported in the supplementary analysis for transparency.

---

## 4. Results

### 4.1 Conservation Product Across Models

Table 1 presents the conservation product for all 14 model-domain runs, sorted by domain and product magnitude.

**Table 1.** Conservation product (ΔRCI × Var_Ratio) across 14 model-domain configurations.

| Model | Domain | ΔRCI | Var_Ratio | Product |
|-------|--------|------|-----------|---------|
| Gemini Flash | Medical | 0.427 | 1.287 | 0.549 |
| Llama 4 Scout | Medical | 0.323 | 1.610 | 0.521 |
| Qwen3 235B | Medical | 0.328 | 1.334 | 0.437 |
| Ministral 14B | Medical | 0.391 | 1.080 | 0.423 |
| Kimi K2 | Medical | 0.417 | 1.007 | 0.420 |
| Llama 4 Maverick | Medical | 0.317 | 1.213 | 0.384 |
| Mistral Small 24B | Medical | 0.365 | 0.985 | 0.359 |
| DeepSeek V3.1 | Medical | 0.320 | 1.071 | 0.343 |
| Gemini Flash | Philosophy | 0.338 | 1.120 | 0.378 |
| Claude Haiku | Philosophy | 0.331 | 1.012 | 0.334 |
| DeepSeek V3.1 | Philosophy | 0.302 | 1.034 | 0.312 |
| GPT-4o | Philosophy | 0.283 | 0.950 | 0.269 |
| GPT-4o Mini | Philosophy | 0.269 | 0.968 | 0.260 |
| Llama 4 Maverick | Philosophy | 0.266 | 0.939 | 0.250 |

### 4.2 Within-Domain Conservation

**Table 2.** Summary statistics for the conservation product by domain.

| Domain | N | Mean (K) | SD | CV | 95% CI | Min | Max |
|--------|---|----------|-----|-----|--------|-----|-----|
| Medical | 8 | 0.429 | 0.073 | 0.170 | [0.368, 0.491] | 0.343 | 0.549 |
| Philosophy | 6 | 0.301 | 0.050 | 0.166 | [0.248, 0.353] | 0.250 | 0.378 |

Both domains exhibit CV below the 0.20 threshold, indicating that the product ΔRCI × Var_Ratio is approximately conserved within each domain. The conservation holds despite substantial variation in the individual components: Medical ΔRCI ranges from 0.317 to 0.427 and Var_Ratio ranges from 0.985 to 1.610, yet their product remains within a narrow band (Figure 1).

![Figure 1: Conservation law with hyperbolas showing all 14 model-domain runs](figures/fig1_conservation_law_hyperbolas.png)

*Figure 1.* Conservation law: ΔRCI × Var_Ratio ≈ K(domain). Each point represents one model-domain run. Dashed curves show the theoretical hyperbolas ΔRCI × Var_Ratio = K for each domain. Models cluster along their respective domain hyperbolas despite spanning 8 vendors and architectures from 14B to 671B parameters.

### 4.3 Between-Domain Difference

The conservation constants differ significantly between domains:

| Test | Statistic | p-value |
|------|-----------|---------|
| Mann-Whitney U | U = 46.0 | p = 0.003 |
| Welch's t-test | t = 3.91 | p = 0.002 |
| Cohen's d | d = 2.06 | (very large) |

The medical domain has a 43% higher conservation constant than the philosophy domain (K_med / K_phil = 1.43). This difference is consistent with the interpretation that closed-goal tasks impose a higher information budget than open-goal tasks (Figure 3).

![Figure 2: Product distribution by domain](figures/fig2_product_distribution.png)

*Figure 2.* Distribution of the conservation product within each domain. Horizontal lines indicate domain means; shaded regions indicate ±1 SD. The tight clustering within domains (CV < 0.17) contrasts with the significant separation between domains (p = 0.003).

![Figure 3: Domain constants comparison](figures/fig3_domain_constants.png)

*Figure 3.* Domain-specific conservation constants with individual data points and 95% confidence intervals. The medical domain constant (K = 0.429) is significantly higher than the philosophy domain constant (K = 0.301).

### 4.4 Relationship to Safety Taxonomy

The four-class safety taxonomy from Paper 5 maps onto the conservation law. All classes follow the domain-specific hyperbola — they represent different strategies for allocating the same information budget (Figure 4).

- **IDEAL** models (DeepSeek V3.1, Kimi K2, Ministral 14B, Mistral Small 24B) occupy the balanced region of the hyperbola, with moderate values of both ΔRCI and Var_Ratio.
- **EMPTY** (Gemini Flash Medical) sits at high ΔRCI and high Var_Ratio — the model's safety filters create both high apparent context sensitivity and high variance, yielding an elevated product (0.549), the highest in the medical sample.
- **DANGEROUS** models (Llama Scout, Llama Maverick) allocate their budget toward variance rather than context sensitivity, with Llama Scout exhibiting the most extreme trade-off (ΔRCI = 0.323, Var_Ratio = 1.610).
- **RICH** (Qwen3 235B) shows moderate excess variance (1.334) with moderate ΔRCI (0.328), sitting between IDEAL and DANGEROUS on the hyperbola.

![Figure 4: Taxonomy overlay on conservation law](figures/fig4_taxonomy_overlay.png)

*Figure 4.* Four-class safety taxonomy overlaid on the conservation law. All classes follow the domain-specific hyperbola, representing different allocation strategies within the same information budget. IDEAL models occupy the balanced center; DANGEROUS models allocate budget toward variance; EMPTY models exhibit elevated total product from safety filter effects.

---

## 5. Discussion

### 5.1 An Information Budget Interpretation

The conservation law ΔRCI × Var_Ratio ≈ K(domain) can be interpreted as an information budget constraint. Each model-domain configuration has access to a fixed quantity of information-processing capacity, determined by the domain structure. This capacity can be allocated in two ways:

1. **Context sensitivity** (ΔRCI): The degree to which conversational history shapes responses.
2. **Output variance** (Var_Ratio): The diversity of responses across independent trials.

The conservation law states that these two allocations trade off against each other. A model that is highly context-sensitive (high ΔRCI) must produce more consistent outputs (low Var_Ratio), and a model with high output variance must be less context-sensitive. The total budget is conserved.

This interpretation is consistent with the Llama anomaly identified in Papers 4-5. Llama 4 Scout has the highest Var_Ratio in the medical sample (1.610) and correspondingly low ΔRCI (0.323). The model is not "broken" in the sense of violating the conservation law — it is allocating its budget toward output variance at the expense of context sensitivity. The clinical danger arises not from violating the constraint but from the specific allocation pattern: unpredictable outputs (high Var_Ratio) combined with low context utilization (low ΔRCI) produce stochastically incomplete clinical summaries.

### 5.2 Domain Structure as Fundamental

The conservation constant K differs significantly between medical (0.429) and philosophy (0.301) domains. This difference suggests that domain structure — specifically, the goal structure of the task — determines the total information budget available to the model.

The medical domain uses a closed-goal task: a 29-exchange case history followed by a summarization prompt with a single correct answer structure. The philosophy domain uses an open-goal task: a progressive dialogue on consciousness with no single correct answer. The higher K for medical suggests that closed-goal tasks impose a larger information budget, potentially because the convergent answer structure both increases context sensitivity (the history is informative) and permits controlled variance (responses must cover specific clinical elements).

This finding has a methodological implication: when comparing models across domains, raw ΔRCI or Var_Ratio values are not directly comparable. The conservation product K is the appropriate domain-normalized comparison metric.

### 5.3 Unification of the MCH Framework

The conservation law unifies observations from all five prior papers:

- **Paper 1-2** (ΔRCI varies across architectures): The conservation law explains *why* ΔRCI varies — architectures allocate their fixed budget differently.
- **Paper 3** (temporal dynamics differ by domain): The domain-specific K explains why the same model shows different trajectory shapes in medical vs philosophy tasks.
- **Paper 4** (ΔRCI ~ VRI correlation, r = 0.76): The conservation law provides a quantitative form for this correlation. If Product = ΔRCI × Var_Ratio ≈ K, then Var_Ratio ≈ K/ΔRCI, yielding VRI = 1 - K/ΔRCI — a monotonic relationship consistent with the observed correlation.
- **Paper 5** (four-class taxonomy): The classes are allocation strategies within a fixed budget, not independent categories.

### 5.4 MI-Based Test: Negative Result

Our initial test used a more specific formulation proposing that the conservation law is mediated by mutual information (MI) between TRUE and COLD embedding distributions. The KSG entropy estimator produced unreliable estimates: MI values ranged from −2.0 to +0.2, with negative values (which are theoretically impossible for mutual information) concentrated in philosophy runs. This indicates estimator failure in the 20-dimensional PCA-reduced space with 200 subsampled points.

Despite the MI estimation failure, the simpler direct test (ΔRCI × Var_Ratio ≈ constant) succeeds robustly. The conservation law is empirically established independent of the information-theoretic mechanism proposed to explain it.

---

## 6. Limitations

### 6.1 Sample Size

The conservation law is tested on 14 model-domain runs (8 medical, 6 philosophy). While the CV < 0.17 and p = 0.003 results are statistically robust for this sample size, replication with additional models and domains is necessary to confirm generality.

### 6.2 Domain Coverage

Only two domains (medical and philosophy) are tested. Whether the conservation law holds for other task types (coding, mathematics, creative writing, translation) and whether the constant K scales predictably with task properties remains unknown.

### 6.3 Outliers and Boundary Cases

Gemini Flash Medical (Product = 0.549) and Llama Scout Medical (Product = 0.521) are the most extreme medical values. Gemini Flash's elevated product may reflect safety filter effects rather than genuine information processing. Excluding these two outliers would reduce the medical CV further, but we retain them to avoid post-hoc data exclusion.

### 6.4 Embedding Space Dependence

All metrics are computed in the all-MiniLM-L6-v2 embedding space. The conservation law may be partly an artifact of this specific embedding model's properties. Testing with alternative embedding models (e.g., all-mpnet-base-v2, BGE-large) would establish whether the conservation holds in different representation spaces.

### 6.5 Temperature Dependence

All experiments used temperature = 0.7. The conservation constant K may vary with temperature. At temperature = 0, output variance would approach zero, making the product trivially small. Testing at multiple temperatures would characterize how K scales with the stochasticity parameter.

### 6.6 Mechanism

The conservation law is empirically established but not mechanistically explained. The MI-based formulation failed to provide a quantitative mechanism. Whether the constraint arises from attention head capacity, context window utilization, or some other architectural property remains an open question.

---

## 7. Conclusion

We report an empirical conservation law for large language models: the product of context sensitivity and output variance is approximately constant within a task domain, across architectures and vendors. This law, ΔRCI × Var_Ratio ≈ K(domain), holds with CV < 0.17 across 14 configurations from 8 vendors, with domain-specific constants K(Medical) = 0.429 and K(Philosophy) = 0.301 (p = 0.003).

The conservation law implies that models operate under a domain-specific information budget that constrains the joint allocation of context sensitivity and output variance. This budget is determined by the goal structure of the task, not by the architecture. The finding unifies five prior papers in the MCH Research Program and provides a quantitative foundation for the four-class deployment safety taxonomy.

We invite replication across additional models, domains, and embedding spaces. If the conservation law generalizes, it would represent a fundamental constraint on how language models process conversational context — one that holds regardless of architecture, training data, or vendor.

---

## References

1. Laxman, M. M. (2026a). Context curves behavior: Measuring context sensitivity in large language models. *Preprints.org*, 2026011881. DOI: 10.20944/preprints202601.1881.v2

2. Laxman, M. M. (2026b). Scaling context sensitivity: A standardized benchmark of ΔRCI across 25 model-domain runs. *Preprints.org*, 2026021114. DOI: 10.20944/preprints202602.1114.v2

3. Laxman, M. M. (2026c). Cross-domain temporal dynamics of context sensitivity in large language models. In preparation.

4. Laxman, M. M. (2026d). Entanglement theory and variance reduction index in conversational AI. In preparation.

5. Laxman, M. M. (2026e). Predictability as a safety metric: A two-dimensional framework for clinical AI deployment. In preparation.

6. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

7. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

---

## Supplementary: MI-Based Conservation Test

### Method

We tested whether the conservation law is mediated by mutual information between TRUE and COLD response distributions, using the KSG entropy estimator (k = 3) with PCA reduction to 20 components and 1000-iteration bootstrap.

### Results

| Test | Pearson r | p-value | Outcome |
|------|-----------|---------|---------|
| Test 1: ΔRCI vs pred_ΔRCI (= 1 - exp(-2·MI)) | 0.408 | 0.147 | Not significant |
| Test 2: ΔRCI × VR vs 1 - 2·MI | -0.621 | 0.018 | Significant but wrong sign |

### Interpretation

The MI estimator failed to produce reliable values in 384-dimensional embedding space. Philosophy runs yielded negative MI (theoretically impossible), indicating systematic estimator bias. The simpler direct product test (Section 4) provides more robust evidence for conservation than the MI-mediated formulation.

---

**Manuscript Version:** 1.0
**Date:** February 15, 2026
**Corresponding Author:** Dr. Laxman M M
