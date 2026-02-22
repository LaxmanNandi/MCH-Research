# Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models

**Dr. Laxman M M, MBBS**
Government Duty Medical Officer, PHC Manchi, Karnataka, India
DNB General Medicine Resident (2026), KC General Hospital, Bangalore

---

## Abstract

We present an entanglement framework for understanding context sensitivity in large language models. Using embedding-level variance analysis across 12 models (4 philosophy, 8 medical) and 360 position-level measurements, we demonstrate that ΔRCI — a measure of context sensitivity introduced in Papers 1–2 — tracks variance reduction in response embeddings. The correlation between ΔRCI and the Variance Reduction Index (VRI = 1 − Var_Ratio) is strong and highly significant (r = 0.76, p = 8.2 × 10⁻⁶⁹, N = 360). This relationship reveals bidirectional context coupling: convergent entanglement (Var_Ratio < 1, ΔRCI > 0) where context narrows the response distribution, and divergent entanglement (Var_Ratio > 1, ΔRCI < 0) where context widens it. Two medical models (Llama 4 Scout: Var_Ratio = 7.46; Llama 4 Maverick: Var_Ratio = 2.64) exhibit extreme divergent entanglement at the summarization position (P30), producing highly unstable outputs when task enablement is expected. We propose Var_Ratio as a practical, low-cost surrogate for entanglement measurement and identify convergent versus divergent entanglement as a deployment-relevant distinction.

---

## 1. Introduction

Context sensitivity in language models — the degree to which conversational history shapes responses — has been characterized across architectures and domains using ΔRCI (Laxman, 2026a, 2026b) and shown to follow task-specific temporal patterns (Paper 3). However, the mechanism by which context shapes responses remains empirically uncharacterized. ΔRCI captures the aggregate change in response alignment with and without context, but does not describe how the distribution of responses changes.

This paper introduces an entanglement framework that connects ΔRCI to the variance structure of response embeddings. The core insight is that context sensitivity can be understood as predictability modulation: context changes the shape of the response distribution, and ΔRCI quantifies that change. When context narrows the distribution (convergent entanglement), responses become more predictable. When context widens the distribution (divergent entanglement), responses become less predictable.

We operationalize this framework using the Variance Reduction Index (VRI), defined as:

```
VRI = 1 − Var_Ratio
Var_Ratio = Var(TRUE embeddings) / Var(COLD embeddings)
```

where variance is computed across 50 independent trials at each conversational position. Positive VRI indicates that context reduces variance (convergent entanglement); negative VRI indicates that context increases variance (divergent entanglement).

The entanglement framework has three contributions. First, it provides a mechanistic interpretation of ΔRCI as predictability modulation rather than a pure helpfulness measure. Second, it reveals bidirectional coupling — context can either stabilize or destabilize responses — resolving the previously unexplained "Sovereign" category from Paper 1. Third, it identifies a concrete safety risk: models that exhibit extreme divergent entanglement at task-critical positions produce unpredictable outputs when predictability is most needed.

---

## 2. Methods

### 2.1 Data

We analyze the 12-model subset from the MCH Research Program that has complete response text preserved (identical to Paper 3). Each model-domain run consists of 50 independent trials under three conditions (TRUE, COLD, SCRAMBLED) with 30 conversational prompts per trial. All runs used temperature = 0.7, max_tokens = 1024, and all-MiniLM-L6-v2 embeddings (384-dimensional).

### 2.2 Models

**Philosophy (4 models):** GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash.

**Medical (8 models):** DeepSeek V3.1, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Mistral Small 24B, Ministral 14B, Qwen3 235B, Gemini Flash.

### 2.3 Metrics

**ΔRCI** was computed as in Papers 1–3: per-position mean(RCI_TRUE) − mean(RCI_COLD), where RCI is the mean pairwise cosine similarity within a condition across 50 trials.

**Var_Ratio** was computed per position as Var(TRUE embeddings across 50 trials) / Var(COLD embeddings across 50 trials), where variance is the mean variance across all 384 embedding dimensions.

**VRI** (Variance Reduction Index) = 1 − Var_Ratio. Positive VRI indicates context reduces variance; negative VRI indicates context increases variance.

Note on RCI_COLD: RCI_COLD reflects responses to prompts delivered with no conversational history (the COLD condition), not cross-condition similarity. Each RCI value is computed within its own condition.

### 2.4 Statistical Analysis

The primary test is the Pearson correlation between ΔRCI and VRI across all 360 position-level measurements (12 models × 30 positions). Secondary analyses examine domain-specific patterns, model-level variance ratios, and the position 30 anomaly.

---

## 3. Results

### 3.1 Finding 1: ΔRCI tracks VRI (entanglement signal)

Across 12 model-domain runs and 30 positions, ΔRCI correlated strongly with VRI:

- **Pooled correlation:** r = 0.76, p = 8.2 × 10⁻⁶⁹ (N = 360 model-position points)
- Data: 12 model-domain runs × 30 positions = 360 points
- Each point aggregates 50 independent trials per condition
- Medical models: DeepSeek V3.1, Kimi K2, Llama 4 Maverick/Scout, Mistral Small 24B, Ministral 14B, Qwen3 235B, Gemini Flash
- Philosophy models: GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash

ΔRCI increases as context reduces response variance. This supports the entanglement interpretation: context couples the response distribution to prior information, changing the predictability of outputs.

*Note: Entanglement analysis requires actual response text to compute embedding variances. Only 12 of the 25 available model-domain runs have complete response text preserved. Expansion to additional models would require rerunning experiments with response text preservation enabled.*

![Figure 1: Entanglement validation](figures/entanglement_validation.png)

*Figure 1.* ΔRCI vs VRI across 360 model-position points. Blue: philosophy models. Red: medical models. The strong positive correlation (r = 0.76) validates the entanglement framework: higher context sensitivity corresponds to greater variance reduction.

---

### 3.2 Finding 2: Bidirectional entanglement (convergent vs divergent)

The variance ratio reveals two regimes:

- **Convergent entanglement:** Var_Ratio < 1, ΔRCI > 0. Context narrows the response distribution, making outputs more predictable.
- **Divergent entanglement:** Var_Ratio > 1, ΔRCI < 0. Context widens the response distribution, making outputs less predictable.

This bidirectional framing resolves the "Sovereign" category from Paper 1: Sovereign behavior corresponds to divergent entanglement, where context destabilizes rather than stabilizes predictability. This is not a failure of context processing but a distinct mode of coupling.

![Figure 2: Entanglement multi-panel](figures/fig4_entanglement_multipanel.png)

*Figure 2.* Multi-panel entanglement analysis. The regime map shows convergent (Var_Ratio < 1) and divergent (Var_Ratio > 1) regions, with domain-specific patterns and position-dependent dynamics.

---

### 3.3 Finding 3: Llama safety anomaly at medical P30

At medical position 30 (summarization), two Llama models exhibit extreme divergent entanglement:

- **Llama 4 Maverick:** Var_Ratio = 2.64, ΔRCI = −0.15
- **Llama 4 Scout:** Var_Ratio = 7.46, ΔRCI = −0.22

While other open medical models (Qwen3 235B, Mistral Small 24B) show mild divergence (Var_Ratio 1.02–1.45), only the Llama models exhibit extreme instability. Convergent models at P30 show Var_Ratio < 1 and positive ΔRCI (Kimi K2, Ministral 14B, DeepSeek V3.1, Gemini Flash).

This identifies a safety-relevant risk class: models that diverge under closed-goal prompts produce highly unstable, unpredictable outputs precisely when task enablement is expected. The clinical implications of this anomaly — stochastically incomplete summaries with intact factual accuracy — are explored in detail in Paper 5 (Laxman, 2026d).

![Figure 3: Llama safety anomaly](figures/fig7_llama_safety_anomaly.png)

*Figure 3.* Llama safety anomaly at medical P30. Divergent variance signatures (Var_Ratio >> 1) at the summarization position indicate extreme output instability.

---

### 3.4 Finding 4: Domain-specific variance patterns

Mean variance ratios differ by domain:

- **Philosophy:** Var_Ratio ~ 1.01 (variance-neutral on average)
- **Medical:** Var_Ratio ~ 1.20 (variance-increasing on average)

Medical prompts tend to destabilize response distributions under context, while philosophy is largely variance-neutral. This domain difference is consistent with the conservation constraint reported in Paper 6 (Laxman, 2026e).

![Figure 4: Independence test](figures/fig5_independence_rci_var.png)

*Figure 4.* RCI vs Variance Ratio across models and positions, showing the domain-specific relationship between context sensitivity and output variance.

---

### 3.5 Finding 5: Variance ratio as a practical surrogate

The variance ratio provides a practical, low-cost surrogate for entanglement measurement. ΔRCI tracks VRI (r = 0.76) without requiring k-NN entropy estimation or full mutual information computation. Computing Var_Ratio requires only response embeddings and basic variance calculations, making entanglement measurement accessible at scale. This enables the deployment assessment framework developed in Paper 5.

---

## 4. Discussion

### 4.1 Entanglement reframes ΔRCI as predictability modulation

The central conceptual shift is that ΔRCI is not merely a measure of helpfulness but a predictability modulation measure. Context changes the shape of the response distribution; ΔRCI quantifies that change. This reframing clarifies why negative ΔRCI values are not inherently problematic: they indicate divergent entanglement, which may be useful for creative tasks but is concerning for safety-critical domains.

### 4.2 Bidirectional entanglement resolves the Sovereign category

The Sovereign category from Paper 1 — models whose responses became less aligned under context — is now grounded in mechanism. Contexts that increase variance produce negative ΔRCI. This is not a failure of context usage but a distinct mode of coupling. Divergent entanglement is a valid, measurable regime rather than a missing category.

### 4.3 Safety implications: predictability is task-dependent

The Llama divergence at medical P30 highlights a concrete risk: models can become less predictable exactly when a task presupposes context. For safety-critical tasks, predictability is a requirement, not an optional characteristic. Divergent entanglement in these settings should be treated as a deployment risk.

We propose (as provisional guidance) that **Var_Ratio > 3** at critical task positions be treated as a flag for further investigation. This threshold is empirically motivated but not validated as a standard; it identifies the range where the Llama anomaly produces clinically problematic output variance.

### 4.4 Architectural interpretation

The emergence of convergent versus divergent classes suggests differences in how models handle context saturation at task-critical positions. This observation is consistent across multiple models but remains descriptive rather than causal. Future work should test whether divergence correlates with training objectives, attention patterns, or safety alignment strategies.

### 4.5 Limitations

**Model subset.** Only 12 of 25 model-domain runs have response text, limiting the analysis to 360 data points. Expansion requires rerunning experiments with text preservation.

**Embedding dependence.** Variance metrics depend on the all-MiniLM-L6-v2 embedding space. Alternative embedding models may yield different variance ratio distributions.

**Cross-model normalization.** Models with higher baseline variance (Var_COLD) will naturally produce different Var_Ratio magnitudes. We report raw values because absolute predictability change is deployment-relevant, but normalized comparisons would be appropriate for ranking models on relative sensitivity.

**Observational scope.** This study analyzes text-only interactions in two domains with a focused model set. Claims are matched to this scope.

---

## 5. Conclusion

We demonstrate that context sensitivity (ΔRCI) tracks variance reduction in response embeddings (VRI), with a pooled correlation of r = 0.76 (p = 8.2 × 10⁻⁶⁹, N = 360). This validates an entanglement framework in which context modulates the predictability of model outputs. The framework reveals bidirectional coupling: convergent entanglement narrows the response distribution, while divergent entanglement widens it.

Two Llama models exhibit extreme divergent entanglement at the medical summarization position (Var_Ratio up to 7.46), producing unstable outputs when task enablement is expected. This identifies a concrete safety risk class that is invisible to aggregate performance metrics.

The Variance Reduction Index (VRI) provides a practical, low-cost surrogate for entanglement measurement, enabling the deployment predictability framework developed in Paper 5 and the conservation constraint reported in Paper 6.

---

## References

1. Laxman, M. M. (2026a). Context curves behavior: Measuring context sensitivity in large language models. *Preprints.org*, 2026011881. DOI: 10.20944/preprints202601.1881.v2

2. Laxman, M. M. (2026b). Scaling context sensitivity: A standardized benchmark of ΔRCI across 25 model-domain runs. *Preprints.org*, 2026021114. DOI: 10.20944/preprints202602.1114.v2

3. Laxman, M. M. (2026c). Domain-specific temporal dynamics of context sensitivity in large language models. In preparation.

4. Laxman, M. M. (2026d). Stochastic incompleteness in LLM summarization: A predictability taxonomy for clinical AI deployment. In preparation.

5. Laxman, M. M. (2026e). An empirical conservation constraint on context sensitivity and output variance: Evidence across LLM architectures. In preparation.

6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

---

## Figure List

### Main Figures
1. **Figure 1:** ΔRCI vs VRI entanglement validation (r = 0.76, 12 models, 360 points).
2. **Figure 2:** Multi-panel entanglement analysis (regime map, position patterns, domain comparison).
3. **Figure 3:** Llama safety anomaly at medical P30 (divergent variance signatures).
4. **Figure 4:** RCI vs Variance Ratio across models and positions.

### Supplementary Figures
- **Figure S1:** Gaussian assumption verification for ΔRCI distributions.
- **Figure S2:** Trial-level convergence analysis (50-trial stability).
- **Figure S3:** Model-level ΔRCI comparison across domains.

![Figure S1: Gaussian verification](figures/figure6_gaussian_verification.png)
![Figure S2: Trial convergence](figures/figure8_trial_convergence.png)
![Figure S3: Model comparison](figures/figure9_model_comparison.png)

---

## Data Availability

All raw data, response text, and analysis scripts are available in the project repository:

Repository: https://github.com/LaxmanNandi/MCH-Research

---

**Manuscript Version:** 1.0
**Date:** February 15, 2026
**Corresponding Author:** Dr. Laxman M M
