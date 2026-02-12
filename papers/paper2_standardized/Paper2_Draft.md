# Cross-Domain Measurement of Context Sensitivity in Large Language Models: Medical vs Philosophical Reasoning

**Authors**: Dr. Laxman M M (Nandi), MBBS
**Affiliation**: Primary Health Centre Manchi, Karnataka, India
**Status**: DRAFT — February 2026

---

## Abstract

We present a standardized cross-domain framework for measuring context sensitivity in large language models (LLMs) using the Delta Relational Coherence Index (ΔRCI). Across 25 model-domain runs (14 unique models, 50 trials each, 112,500 total responses), we compare medical (closed-goal) and philosophical (open-goal) reasoning domains using a three-condition protocol (TRUE/COLD/SCRAMBLED). We find that: (1) both domains elicit robust positive context sensitivity (mean ΔRCI: philosophy=0.317, medical=0.308), with no significant domain-level difference (U=51, p=0.149); (2) medical domain exhibits substantially higher inter-model variance (SD=0.131 vs 0.045), driven by a Gemini Flash safety-filter anomaly (ΔRCI=−0.133); (3) vendor signatures show marginal differentiation (F(7,17)=2.31, p=0.075), with Moonshot (Kimi K2) showing highest context sensitivity and Google lowest; (4) the expected information hierarchy (SCRAMBLED > COLD) is inverted in 23/24 model-domain runs, suggesting that COLD baselines may capture residual context effects; and (5) position-level analysis reveals domain-specific temporal signatures consistent with theoretical predictions. This dataset provides the first standardized benchmark for cross-domain context sensitivity measurement in state-of-the-art LLMs.

**Keywords**: Context sensitivity, ΔRCI, cross-domain AI evaluation, medical reasoning, philosophical reasoning, LLM benchmarking

---

## 1. Introduction

### 1.1 Background

Large language models increasingly serve as reasoning tools across diverse domains, from medical diagnostics to philosophical inquiry. How domain structure shapes model behavior—particularly sensitivity to conversational context—remains poorly understood. Prior work (Laxman, 2026; Paper 1) introduced the Delta Relational Coherence Index (ΔRCI) and demonstrated dramatic behavioral mode-switching between domains using 7 closed models. However, that study used aggregate metrics, mixed trial definitions, and lacked open-weight model comparisons.

### 1.2 Research Gap

No existing benchmark provides:
- Standardized cross-domain context sensitivity measurement
- Unified methodology across open and closed architectures
- Position-level temporal analysis across task types
- Systematic vendor-level behavioral characterization

### 1.3 Research Questions

1. **RQ1**: How does domain structure (closed-goal vs open-goal) affect aggregate context sensitivity?
2. **RQ2**: Do temporal dynamics differ systematically between domains at the position level?
3. **RQ3**: Are architectural differences (open vs closed models) domain-specific?
4. **RQ4**: Do vendor-level behavioral signatures persist across domains?

### 1.4 Contributions

1. **Standardized framework**: Unified 50-trial methodology with corrected trial definition across 14 models and 2 domains
2. **Cross-domain validation**: First systematic comparison of ΔRCI in medical (closed-goal) vs philosophical (open-goal) reasoning
3. **Architectural diversity**: Balanced open (7) and closed (5–6) model inclusion in both domains
4. **Baseline dataset**: 25 model-domain runs providing reproducible benchmarks for 14 state-of-the-art LLMs
5. **Anomaly detection**: Identification of safety-filter-induced context sensitivity inversion (Gemini Flash medical)

---

## 2. Related Work

### 2.1 Context Sensitivity in LLMs
- Multi-turn coherence studies
- Prompt sensitivity and instruction following
- Conversational grounding in dialogue systems

### 2.2 Cross-Domain AI Evaluation
- MMLU, HELM, and general-purpose benchmarks
- Domain-specific evaluation (medical: MedQA, philosophical reasoning tasks)
- Gap: No cross-domain *behavioral* metric (as opposed to accuracy)

### 2.3 Paper 1 Foundation
- Introduced ΔRCI and three-condition protocol
- Demonstrated domain flip effect (Cohen's d > 2.7)
- Limitations: aggregate-only, mixed methodology, closed models only

---

## 3. Methodology

### 3.1 Experimental Design

**Three-condition protocol** applied to each trial:
- **TRUE**: Model receives coherent 29-message conversational history before prompt
- **COLD**: Model receives prompt with no prior context
- **SCRAMBLED**: Model receives same 29 messages in randomized order before prompt

**ΔRCI computation**:
```
ΔRCI = mean(RCI_TRUE) − mean(RCI_COLD)
```
Where RCI is computed via cosine similarity of response embeddings (all-MiniLM-L6-v2, 384D).

### 3.2 Domains

| Feature | Medical (Closed-Goal) | Philosophy (Open-Goal) |
|---------|----------------------|----------------------|
| Scenario | 52-year-old STEMI case | Consciousness inquiry |
| Goal structure | Diagnostic/therapeutic targets | No single correct answer |
| Prompt count | 30 per trial | 30 per trial |
| Expected pattern | U-shaped + P30 spike | Inverted-U |

### 3.3 Models

**14 unique models across 25 model-domain runs:**

| Vendor | Model | Medical | Philosophy |
|--------|-------|---------|------------|
| OpenAI | GPT-4o | ✓ | ✓ |
| OpenAI | GPT-4o-mini | ✓ | ✓ |
| OpenAI | GPT-5.2 | ✓ | ✓ |
| Anthropic | Claude Haiku | ✓ | ✓ |
| Anthropic | Claude Opus | ✓ | — |
| Google | Gemini Flash | ✓ | ✓ |
| DeepSeek | DeepSeek V3.1 | ✓ | ✓ |
| Moonshot | Kimi K2 | ✓ | ✓ |
| Meta | Llama 4 Maverick | ✓ | ✓ |
| Meta | Llama 4 Scout | ✓ | ✓ |
| Mistral | Mistral Small 24B | ✓ | ✓ |
| Mistral | Ministral 14B | ✓ | ✓ |
| Alibaba | Qwen3 235B | ✓ | ✓ |

- **Medical**: 13 models (6 closed + 7 open)
- **Philosophy**: 12 models (5 closed + 7 open)
- **12 models** appear in both domains (paired comparison)

### 3.4 Parameters

- **Trials per model**: 50 (standardized)
- **Temperature**: 0.7
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2 (384D)
- **API providers**: Direct API (closed), Together AI (open)

### 3.5 Data Scale

| Metric | Count |
|--------|-------|
| Unique models | 14 |
| Model-domain runs | 25 |
| Trials per run | 50 |
| Prompts per trial | 30 |
| Conditions per trial | 3 (TRUE, COLD, SCRAMBLED) |
| Total trials | 1,250 |
| Total responses | 112,500 |

---

## 4. Results

### 4.1 Dataset Overview (Figure 1)

Figure 1 presents a heatmap of mean ΔRCI across all 14 models and both domains. Key observations:
- **23/25 model-domain runs** show positive ΔRCI (context enhances coherence)
- **Kimi K2** shows highest sensitivity in both domains (philosophy: 0.428, medical: 0.417)
- **Gemini Flash medical** is the sole negative outlier (ΔRCI = −0.133), attributed to safety-filter interference with medical content
- **Claude Opus** appears only in medical domain (philosophy data not collected with standardized methodology)

### 4.2 Domain Comparison (Figure 2)

**Aggregate comparison**: No significant difference between domains (Mann-Whitney U=51, p=0.149).
- Philosophy: mean ΔRCI = 0.317 ± 0.045 (n=12)
- Medical: mean ΔRCI = 0.308 ± 0.131 (n=13)

**Paired comparison** (12 models in both domains): Most models show similar ΔRCI across domains, with notable exceptions:
- **Gemini Flash**: Dramatic divergence (philosophy: 0.338, medical: −0.133), Δ=0.471
- **GPT-5.2**: Higher in medical (0.379 vs 0.308)
- **Kimi K2**: Consistently highest in both domains

**Interpretation**: Domain structure does not systematically shift *aggregate* context sensitivity, but individual model responses to domain can be dramatic (Gemini Flash).

### 4.3 Vendor Signatures (Figure 3)

One-way ANOVA across 8 vendors: F(7,17) = 2.31, p = 0.075 (marginal significance).

**Vendor ranking** (by mean ΔRCI across model-domain runs):
1. **Moonshot** (Kimi K2): 0.423 — highest, most consistent
2. **Mistral** (Mistral Small + Ministral): 0.352 — strong, balanced
3. **Anthropic** (Claude Haiku + Opus): 0.336 — moderate-high
4. **Alibaba** (Qwen3 235B): 0.325 — moderate
5. **DeepSeek** (V3.1): 0.312 — moderate
6. **OpenAI** (GPT-4o/mini/5.2): 0.310 — moderate, high within-vendor variance
7. **Meta** (Llama 4 Maverick/Scout): 0.301 — lower range
8. **Google** (Gemini Flash): 0.103 — lowest, driven by medical anomaly

**Note**: Google's low ranking is entirely driven by the Gemini Flash medical anomaly. Excluding that run, Gemini Flash philosophy (0.338) ranks competitively.

### 4.4 Position-Level Patterns (Figure 4)

Position-level ΔRCI analysis reveals domain-specific temporal signatures:

**Philosophy domain** (12 models):
- Noisy but generally elevated sensitivity across positions 1–29
- Mean domain trajectory shows slight upward trend
- No dramatic P30 effect

**Medical domain** (12 models with position data):
- Higher amplitude oscillations
- Several models show elevated P30 (position 30 = summarization prompt)
- Greater inter-model variability at each position

**Interpretation**: Position-level patterns are consistent with theoretical predictions from Paper 3 (inverted-U for philosophy, U-shaped for medical), though individual model noise is substantial at 50-trial resolution.

### 4.5 Information Hierarchy (Figure 5)

The theoretical prediction that SCRAMBLED context should provide more information than COLD (no context) was tested:

**Expected**: ΔRCI_SCRAMBLED > ΔRCI_COLD (scrambled context retains some useful information)
**Observed**: Hierarchy holds in only **1/24** model-domain runs

This unexpected finding suggests that:
1. COLD baselines may capture residual effects not present in scrambled conditions
2. Scrambled context may actively interfere with coherent generation
3. The ΔRCI metric may not be sensitive to partial information in scrambled sequences

### 4.6 Model Rankings (Figure 6)

**Philosophy rankings** (top 3):
1. Kimi K2 (O): 0.428 ± 0.023
2. Ministral 14B (O): 0.373 ± 0.033
3. Gemini Flash (C): 0.338 ± 0.023

**Medical rankings** (top 3):
1. Kimi K2 (O): 0.417 ± 0.016
2. Ministral 14B (O): 0.391 ± 0.034
3. GPT-5.2 (C): 0.379 ± 0.021

**Cross-domain consistency**: Kimi K2 and Ministral 14B rank #1 and #2 in both domains. GPT-4o and Llama 4 Maverick consistently rank in the lower half.

---

## 5. Discussion

### 5.1 Domain Invariance of Aggregate ΔRCI

The lack of significant domain-level difference (p=0.149) suggests that aggregate context sensitivity is relatively domain-invariant—models that are sensitive to context in one domain tend to be sensitive in the other. This supports ΔRCI as a generalizable metric rather than a domain-specific artifact.

However, the medical domain's much higher variance (SD=0.131 vs 0.045) indicates that closed-goal tasks create more extreme behavioral differentiation between models.

### 5.2 The Gemini Flash Medical Anomaly

Gemini Flash shows the most dramatic domain effect: positive in philosophy (0.338) but negative in medical (−0.133). This is attributed to safety filters that activate on medical content, disrupting coherent context utilization. This has important implications for medical AI deployment—safety mechanisms can paradoxically *reduce* response quality by interfering with context integration.

### 5.3 Open vs Closed Architecture

Open models show competitive or superior context sensitivity compared to closed models in both domains:
- Medical open mean: 0.348 vs closed mean: 0.257 (excluding Gemini Flash anomaly: 0.335)
- Philosophy open mean: 0.325 vs closed mean: 0.306

This suggests that open-weight models, despite generally smaller parameter counts, can achieve comparable context sensitivity.

### 5.4 Vendor Clustering

The marginal vendor effect (p=0.075) suggests that organizational-level design decisions (training data, RLHF procedures, safety tuning) create subtle but potentially meaningful behavioral signatures. Moonshot's consistent dominance and Google's safety-filter-driven anomaly represent the extremes.

### 5.5 Information Hierarchy Inversion

The near-universal inversion of the expected SCRAMBLED > COLD hierarchy (23/24 runs) is a significant methodological finding. It suggests that scrambled context may be actively harmful rather than partially informative—disrupted conversational structure may confuse models more than absence of context.

### 5.6 Limitations

1. **Single scenario per domain**: One medical case (STEMI) and one philosophical topic (consciousness)
2. **Embedding model ceiling**: all-MiniLM-L6-v2 may not capture all semantic distinctions
3. **Temperature fixed at 0.7**: Other settings may yield different patterns
4. **Claude Opus**: Only 43 trials in medical, absent from philosophy
5. **Position-level noise**: 50 trials provide limited statistical power for 30-position analysis (addressed in Paper 3 with focused subset)

---

## 6. Conclusion

This study establishes a standardized cross-domain framework for measuring context sensitivity in LLMs. Across 14 models and 112,500 responses, we find that:

1. **Context sensitivity is robust and positive** for nearly all models in both domains (23/25 runs)
2. **Domain structure shapes variance, not mean**: Medical and philosophical domains yield similar average ΔRCI but dramatically different inter-model spread
3. **Safety mechanisms can invert context sensitivity**: Gemini Flash medical anomaly demonstrates deployment-critical risk
4. **Open models compete with closed**: No systematic architectural disadvantage for open-weight models
5. **Vendor signatures are detectable**: Organizational design choices create marginal but consistent behavioral patterns

This dataset and methodology provide the foundation for deeper analyses of temporal dynamics (Paper 3) and information-theoretic mechanisms (Paper 4).

---

## 7. Figures

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Dataset overview heatmap (14 models × 2 domains) | `fig1_dataset_overview.png` |
| Figure 2 | Domain comparison (violin + paired bar) | `fig2_domain_comparison.png` |
| Figure 3 | Vendor signatures (bar + ANOVA) | `fig3_vendor_signatures.png` |
| Figure 4 | Position-level temporal patterns | `fig4_position_patterns.png` |
| Figure 5 | Information hierarchy (SCRAMBLED vs COLD) | `fig5_information_hierarchy.png` |
| Figure 6 | Model rankings with 95% CI | `fig6_model_rankings.png` |

---

## 8. Data Availability

All experimental data and analysis code are available at:
https://github.com/LaxmanNandi/MCH-Experiments

---

## References

1. Laxman, M M (Nandi). (2026). Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI. *Preprints.org*. DOI: 10.20944/preprints202601.1881.v2

---

**Appendix A**: Complete per-model statistics (50 trials each)

| Model | Domain | Type | n | Mean ΔRCI | SD | 95% CI |
|-------|--------|------|---|-----------|-----|--------|
| GPT-4o | Philosophy | Closed | 50 | 0.283 | — | — |
| GPT-4o-mini | Philosophy | Closed | 50 | 0.269 | — | — |
| GPT-5.2 | Philosophy | Closed | 50 | 0.308 | — | — |
| Claude Haiku | Philosophy | Closed | 50 | 0.331 | — | — |
| Gemini Flash | Philosophy | Closed | 50 | 0.338 | — | — |
| DeepSeek V3.1 | Philosophy | Open | 50 | 0.304 | — | — |
| Kimi K2 | Philosophy | Open | 50 | 0.428 | — | — |
| Llama 4 Maverick | Philosophy | Open | 50 | 0.269 | — | — |
| Llama 4 Scout | Philosophy | Open | 50 | 0.298 | — | — |
| Ministral 14B | Philosophy | Open | 50 | 0.373 | — | — |
| Mistral Small 24B | Philosophy | Open | 50 | 0.281 | — | — |
| Qwen3 235B | Philosophy | Open | 50 | 0.322 | — | — |
| GPT-4o | Medical | Closed | 50 | 0.299 | — | — |
| GPT-4o-mini | Medical | Closed | 50 | 0.319 | — | — |
| GPT-5.2 | Medical | Closed | 50 | 0.379 | — | — |
| Claude Haiku | Medical | Closed | 50 | 0.340 | — | — |
| Claude Opus | Medical | Closed | 43 | 0.338 | — | — |
| Gemini Flash | Medical | Closed | 50 | −0.133 | — | — |
| DeepSeek V3.1 | Medical | Open | 50 | 0.320 | — | — |
| Kimi K2 | Medical | Open | 50 | 0.417 | — | — |
| Llama 4 Maverick | Medical | Open | 50 | 0.317 | — | — |
| Llama 4 Scout | Medical | Open | 50 | 0.323 | — | — |
| Ministral 14B | Medical | Open | 50 | 0.391 | — | — |
| Mistral Small 24B | Medical | Open | 50 | 0.365 | — | — |
| Qwen3 235B | Medical | Open | 50 | 0.328 | — | — |
