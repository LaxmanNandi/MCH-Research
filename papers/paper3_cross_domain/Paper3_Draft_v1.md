# Domain-Specific Temporal Dynamics of Context Sensitivity in Large Language Models

**Dr. Laxman M M, MBBS**
Government Duty Medical Officer, PHC Manchi, Karnataka, India
DNB General Medicine Resident (2026), KC General Hospital, Bangalore

---

## Abstract

How a language model uses conversational context evolves across the course of a dialogue, but whether this temporal evolution is universal or task-dependent remains unexplored. We analyze position-dependent context sensitivity (ΔRCI) across 12 large language models in two epistemologically distinct domains: philosophy (open-goal reasoning, 4 models) and medical reasoning (closed-goal diagnosis, 8 models). Each model was evaluated at 30 conversational positions under three conditions (full context, no context, scrambled context) with 50 independent trials, generating approximately 54,000 responses with complete text preserved. Two fundamental patterns emerge. In philosophy, ΔRCI follows an inverted-U trajectory under 3-bin aggregation (Early/Mid/Late), peaking at mid-conversation (0.331) and declining in late turns (0.270). In medical reasoning, ΔRCI follows a U-shaped trajectory, with a diagnostic independence trough at mid-conversation (0.311) and recovery during clinical integration (0.371). These patterns are architecture-invariant: 4/4 philosophy and 8/8 medical models conform. The sharpest domain contrast occurs at position 30 (summarization): all 8 medical models show extreme outlier Z-scores (mean +3.47 ± 0.51, all > +2.7), while no philosophy model exceeds |Z| > 1.0. This reflects task enablement — medical summarization requires case context, while philosophical summarization is feasible without it. Disruption Sensitivity analysis (12/12 models DS < 0) indicates that context presence provides more value than context ordering. These findings demonstrate that task structure — specifically whether the goal is open or closed — shapes the temporal dynamics of context sensitivity more strongly than model architecture does.

---

## 1. Introduction

Context sensitivity in language models — how much conversational history shapes responses — has been established as a measurable and variable property across architectures (Laxman, 2026a, 2026b). However, prior work measured context sensitivity as an aggregate quantity, averaging across all conversational positions. This aggregate view obscures a fundamental question: does context sensitivity evolve differently across the course of a conversation depending on the task?

This question has practical implications. If context sensitivity follows task-specific temporal patterns, then evaluation protocols that average across positions will miss critical dynamics. A model that shows adequate aggregate performance may exhibit dangerous context dependence at specific positions — or dangerous context independence at positions where the task requires it.

We address this question using a cross-domain experimental design that contrasts two epistemologically distinct tasks:

- **Philosophy** (consciousness reasoning): An open-goal task where multiple valid responses exist, context enriches but does not determine the answer, and summarization is feasible even without prior turns.
- **Medical reasoning** (STEMI diagnosis): A closed-goal task where a convergent answer structure exists, context is informative for the clinical trajectory, and summarization at the final position requires the accumulated case history.

By analyzing 12 models across 30 positions in both domains, we can distinguish effects of task structure from effects of model architecture. Our central finding is that two fundamental patterns emerge — an inverted-U in philosophy and a U-shape in medical reasoning — and that these patterns are determined by the goal structure of the task, not by the architecture of the model.

---

## 2. Methods

### 2.1 Experimental Design

We employed a three-condition, repeated-measures design identical to Papers 1–2 (Laxman, 2026a, 2026b):

- **TRUE condition**: 30 prompts delivered with the full preceding conversational history (coherent 29-message context).
- **COLD condition**: Each prompt delivered independently with no conversational history.
- **SCRAMBLED condition**: Each prompt delivered with a randomized permutation of the preceding conversational history.

Each model × domain combination was evaluated for 50 independent trials, with all three conditions evaluated per trial. Temperature was set to 0.7 and max tokens to 1024 for all models.

### 2.2 Domains

**Philosophy (consciousness)**: A 30-prompt progressive dialogue exploring consciousness — from definitions through phenomenological analysis to self-reference and meta-reflection. This represents an open-goal task: responses are valid across a range of perspectives, and no single correct answer exists.

**Medical (STEMI diagnosis)**: A 29-exchange clinical case history of an ST-elevation myocardial infarction, followed by a summarization prompt at position 30 ("Summarize this case..."). This represents a closed-goal task: the clinical trajectory is fixed, and the final summary should capture specific diagnostic and management elements.

### 2.3 Models

We analyzed 12 models with complete response text preserved (a subset of the 25 model-domain runs in Paper 2):

**Philosophy domain (4 models, closed-source):**
GPT-4o, GPT-4o-mini (OpenAI); Claude Haiku (Anthropic); Gemini Flash (Google).

**Medical domain (8 models, 7 open-source + 1 closed):**
DeepSeek V3.1 (DeepSeek, 671B MoE); Kimi K2 (Moonshot, ~1T MoE); Llama 4 Maverick (Meta, 400B MoE), Llama 4 Scout (Meta, 109B MoE); Mistral Small 24B, Ministral 14B (Mistral); Qwen3 235B (Alibaba, MoE); Gemini Flash (Google).

This yields 12 model-domain runs, 360 position-level measurements (12 × 30), and approximately 54,000 total responses (12 × 30 × 3 × 50).

### 2.4 Metrics

**ΔRCI** (context sensitivity): Computed per position as mean(RCI_TRUE) − mean(RCI_COLD), where RCI is the mean cosine similarity between a response's embedding and all other same-condition responses at that position. Embeddings were generated using all-MiniLM-L6-v2 (384-dimensional). Higher ΔRCI indicates greater context sensitivity.

**Disruption Sensitivity (DS)**: Quantifies the relative value of context presence versus context ordering:

```
DS = ΔRCI_scrambled − ΔRCI_cold
```

Negative DS indicates that scrambled context is closer to TRUE than no context — i.e., context presence provides more value than correct ordering.

**P30 Z-score**: Computed for each model as the Z-score of position 30 ΔRCI relative to positions 1–29: Z = (ΔRCI_P30 − mean_P1-29) / SD_P1-29.

**Three-bin aggregation**: Positions 1–29 were divided into Early (1–10), Mid (11–20), and Late (21–29) bins. Position 30 was excluded from bin analysis due to its unique task structure (summarization). Bin means were computed per model, then averaged within each domain.

### 2.5 Statistical Analysis

Within-domain pattern consistency was assessed by counting models conforming to the predicted pattern (inverted-U or U-shape). Between-domain P30 contrast was quantified by mean Z-scores with standard deviations. Disruption Sensitivity was summarized as the proportion of models with DS < 0 and domain-level means.

---

## 3. Results

### 3.1 Finding 1: Two fundamental temporal patterns shaped by domain

Under 3-bin aggregation (positions 1–29), the two domains exhibit distinct temporal trajectories:

**Table 1.** Cross-domain summary: Early/Mid/Late ΔRCI patterns (positions 1–29).

| Domain | N Models | Early (1–10) | Mid (11–20) | Late (21–29) | Pattern | Mean DS |
|--------|----------|-------------|-------------|--------------|---------|---------|
| Philosophy | 4 | 0.307 (0.028) | 0.331 (0.026) | 0.270 (0.046) | Inverted-U | −0.073 |
| Medical | 8 | 0.347 (0.043) | 0.311 (0.045) | 0.371 (0.056) | U-shaped | −0.115 |

*Note: Values are domain means (SD across models). Position 30 excluded from pattern analysis.*

**Philosophy (open-goal):** Across 4 models, ΔRCI peaked at mid-conversation (0.331) and declined in late turns (0.270), yielding an inverted-U pattern under 3-bin aggregation. This is consistent with recursive abstraction: context sensitivity peaks during active synthesis and declines as responses become increasingly abstract and self-referential.

**Medical (closed-goal):** Across 8 models, ΔRCI was lowest at mid-conversation (0.311, the "diagnostic independence trough") and highest in late turns (0.371), yielding a U-shaped pattern under 3-bin aggregation. This is consistent with clinical reasoning: context sensitivity drops during focused diagnostic analysis (where independent assessment is valued) and rises during clinical integration.

These patterns are visible in the individual model curves (Figure 1) and the domain grand means (Figure 2), though raw 30-position curves are oscillatory — the inverted-U and U-shape emerge clearly only under 3-bin aggregation (Figure 4).

![Figure 1: Position-dependent ΔRCI by domain](figures/fig1_position_drci_domains.png)

*Figure 1.* Position-dependent ΔRCI across 30 positions. Left: Philosophy domain (4 models). Right: Medical domain (8 models). Black lines indicate grand means. Raw curves are oscillatory; temporal patterns emerge under aggregation.

![Figure 2: Domain grand mean comparison](figures/fig2_domain_grand_mean.png)

*Figure 2.* Cross-domain comparison of grand mean ΔRCI curves (±SEM). Both domains show oscillatory position-level dynamics. The medical P30 spike is visible as the sharp uptick at position 30.

![Figure 4: Three-bin analysis](figures/fig4_three_bin_analysis.png)

*Figure 4.* Three-bin aggregation (positions 1–29) showing the inverted-U (philosophy) and U-shape (medical) patterns across all models. Position 30 excluded.

---

### 3.2 Finding 2: Task enablement creates a stark domain contrast at P30

Position 30 used a summarization prompt ("Summarize this case..."). Z-scores were computed relative to positions 1–29 within each model.

**Table 2.** Position 30 outlier analysis.

| Domain | Mean Z-score | Range | Interpretation |
|--------|-------------|-------|----------------|
| Medical (n = 8) | **+3.47 ± 0.51** | +2.77 to +4.23 | All models show extreme task enablement |
| Philosophy (n = 4) | **+0.25 ± 0.37** | −0.38 to +0.58 | No outliers; summarization feasible without context |

All 8 medical models showed P30 as a strong outlier (Z > +2.7), while no philosophy model exceeded |Z| > 1.0 (Figure 3). This domain contrast is the single strongest finding in the study.

The medical P30 spike reflects **task enablement**, not mere performance enhancement. In the COLD condition (no case history), medical models cannot execute clinical summarization — they produce refusals ("I don't have enough information to summarize a case") or generic templates. The summarization task is impossible without context. By contrast, philosophy models can generate reasonable summaries of consciousness themes even without prior turns, because the topic does not depend on accumulated case-specific information.

This distinction — between tasks where context **enhances** performance (philosophy) and tasks where context **enables** the task itself (medical) — has direct implications for deployment. Systems used in closed-goal settings must be evaluated specifically at summarization or integration positions, where context dependence is maximal.

![Figure 3: P30 Z-scores by domain](figures/fig3_zscores.png)

*Figure 3.* Position 30 Z-scores by domain. Left: All 8 medical models exceed Z = +2 (dashed line). Right: No philosophy model approaches outlier status. This stark contrast reflects the difference between task enablement (medical) and performance enhancement (philosophy).

---

### 3.3 Finding 3: Patterns are robust across model architectures

The domain-specific patterns hold across all tested architectures, regardless of vendor, parameter count, or model type.

**Table 3.** Cross-domain pattern robustness (positions 1–29, 3-bin aggregation).

| Domain | Dominant Pattern | Models Showing Pattern | Key Feature |
|--------|-----------------|----------------------|-------------|
| Philosophy | Inverted-U | 4/4 (100%) | Mid-conversation peak (0.331), late decline (0.270) |
| Medical | U-shaped | 8/8 (100%) | Diagnostic trough (0.311), integration peak (0.371) |

Both closed-source APIs (GPT-4o, Claude Haiku, Gemini Flash) and open-source models (DeepSeek, Kimi, Llama, Mistral, Qwen) exhibit domain-consistent dynamics. Dense transformers (Mistral Small 24B, Ministral 14B), mixture-of-experts architectures (DeepSeek V3.1, Qwen3 235B, Llama 4, Kimi K2), and closed-weight models (GPT-4o, Claude Haiku, Gemini Flash) all conform to their domain's pattern.

This architecture-invariance suggests that the temporal pattern is determined by the structure of the task — specifically whether the goal is open or closed — rather than by the internal architecture of the model.

---

### 3.4 Finding 4: Context presence provides more value than context ordering

Disruption Sensitivity was negative for all 12 model-domain runs, indicating that scrambled context is closer to TRUE condition responses than no context at all.

- **Philosophy:** 4/4 models had DS < 0, mean DS = −0.073 (SD = 0.037, range −0.113 to −0.039).
- **Medical:** 8/8 models had DS < 0, mean DS = −0.115 (SD = 0.026, range −0.148 to −0.065).

This result extends an aggregate-level finding from Paper 2 (Laxman, 2026b), which reported that ΔRCI_COLD > ΔRCI_SCRAMBLED in 25/25 model-domain runs — i.e., the presence of context (even disordered) always brings responses closer to the TRUE condition than its complete absence. DS < 0 is mathematically equivalent to this inequality (DS = ΔRCI_scrambled − ΔRCI_cold < 0 ⟺ ΔRCI_COLD > ΔRCI_SCRAMBLED). What Paper 3 adds is position-level granularity: the aggregate "presence > absence" finding holds at every conversational position, but its magnitude varies systematically with task structure.

Medical models show stronger negative DS (mean −0.115 vs −0.073), consistent with the clinical setting where having patient information — even disordered — is more valuable than having no information at all.

Per-position analysis reveals task-specific structure: in the medical domain, DS is most strongly negative during diagnostic reasoning positions (15–20) and approaches zero at P30 (Figure 5B). This suggests that context ordering becomes less important at the summarization position, where the model must synthesize all available information regardless of presentation order.

![Figure 5A: Disruption Sensitivity by model](figures/fig5a_disruption_sensitivity.png)

*Figure 5A.* Disruption Sensitivity by model (all 12 models). Negative values indicate context presence is more valuable than context ordering. Medical models (red) show consistently stronger negative DS than philosophy models (blue).

![Figure 5B: Per-position Disruption Sensitivity](figures/fig5b_position_disruption.png)

*Figure 5B.* Per-position Disruption Sensitivity. Top: Medical domain. Bottom: Philosophy domain. Medical DS is most strongly negative during diagnostic reasoning positions (15–20).

---

## 4. Discussion

### 4.1 Two fundamental patterns, one organizing principle

The central finding is that task goal structure shapes the temporal dynamics of context sensitivity. Open-goal tasks (philosophy) show an inverted-U pattern under 3-bin aggregation — context sensitivity peaks during active synthesis and declines as responses become increasingly abstract. Closed-goal tasks (medical) show a U-shaped pattern — context sensitivity dips during focused diagnostic reasoning and rises during clinical integration. These patterns are not architectural: 12/12 models conform to their domain's expected trajectory.

We describe these as two fundamental patterns rather than cognitive architectures, because the evidence establishes the patterns in only two domains. Whether these patterns generalize to other open-goal tasks (creative writing, philosophical debate) and closed-goal tasks (legal analysis, code review) remains an empirical question. The consistency across 12 architectures spanning 8 vendors is encouraging, but domain replication is needed before claiming generality.

### 4.2 Task enablement as a distinct mechanism

The P30 contrast (medical Z = +3.47 vs philosophy Z = +0.25) identifies task enablement as a distinct mechanism. In closed-goal tasks, specific conversational positions create categorical context dependence — the task cannot be executed without accumulated context. This is fundamentally different from performance enhancement, where context improves quality but is not required.

This distinction has deployment implications. Systems used for closed-goal tasks (clinical summarization, legal analysis, technical reporting) should be evaluated at integration and summarization positions specifically. Aggregate performance metrics will miss the extreme context dependence at these critical positions.

### 4.3 Disruption Sensitivity and system design

The finding that context presence outweighs ordering (12/12 models, DS < 0) suggests a practical heuristic for retrieval-augmented systems: prioritize information recall over perfect chronological ordering. However, the position-specific DS patterns indicate that ordering matters more at some positions than others — during diagnostic reasoning, ordering is less important than during history-taking.

### 4.4 Hypothetical: graded context dependence at summarization positions

A summarization prompt inserted at position 10 ("Summarize what we have discovered so far") did not produce a spike in the medical domain (Z = −0.59), while the position 30 summarization prompt did (Z = +2.01). This two-point observation is consistent with a graded scaling of task enablement with context volume — context dependence at summarization positions increases with the amount of prior information available. A logarithmic fit (ΔRCI ∝ log(P − 1)) connects these two points (Figure 6), but we emphasize that a fit through two data points is illustrative only. A rigorous test would require inserting identical summarization prompts at positions 5, 10, 15, 20, 25, and 30 to characterize the scaling function.

![Figure 6: Type 2 scaling](figures/fig6_type2_scaling.png)

*Figure 6.* Illustrative log fit for context dependence at summarization positions (anchored at P10 and P30 Z-scores). This is a hypothesis for future testing, not a validated scaling law.

### 4.5 Model scale effects

Among the 12 models, one showed a notable deviation: Kimi K2 (~1T parameters, MoE) exhibited a positive linear trend across positions 1–29 in the medical domain (slope = +0.006, r = 0.40, p = 0.030) and the lowest disruption sensitivity (DS = −0.027). This suggests that very large models may sustain coherence accumulation across longer conversations without the late-stage plateau seen in smaller architectures. However, this is a single-model observation and should be treated as a hypothesis for future investigation.

### 4.6 Limitations

**Domain coverage.** Only two domains are tested. The "open-goal" and "closed-goal" distinction is supported by the data but may oversimplify the space of possible task structures. Additional domains (legal, technical, creative) are needed.

**Model subset.** The 12-model subset was determined by response text availability, not by experimental design. Four of 14 Paper 2 models lack philosophy response text; all 8 medical models with response text are included. Expansion to additional models would strengthen the architecture-invariance claim.

**Three-bin aggregation.** The inverted-U and U-shape patterns emerge under 3-bin aggregation (Early/Mid/Late). Raw 30-position curves are oscillatory (Figure 1), and the bin boundaries (1–10, 11–20, 21–29) were chosen based on conversational structure rather than data-driven optimization.

**Embedding dependence.** All metrics are computed in the all-MiniLM-L6-v2 embedding space. Whether the temporal patterns are robust across different embedding models has not been tested.

**Observational design.** This study establishes that domain structure shapes temporal dynamics but does not identify the mechanism. Whether the patterns reflect attention allocation, training distribution effects, or some other process is unknown.

---

## 5. Conclusion

We identify two fundamental temporal patterns of context sensitivity in large language models, shaped by the goal structure of the task rather than by model architecture. Open-goal tasks (philosophy) produce an inverted-U pattern under 3-bin aggregation, with context sensitivity peaking at mid-conversation and declining in late turns. Closed-goal tasks (medical reasoning) produce a U-shaped pattern, with a diagnostic independence trough at mid-conversation and context sensitivity rising during clinical integration. These patterns hold across all 12 tested models spanning 8 vendors and diverse architectures.

The sharpest domain contrast occurs at position 30 (summarization): all 8 medical models show extreme outlier Z-scores (mean +3.47, all > +2.7), reflecting task enablement — the model cannot execute the task without context. No philosophy model approaches outlier status (mean Z = +0.25), because philosophical summarization is feasible without accumulated history.

Disruption Sensitivity is negative for all 12 models (mean DS: philosophy −0.073, medical −0.115), indicating that context presence provides more value than context ordering in both domains.

These findings demonstrate that temporal evaluation of context sensitivity — analyzing how it evolves across conversational positions — reveals task-specific dynamics invisible to aggregate measures. Systems deployed for closed-goal tasks should be evaluated at summarization and integration positions specifically, where context dependence is maximal.

---

## References

1. Laxman, M. M. (2026a). Context curves behavior: Measuring context sensitivity in large language models. *Preprints.org*, 2026011881. DOI: 10.20944/preprints202601.1881.v2

2. Laxman, M. M. (2026b). Scaling context sensitivity: A standardized benchmark of ΔRCI across 25 model-domain runs. *Preprints.org*, 2026021114. DOI: 10.20944/preprints202602.1114.v2

3. Laxman, M. M. (2026c). Engagement as entanglement: Variance signatures of bidirectional context coupling in large language models. In preparation.

4. Laxman, M. M. (2026d). Stochastic incompleteness in LLM summarization: A predictability taxonomy for clinical AI deployment. In preparation.

5. Laxman, M. M. (2026e). An empirical conservation constraint on context sensitivity and output variance: Evidence across LLM architectures. In preparation.

6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

---

## Figure List

### Main Figures
1. **Figure 1:** Position-dependent ΔRCI by domain (per-model curves with grand mean).
2. **Figure 2:** Cross-domain grand mean ΔRCI comparison (±SEM).
3. **Figure 3:** Position 30 Z-score analysis (medical task enablement vs philosophy baseline).
4. **Figure 4:** Three-bin aggregation showing inverted-U (philosophy) vs U-shape (medical).
5. **Figure 5A:** Disruption Sensitivity by model (all 12 models, colored by domain).
6. **Figure 5B:** Per-position Disruption Sensitivity curves.
7. **Figure 6:** Illustrative log fit for graded context dependence at summarization positions.

### Main Tables
1. **Table 1:** Cross-domain ΔRCI summary (Early/Mid/Late bins).
2. **Table 2:** Position 30 outlier analysis (medical vs philosophy Z-scores).
3. **Table 3:** Cross-domain pattern robustness across architectures.

---

## Data Availability

Complete dataset with response text preservation:
- 12 models × 30 positions × 3 conditions × 50 trials = ~54,000 responses
- Philosophy domain: 4 models (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
- Medical domain: 8 models (DeepSeek V3.1, Gemini Flash, Kimi K2, Llama 4 Maverick/Scout, Mistral Small 24B, Ministral 14B, Qwen3 235B)
- All 50-trial JSON files include complete response text for qualitative validation

Repository: https://github.com/LaxmanNandi/MCH-Research

---

**Manuscript Version:** 1.0
**Date:** February 15, 2026
**Corresponding Author:** Dr. Laxman M M
