# Paper 4: Results and Discussion
**Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling**

---

## RESULTS

### Study overview and notation
We quantify entanglement using the Delta Relational Coherence Index (DRCI), computed from within-condition response coherence (TRUE vs COLD). DRCI is interpreted here as **context-induced predictability change**. We operationalize predictability with a variance ratio measured over response embeddings, and relate DRCI to a mutual-information (MI) proxy.

Key quantities:

```
DRCI = mean(RCI_TRUE) - mean(RCI_COLD)
Var_Ratio = Var_TRUE / Var_COLD
MI_Proxy = 1 - Var_Ratio
```

Where:
- RCI_TRUE = mean self-similarity of each response to all other responses in TRUE condition
- RCI_COLD = mean self-similarity of each response to all other responses in COLD condition
- Var_TRUE and Var_COLD are the variances of response embeddings under TRUE and COLD conditions

Positive MI_Proxy indicates reduced variance (more predictability) with context.

---

### Finding 1: DRCI tracks an MI proxy (entanglement signal)
Across 8 model-domain runs (4 philosophy, 4 medical) and 30 positions, DRCI correlated strongly with the MI proxy derived from variance ratios:

- Pooled correlation: r = 0.74, p = 3.0e-42 (N = 240 model-position points)
  - Data: 8 model-domain runs × 30 positions = 240 points
  - Each point aggregates 50 independent trials per condition (4,500 total responses per model)

Interpretation: DRCI increases as context **reduces** response variance. This supports the entanglement view: context couples the response distribution to prior information, changing the predictability of outputs.

**Figure 1.** DRCI vs MI_Proxy (pooled model-position points).

---

### Finding 2: Bidirectional entanglement (convergent vs divergent)
The variance ratio reveals two regimes:

- **Convergent entanglement:** Var_Ratio < 1, DRCI > 0. Context narrows possibility space (more predictable).
- **Divergent entanglement:** Var_Ratio > 1, DRCI < 0. Context widens possibility space (less predictable).

This bidirectional framing resolves the SOVEREIGN category: **SOVEREIGN = divergent entanglement** (context destabilizes predictability).

**Figure 2.** Regime map of DRCI vs Var_Ratio.

---

### Finding 3: Llama safety anomaly at medical summarization (Type 2)
At medical position 30 (summarization), we observe extreme divergence in two Llama models:

- Llama 4 Maverick: Var_Ratio = 2.64, DRCI = -0.15
- Llama 4 Scout: Var_Ratio = 7.46, DRCI = -0.22

In contrast, convergent models at P30 show Var_Ratio < 1 and positive DRCI (e.g., Gemini Flash, DeepSeek V3.1). This identifies a **safety risk class**: models that diverge under Type 2 prompts produce highly unstable, unpredictable outputs precisely when task enablement is expected.

For audit detail (ESI calculation and “intact capability + broken coherence” framing), see `docs/Llama_Safety_Anomaly.md`.

**Figure 3.** Var_Ratio at medical P30 across models (convergent vs divergent classes).

---

### Finding 4: Domain architecture differences in variance response
Mean variance ratios differ by domain:

- Philosophy: Var_Ratio ~ 1.01 (variance-neutral)
- Medical: Var_Ratio ~ 1.30 (variance-increasing)

This suggests domain-specific architecture effects: medical prompts tend to **destabilize** response distributions under context, while philosophy is largely variance-neutral on average.

**Figure 4.** Domain-level Var_Ratio comparison.

---

### Finding 5: Variance sufficiency (simple surrogate works)
The variance ratio provides a practical, low-cost surrogate for entanglement. DRCI tracks MI_Proxy without requiring k-NN entropy estimation or full mutual information computation. This makes entanglement measurement accessible at scale.

---

## DISCUSSION

### Entanglement reframes DRCI as predictability modulation
The central shift is conceptual: DRCI is not just a measure of helpfulness, but a **predictability modulation** measure. Context changes the shape of the response distribution; DRCI quantifies that change. This reframing clarifies why negative values are not inherently bad: they indicate divergence (useful for creativity, risky for safety-critical domains).

### Bidirectional entanglement fills the SOVEREIGN category
SOVEREIGN behavior is now grounded in mechanism: contexts that **increase** variance produce negative DRCI. This is not a failure of context usage but a distinct mode of coupling. It suggests that SOVEREIGN is a valid, measurable regime rather than a missing category.

### Safety implications: predictability is task-dependent
The Llama divergence at medical summarization highlights a concrete risk: models can become less predictable exactly when a task presupposes context. For safety-critical tasks, **predictability is a requirement**, not an optional characteristic. Divergent entanglement in these settings should be treated as a deployment risk.

We propose (as provisional guidance) that **Var_Ratio > 3** or **ESI < 1** be treated as a red flag for critical tasks. These thresholds are not standards yet; they are empirically motivated guardrails for further validation.

### Architectural interpretation and open questions
The emergence of convergent vs divergent classes suggests architectural differences in how models handle context saturation. The observation is consistent across multiple models but remains mechanistic rather than causal. Future work should test whether divergence correlates with training objectives, attention patterns, or safety alignment strategies.

### Limitations
1. **MI proxy definition:** We use variance ratio as a proxy; full mutual information estimation may reveal non-linear effects.
2. **Pooling effects:** The r = 0.74 correlation is pooled across models and positions; per-model correlations may vary in magnitude and sign depending on proxy definition.
3. **Domain scope:** Two domains only; additional domains are required to assess generality.
4. **Type 2 coverage:** The strongest safety signal is at P30; more positions are needed to validate a general rule for Type 2 divergence.

### Future work
- Run the planned Type 2 scaling experiment at P5-P30 and compute Var_Ratio/ESI at each position.
- Expand to third domains (legal, technical, creative) to test domain architecture claims.
- Compare embedding models to ensure variance effects are robust.
- Examine attention patterns in open models to link divergence to mechanism.

---

## FIGURE LIST (Paper 4)
1. DRCI vs MI_Proxy (pooled model-position points).
2. Regime map: convergent vs divergent entanglement (DRCI vs Var_Ratio).
3. Llama safety anomaly at medical P30 (Var_Ratio by model).
4. Domain architecture comparison (Var_Ratio by domain).

---

## DATA AVAILABILITY
All raw data and analysis scripts are available in the project repository:

```
https://github.com/LaxmanNandi/MCH-Experiments
```
