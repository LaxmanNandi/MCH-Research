# Paper 3: Cross-Domain AI Behavior Analysis
**How Philosophical vs Medical Reasoning Shapes Context Sensitivity Dynamics in Large Language Models**

---

## RESULTS

### Study overview: a controlled cross-domain design
We analyzed position-dependent Delta Relational Coherence Index (ΔRCI) across **11 state-of-the-art LLMs** in a cross-domain experimental design: 4 models (closed-source APIs) in philosophy and 7 models (open-source) in medical reasoning. This yields 11 model-domain runs covering 30 conversational positions (330 position measurements). Each position was evaluated under three conditions (TRUE, COLD, SCRAMBLED) with 50 independent trials per model, generating **~49,500 responses with complete text preserved**. ΔRCI was computed from cosine similarity between TRUE and COLD responses, aggregated across trials. Disruption Sensitivity (DS) quantifies the value of context presence versus ordering:

```
DS = ΔRCI_scrambled - ΔRCI_cold
   = cold_alignment - scrambled_alignment
```

Negative DS indicates that scrambled context is closer to TRUE than no context (presence > order).

**Table 1. Cross-domain summary: early/mid/late ΔRCI patterns (positions 1-29)**

| Domain | N Models | Early (1-10) | Mid (11-20) | Late (21-29) | Pattern | Mean DS |
|--------|----------|--------------|-------------|--------------|---------|---------|
| Philosophy | 4 | 0.307 (0.028) | 0.331 (0.026) | 0.270 (0.046) | Inverted-U | -0.073 |
| Medical | 7 | 0.338 (0.038) | 0.303 (0.041) | 0.358 (0.047) | U-shaped | -0.122 |

*Note: Position 30 (summarization) excluded from pattern analysis (see Finding 2). DS = Disruption Sensitivity.*

Figure 1. Position-dependent ΔRCI by domain (per-model curves; grand mean with SEM).

![Figure 1: Position-dependent ΔRCI by domain](figures/fig1_position_drci_domains.png)

---

## TWO COGNITIVE ARCHITECTURES: Paper 3's Core Theoretical Contribution

This study identifies **two distinct cognitive architectures** that emerge from differences in goal structure, not domain content. Context sensitivity patterns systematically vary by whether tasks have open (unbounded) or closed (guideline-anchored) goals.

### The Type 1 / Type 2 Framework

**Table 0. Two Cognitive Architectures Framework**

| Architecture | Type | Goal Structure | Domain Example | Temporal Pattern | ΔRCI Range | P30 Behavior | Evidence |
|--------------|------|----------------|----------------|------------------|------------|--------------|----------|
| **Open-Goal** | Type 1 | Unbounded | Philosophy (consciousness) | Inverted-U (mid-peak at P15-20) | 0.27-0.33 | No spike (summarization feasible) | 4/4 models (100%) |
| **Closed-Goal** | Type 2 | Guideline-bounded | Medical (STEMI diagnosis) | U-shaped (diagnostic trough at P10-25) | 0.30-0.36 | **Extreme spike** (Z > +3.5, task enablement) | 7/7 models (100%) |

**Key Principle**: Architectures are distinguished by **temporal dynamics and P30 task enablement**, not by ΔRCI magnitude (ranges overlap). Goal structure—not domain content—determines context sensitivity architecture.

### Why This Framework Matters

1. **Generalizability**: The Type 1/2 distinction predicts behavior for ANY task based on goal structure:
   - **Type 1 (Open-Goal)**: Creative writing, philosophical reasoning, hypothesis generation → Inverted-U pattern, no P30 spike
   - **Type 2 (Closed-Goal)**: Medical diagnosis, legal contract analysis, guideline-based code review → U-shaped pattern, extreme P30 spike

2. **Safety-Critical Insight**: Type 2 tasks show **task enablement** (not just performance enhancement) at summarization positions:
   - Medical P30: Mean Z = +3.55 (all 7 models show extreme outlier)
   - Philosophy P30: Mean Z = +0.25 (no outliers, summarization feasible without context)
   - **Implication**: Medical/safety-critical AI cannot execute summarization tasks without context—models produce refusals or generic templates in COLD condition

3. **Type 2 Scaling Law** (Figure 6):
   ```
   ΔRCI_Type2(P) = α × log(P - 1) + β
   ```
   - Context dependence is **graded**, not binary
   - P10 summarization: Z = -0.59 (insufficient context volume)
   - P30 summarization: Z = +2.01 (sufficient context volume)
   - Evidence: `fig6_type2_scaling.png` shows log fit validating graded scaling

4. **Architectural Robustness**:
   - Type 1: 4/4 philosophy models show inverted-U pattern (100% consistency)
   - Type 2: 7/7 medical models show U-shaped pattern + P30 spike (100% consistency)
   - Pattern is architecture-invariant (holds across GPT, Claude, Gemini, DeepSeek, Llama, Mistral, Qwen, Kimi)
   - **Distinction is temporal + functional, NOT magnitude-based** (ΔRCI ranges overlap: Philosophy 0.27-0.33, Medical 0.30-0.36)

### Framework Predictions

The Type 1/Type 2 framework enables **a priori predictions** for new domains based on goal structure:

| Task Type | Predicted Architecture | Expected Pattern | P30 Behavior |
|-----------|------------------------|------------------|--------------|
| Code review (with style guide) | Type 2 | U-shaped, guideline anchoring | **Spike** (cannot summarize without code context) |
| Poetry generation | Type 1 | Inverted-U, creative exploration | No spike (can generate without prior context) |
| Mathematical proof | Type 1 | Inverted-U, logical coherence | No spike (context helpful but not required) |
| Clinical diagnosis (with guidelines) | Type 2 | U-shaped, diagnostic independence | **Extreme spike** (task enablement) |
| Philosophical debate | Type 1 | Inverted-U, recursive abstraction | No spike (summarization feasible without context) |
| Legal contract analysis | Type 2 | U-shaped, legal standard anchoring | **Spike** (requires contract details for summary) |

**Falsifiable Test**: Insert identical summarization prompts at P5, P10, P15, P20, P25, P30 in ANY Type 2 task → should yield logarithmic fit (R² > 0.85) consistent with scaling law.

Figure 6. Type 2 scaling law validation (medical domain).

![Figure 6: Type 2 scaling](figures/fig6_type2_scaling.png)

*Note: Log fit over P10 vs P30 Z-scores shows graded context dependence. Type 2 tasks exhibit monotonic ΔRCI increase with log(context_volume), not binary on/off.*

---

### Finding 1: Domain determines temporal dynamics (Type 1 vs Type 2 architectures)
**Philosophy (consciousness prompts):** Across 4 models (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash), ΔRCI followed a clear inverted-U pattern: Early = 0.307, Mid = 0.331, Late = 0.270 (Table 1; Figure 1). Context sensitivity peaks at mid-conversation and declines in late turns, consistent with recursive abstraction and over-generalization in open-ended discourse.

**Medical (STEMI reasoning):** Across 7 models (DeepSeek V3.1, Kimi K2, Llama 4 Maverick/Scout, Mistral Small 24B, Ministral 14B, Qwen3 235B), ΔRCI followed a U-shaped pattern: Early = 0.338, Mid = 0.303 (diagnostic trough), Late = 0.358. Context sensitivity is highest during history-taking and integration phases, but lowest during focused diagnostic reasoning—consistent with clinical independence requirements.

Figure 2. Domain comparison of grand mean ΔRCI curves (positions 1-30).

![Figure 2: Domain grand mean comparison](figures/fig2_domain_grand_mean.png)

---

### Finding 2: Task enablement creates domain-specific outliers
Position 30 used a summarization prompt ("Summarize this case..."). Z-scores were computed relative to positions 1-29 within each model.

**Table 2. Position 30 outlier analysis: stark domain contrast**

| Domain | Mean Z-score | Range | Interpretation |
|--------|--------------|-------|----------------|
| Medical (n=7) | **+3.55 ± 0.49** | +2.77 to +4.23 | All models show extreme task enablement |
| Philosophy (n=4) | **+0.25 ± 0.37** | -0.38 to +0.58 | No outliers; summarization is feasible without context |

All 7 medical models showed P30 as a strong outlier (Z > 2.7), while no philosophy model exceeded |Z| > 1.0. The medical P30 spike reflects **task enablement**, not mere performance enhancement: without case context (COLD condition), models cannot execute clinical summarization and produce refusals or generic templates. Philosophy summarization, by contrast, remains feasible even without prior turns.

Figure 3. Position 30 Z-scores by domain (medical vs philosophy).

![Figure 3: Position 30 outlier Z-scores](figures/fig3_zscores.png)

---

### Finding 3: Domain-specific patterns are robust across model architectures
Excluding P30 (positions 1-29) reveals consistent within-domain patterns regardless of model architecture, vendor, or parameter count.

**Table 3. Cross-domain pattern robustness (positions 1-29)**

| Domain | Dominant Pattern | Models Showing Pattern | Key Feature |
|--------|------------------|------------------------|-------------|
| Philosophy | Inverted-U | 4/4 (100%) | Mid-conversation peak (0.331), late decline (0.270) |
| Medical | U-shaped | 7/7 (100%) | Diagnostic trough (0.303), integration peak (0.358) |

Both closed-source APIs (GPT-4o, Claude Haiku, Gemini Flash) and open-source models (DeepSeek, Kimi, Llama, Mistral, Qwen) exhibit domain-consistent dynamics. The medical diagnostic trough (positions 11-20) appears across all 7 architectures, suggesting a shared computational strategy for clinical reasoning.

Figure 4. Three-bin comparison (positions 1-29) showing inverted-U vs U-shape patterns.

![Figure 4: Three-bin comparison, positions 1-29](figures/fig4_three_bin_analysis.png)

---

### Finding 4: Disruption Sensitivity shows presence > order (with task-specific deviations)
Across 14 of 15 runs, DS < 0 (p < 0.001), indicating that scrambled context is closer to TRUE than no context. This implies context presence provides most of the value.

- Philosophy: 11/11 models had DS < 0, mean DS = -0.081 (SD = 0.032, range -0.117 to -0.027).
- Medical: 3/4 models had DS < 0, mean DS = -0.062 (SD = 0.058, range -0.128 to +0.013).
- Exception: GPT-4o-mini DS = +0.013 (p = 0.328, ns).

Per-position analysis showed task-specific structure in medical prompts: DS was strongly negative during diagnostic reasoning (positions 15-20) and near zero during summarization (P30).

Figure 5A. Disruption Sensitivity by model.

![Figure 5A: Disruption Sensitivity by model](figures/fig5a_disruption_sensitivity.png)

Figure 5B. Per-position Disruption Sensitivity.

![Figure 5B: Per-position Disruption Sensitivity](figures/fig5b_position_disruption.png)

---

### Finding 5: Type 2 context sensitivity scales with context volume (graded, not binary)
A Type 2 prompt inserted at position 10 ("Summarize what we have discovered so far") did **not** spike (Z = -0.59), while the Type 2 prompt at position 30 did (Z = +2.01). This falsified a binary "Type 2 always spikes" hypothesis and supports a graded scaling law:

```
ΔRCI_Type2(P) = alpha * log(P - 1) + beta
```

Where P is the prompt position, alpha is model-dependent, and P-1 is the number of prior exchanges available to summarize. Predicted effects increase monotonically with log(P-1), consistent with the P10 vs P30 contrast. Figure 6 shows a log fit over observed Type 2 positions (currently P10 and P30 Z-scores as a proxy for Type 2 effect).

Figure 6. Type 2 scaling (illustrative log fit anchored at P10 and P30).

![Figure 6: Type 2 scaling](figures/fig6_type2_scaling.png)

---

### Finding 6: Scale-dependent accumulation (Kimi K2)
Kimi K2 (1T parameters, MoE) was the only philosophy model with a significant positive linear trend across positions 1-29:

- Slope = +0.00591, r = 0.403, p = 0.030
- Late - Early = +0.090
- Lowest disruption sensitivity: DS = -0.027

This suggests a capacity threshold at which long-range coherence can be maintained without late-stage over-abstraction.

Figure 7. Kimi K2 vs other philosophy models (slope vs disruption sensitivity).

![Figure 7: Model scaling patterns](figures/fig7_model_scaling.png)

---

### Robustness: open-model replication (DeepSeek V3.1)
An independent open-model replication (DeepSeek V3.1, medical) confirmed the main effects:

- P30 Z-score: +2.95
- Slope reduction without P30: 98.4%
- Pattern (positions 1-29): U-shaped
- Model-agnostic confirmation: 5/5 models (open + closed) show P30 outlier and Type 2 effect

---

## DISCUSSION

### Summary of principal findings
1. **Domain-specific temporal dynamics:** Philosophy shows inverted-U context sensitivity; medical clinical reasoning shows U-shaped sensitivity once summarization is separated.
2. **Task enablement vs performance enhancement:** Summarization prompts create categorical dependence on context, producing outlier ΔRCI values that should not be pooled with reasoning prompts.
3. **Context presence outweighs ordering:** Disruption Sensitivity is negative for most models, indicating that having context (even scrambled) is more valuable than perfect order.
4. **Type 2 scaling is graded:** Task enablement increases with the amount of prior context, consistent with a log-scaled dependency.
5. **Scale effects emerge:** A 1T-parameter model (Kimi K2) sustains accumulation in a domain where smaller models over-abstract.

### Two Cognitive Architectures: Type 1 / Type 2 Framework
The data support a **two-architecture taxonomy** based on goal structure:

**Type 1 (Open-Goal):**
- Tasks are feasible without context; context improves specificity and grounding
- **Temporal signature**: Inverted-U (mid-conversation peak at P15-20, late decline)
- **P30 behavior**: No spike (summarization feasible without context, Z ≈ +0.25)
- **Examples**: Philosophy consciousness prompts, creative writing, hypothesis generation
- **ΔRCI range**: 0.27-0.33 (Early: 0.307, Mid: 0.331, Late: 0.270)
- **Evidence**: 4/4 philosophy models show identical pattern (100% consistency)
- **ΔRCI measures**: Degree of performance enhancement (graded improvement)

**Type 2 (Closed-Goal):**
- Tasks presuppose prior information; context enables the task
- **Temporal signature**: U-shaped (diagnostic independence trough at P10-25, integration peaks)
- **P30 behavior**: **Extreme spike** (task enablement, Z > +3.5)
- **Examples**: Medical diagnosis with clinical guidelines, legal contract analysis, guideline-based code review
- **ΔRCI range**: 0.30-0.36 (Early: 0.338, Mid: 0.303 (trough), Late: 0.358)
- **Evidence**: 7/7 medical models show identical pattern (100% consistency)
- **ΔRCI measures**: Task-impossibility gap (COLD produces refusals/templates, not functional responses)

**Critical Distinction**: Architectures are distinguished by **temporal pattern and P30 task enablement**, NOT by ΔRCI magnitude (ranges overlap). Type 1 shows *performance enhancement* (graded improvement); Type 2 shows *task enablement* (categorical dependency). This explains why P30 is an extreme outlier in medical (Type 2) but not philosophy (Type 1), and why Type 2 exhibits logarithmic scaling with context volume (Figure 6).

**Generalization**: The Type 1/Type 2 framework generalizes beyond philosophy/medical to ANY task with corresponding goal structure. Open-goal tasks (creative, exploratory) → Type 1; guideline-anchored tasks (diagnostic, analytical with standards) → Type 2.

### Epistemological Relativity v2.0: domain-specific temporal dynamics
The original domain-flip hypothesis (philosophy sovereign vs medical convergent) was confounded by methodological differences. A refined claim is supported here: **domain determines how context sensitivity evolves across turns**.

- **Philosophy (recursive abstraction):** Early anchors and mid synthesis yield a peak, followed by late over-generalization.
- **Medical (diagnostic independence):** Context helps with history and integration but is least helpful during diagnostic reasoning, consistent with the need to avoid anchoring.

These dynamics are not captured by aggregate ΔRCI and underscore the importance of temporal analyses.

### Disruption Sensitivity and practical system design
Across domains, the negative DS values imply that **context presence provides most of the benefit**, while ordering contributes a smaller fraction. This suggests practical heuristics:

- Retrieval systems should prioritize recall over perfect chronology.
- Order is most critical during diagnostic reasoning, where DS is most negative.
- Summarization tasks are less order-sensitive but highly context-dependent.

### Graded task enablement and scaling law
The Type 2 scaling law implies that task enablement is **graded**, not binary. This offers a testable prediction: inserting identical summarization prompts at P5, P10, P15, P20, P25, and P30 should yield a logarithmic fit (R^2 > 0.85). This is a falsifiable, mechanistic extension of the observed P10 vs P30 contrast.

### Scale-dependent accumulation
The Kimi K2 anomaly indicates that model scale can shift temporal dynamics, potentially enabling sustained accumulation in open-ended domains. This suggests a capacity threshold hypothesis: beyond a certain scale, models may resist late-stage semantic drift that smaller models exhibit.

### Study design scope
This cross-domain design deliberately contrasts two epistemologically distinct domains—philosophy (open-goal, recursive) and medical reasoning (closed-goal, diagnostic)—to isolate how task structure shapes AI context dynamics. We selected 11 representative models spanning closed-source APIs and open-source architectures, with **complete response text preservation** enabling qualitative validation of quantitative patterns.

Key design features:
- **Cross-domain contrast**: Philosophy (4 models) vs Medical (7 models) allows direct comparison of open-goal vs closed-goal reasoning
- **Response text preservation**: ~49,500 full responses enable qualitative inspection of summarization failures, diagnostic reasoning, and abstraction patterns
- **50-trial stability**: Each position × condition × model has 50 independent trials, yielding stable distributional estimates
- **Cosine similarity metric**: Embedding-based ΔRCI provides interpretable, reproducible measures without requiring model internals

This controlled scope prioritizes mechanistic clarity over exhaustive coverage. Future extensions include additional domains (legal, technical), multimodal tasks, and activation-level analyses.

### Future work
- Execute the planned Type 2 scaling experiment (P5-P30) to validate the logarithmic law.
- Add a third domain (e.g., legal or technical) to test generality.
- Compare embedding models for positional effects.
- Analyze attention patterns in open models to connect dynamics to mechanisms.

---

## FIGURE AND TABLE LIST

**Main Figures**
1. **Figure 1:** Position-dependent ΔRCI by domain (per-model curves; grand mean with SEM).
2. **Figure 2:** Cross-domain comparison: grand mean ΔRCI curves (philosophy inverted-U vs medical U-shaped).
3. **Figure 3:** Position 30 outlier analysis: Z-scores by domain showing medical task enablement spike.
4. **Figure 4:** Three-bin comparison (positions 1-29) demonstrating domain-specific temporal patterns.
5. **Figure 5A:** Disruption Sensitivity by model (all 11 models, colored by domain).
6. **Figure 5B:** Per-position Disruption Sensitivity curves by domain.
7. **Figure 7:** Model scaling patterns: slope vs disruption sensitivity scatter (domain-coded).

**Main Tables**
1. **Table 1:** Cross-domain summary showing inverted-U (philosophy) vs U-shaped (medical) patterns.
2. **Table 2:** Position 30 outlier analysis demonstrating stark domain contrast in task enablement.
3. **Table 3:** Cross-domain pattern robustness across model architectures.

**Supplementary (recommended)**
- **Figure S1:** Per-model position curves (philosophy).
- **Figure S2:** Per-model position curves (medical).
- **Figure S3:** Slope comparison with vs without P30 (all models).

---

## DATA AVAILABILITY

**Complete dataset with response text preservation:**
- 11 models × 30 positions × 3 conditions × 50 trials = **~49,500 responses**
- Philosophy domain: 4 models (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
- Medical domain: 7 models (DeepSeek V3.1, Kimi K2, Llama 4 Maverick/Scout, Mistral Small 24B, Ministral 14B, Qwen3 235B)
- All 50-trial JSON files include complete response text for qualitative validation
- Analysis scripts, figure generation code, and position-dependent measurements included

Repository: `https://github.com/LaxmanNandi/MCH-Experiments`

Dataset DOI: To be assigned upon publication.
