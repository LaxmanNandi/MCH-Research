# Paper 7: The Anatomy of Context — Depth, Structure, and Trajectory of Context Sensitivity in Large Language Models

## Authors
Dr. Laxman M M, MBBS

## Abstract (Draft)

Papers 1-6 of the MCH program established that context sensitivity (ΔRCI) is universal, domain-shaped, and conserved. This paper asks the next question: **what is context made of?** We decompose context sensitivity along three dimensions: (1) **depth** — how many prior messages a model actually utilizes (Context Utilization Depth, CUD); (2) **structure** — whether content or sequential order drives the effect (Content-Order Decomposition); and (3) **trajectory** — whether response diversity expands or contracts across the conversation (Exploration Arc). Using existing three-condition data (TRUE, SCRAMBLED, COLD) from 14 LLMs across medical and philosophy domains, plus new CUD pilot data from 4 models, we find that: (a) most models are recency-dominant (CUD ≈ 1), with Llama 4 Maverick as a notable exception showing genuine depth utilization; (b) content accounts for ~60% of context sensitivity in closed-goal (medical) domains while order accounts for ~60% in open-goal (philosophy) domains (Mann-Whitney p = 0.000239, Cohen's d = 1.85); (c) content and order play **opposite roles in variance** — content tokens amplify response variance by 4-6× while correct ordering suppresses it to ~30%, with the two forces approximately cancelling (Var_Ratio ≈ 1); (d) the conservation product K = ΔRCI × Var_Ratio decomposes into content and order components (K_content and K_order), both significantly domain-dependent; and (e) Exploration Arc patterns are domain-dependent, with philosophy models uniformly expanding diversity while medical models remain stable. These dimensions provide a complete structural account of how LLMs process conversational context, with direct implications for RAG system design, prompt engineering, and clinical AI deployment.

---

## 1. Introduction

### 1.1 The Missing Question

The MCH research program (Papers 1-6) has established several invariant properties of LLM context sensitivity:

- **ΔRCI** measures how much context changes model responses (Paper 2)
- **Temporal dynamics** show domain-specific patterns across conversation positions (Paper 3)
- **Variance signatures** reveal bidirectional context coupling (Paper 4, VRI)
- **Predictability taxonomy** classifies model-domain combinations into four deployment classes (Paper 5)
- **Conservation constraint** shows ΔRCI × Var_Ratio ≈ K(domain) (Paper 6)

These findings tell us that context *matters*, *when* it matters, and *how much* it matters. But they don't tell us **what context is made of**. This paper decomposes context sensitivity into three orthogonal dimensions:

1. **Depth**: How far back into conversation history does the model actually reach?
2. **Structure**: Is it the content (facts) or the order (sequence) that drives context sensitivity?
3. **Trajectory**: Does the model's response diversity expand or contract as context accumulates?

### 1.2 The Three-Condition Design as Decomposition Tool

The MCH protocol's three conditions — TRUE (coherent history), SCRAMBLED (randomized order, same content), and COLD (no history) — were originally designed to establish the information hierarchy (TRUE > SCRAMBLED > COLD). We show here that these same three conditions enable a complete structural decomposition of context:

- **TRUE - COLD** = total context effect
- **SCRAMBLED - COLD** = content effect (what the facts contribute, regardless of order)
- **TRUE - SCRAMBLED** = order effect (what sequential structure contributes beyond content)
- **Content Fraction** = (SCRAMBLED - COLD) / (TRUE - COLD)

This decomposition requires no new experiments — it is latent in every MCH dataset.

---

## 2. Methods

### 2.1 Data Sources

| Source | Models | Domains | Trials | Conditions |
|--------|--------|---------|--------|------------|
| Papers 2-6 (existing) | 14 LLMs | Medical (P30), Philosophy (P15) | 50 per config | TRUE, SCRAMBLED, COLD |
| CUD Pilot (new) | 5 LLMs | Medical (P30), Philosophy (P15) | 50 per config | TRUE, COLD, TRUNCATED(K) |
| Paper 6 Legal (new) | 7 LLMs (in progress) | Legal (P30) | 50 per config | TRUE, SCRAMBLED, COLD |

### 2.2 Metrics

#### 2.2.1 Context Utilization Depth (CUD)

**CUD** = minimum K where ΔRCI_TRUNCATED(K) >= 0.90 × ΔRCI_TRUE

Where TRUNCATED(K) provides only the last K message pairs as context. K-values tested:
- Medical P30: K = {1, 5, 10, 15, 20, 29}
- Philosophy P15: K = {1, 3, 5, 7, 10, 14}

The **K-curve** plots ΔRCI_TRUNCATED(K) / ΔRCI_TRUE at each K value.

#### 2.2.2 Content-Order Decomposition

At each position P (or at the final position for summary statistics):

- **Content Effect** = RCI_SCRAMBLED(P) - RCI_COLD(P)
- **Order Effect** = RCI_TRUE(P) - RCI_SCRAMBLED(P)
- **Content Fraction** = Content Effect / (Content Effect + Order Effect) × 100%

#### 2.2.3 Variance Decomposition

The Content-Order decomposition extends to response variance using embedding dimension variance (Paper 6 method):

- **VR_Content** = Var(SCRAMBLED) / Var(COLD) — content-only variance effect
- **VR_Order** = Var(TRUE) / Var(SCRAMBLED) — order-only variance effect
- **VR_Total** = Var(TRUE) / Var(COLD) ≈ VR_Content × VR_Order (multiplicative decomposition)

Where variance is computed as mean variance across 384 embedding dimensions at each position, across 50 trials, then averaged across positions.

#### 2.2.4 Conservation Product Decomposition

The conservation product K = ΔRCI × Var_Ratio (Paper 6) decomposes into:

- **K_content** = ΔRCI × VR_Content
- **K_order** = ΔRCI × VR_Order

#### 2.2.5 Exploration Arc

- **Diversity_Early** = mean pairwise response diversity at positions P1-P5
- **Diversity_Late** = mean pairwise response diversity at final 5 positions
- **Exploration Arc** = Diversity_Late / Diversity_Early

Arc > 1: responses diversify over time (exploration). Arc < 1: responses converge (exploitation).

### 2.3 Models

**CUD Pilot Models (4):**
| Model | Vendor | Architecture |
|-------|--------|-------------|
| DeepSeek V3.1 | DeepSeek | 685B MoE |
| Gemini 2.5 Flash | Google | Undisclosed |
| Llama 4 Maverick | Meta | 17B-128E MoE |
| Qwen3 235B | Alibaba | 235B MoE |

**Content-Order & Exploration Arc (13-14 models):**
All Paper 2 models with SCRAMBLED data: Claude Haiku, Gemini Flash, GPT-4o, GPT-4o-mini, GPT-5.2, DeepSeek V3.1, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B (+ legal domain models when complete)

### 2.4 Statistical Tests

- Mann-Whitney U for domain comparisons (non-parametric)
- Cohen's d for effect sizes
- Pearson/Spearman correlations for CUD-Var_Ratio relationships

---

## 3. Results

### 3.1 Context Utilization Depth (CUD)

#### 3.1.1 CUD Classes

| Model | Medical CUD | K-Curve Shape | Classification |
|-------|-------------|---------------|----------------|
| DeepSeek V3.1 | 1 | Flat at ~100% | Recency-dominant |
| Gemini Flash | 1 | Flat at ~100% | Recency-dominant |
| Qwen3 235B | 1 | Flat at ~88-96% | Recency-dominant (slight gradient) |
| Llama 4 Maverick | >1 | Rising: 73% → 89% | Genuine depth utilization |

#### 3.1.2 Domain Contrast

- Medical: CUD is meaningful (ΔRCI ~0.82-0.95)
- Philosophy: CUD is undefined (ΔRCI ~0.05-0.22, K-curves are noise)

CUD is only interpretable where ΔRCI is substantial. This is consistent with Papers 2-6 domain contrast.

#### 3.1.3 Sensitivity vs Accuracy Dissociation

Two metrics reveal different aspects of context depth:
- **Sensitivity** (ΔRCI_TRUNCATED): How different from COLD? Saturates at K=1 for most models.
- **Accuracy** (sim_trunc_true): How close to TRUE? Requires more context.

Models can appear context-sensitive with K=1 while still not reproducing the full-context response.

### 3.2 Content-Order Decomposition

#### 3.2.1 Domain-Dependent Decomposition

**Medical (closed-goal): Content dominates**
- Mean Content Fraction: 59.8% ± 9.0%
- N = 12 model-domain configurations
- Content > Order in majority of models

**Philosophy (open-goal): Order dominates**
- Mean Content Fraction: 39.8% ± 12.3%
- N = 12 model-domain configurations
- Order > Content in majority of models

**Statistical significance:**
- Mann-Whitney U = 133.0, p = 0.000239
- Cohen's d = 1.85 (large effect)

#### 3.2.2 Interpretation

- **Medical (closed-goal)**: Facts matter most. Symptoms, vitals, history — even scrambled, the model can integrate toward a diagnosis. Content is approximately 60% of the signal.
- **Philosophy (open-goal)**: Argument structure matters most. Logical flow, premise-conclusion chains — scrambling destroys what matters. Order is approximately 60% of the signal.

#### 3.2.3 P30 Spike Decomposition

The documented P30 spike in medical ΔRCI (Paper 3) is present in both TRUE-COLD and SCRAMBLED-COLD:

| Condition | P30 z-score range | Interpretation |
|-----------|-------------------|----------------|
| TRUE - COLD | +2.00 to +4.17 | Full context spike (documented) |
| SCRAMBLED - COLD | +0.97 to +4.75 | Content-only spike (equally strong) |

The P30 spike is driven by **content accumulation**, not sequential reasoning. At P30, the model has seen all 29 messages — even in scrambled order, content saturation produces the spike.

#### 3.2.4 Formal Statement

Context Sensitivity decomposes as:

```
ΔRCI(TRUE-COLD) = ΔRCI(Content) + ΔRCI(Order)

Where:
  ΔRCI(Content) = RCI(SCRAMBLED) - RCI(COLD)     [content contribution]
  ΔRCI(Order)   = RCI(TRUE) - RCI(SCRAMBLED)      [order contribution]

  Content Fraction = f(domain_goal_structure)

  Closed-goal domains: Content Fraction ≈ 60%
  Open-goal domains:   Content Fraction ≈ 40%
```

### 3.3 Variance Decomposition: Content Destabilizes, Order Stabilizes

#### 3.3.1 Core Finding

Content and order play **opposite roles** in response variance:

| Component | Medical (N=8) | Philosophy (N=7) | Meaning |
|-----------|---------------|-------------------|---------|
| VR_Content (SCRAM/COLD) | 4.22 ± 1.20 | 5.69 ± 2.73 | Content amplifies variance 4-6× |
| VR_Order (TRUE/SCRAM) | 0.32 ± 0.07 | 0.32 ± 0.08 | Order suppresses variance to ~30% |
| VR_Total (TRUE/COLD) | 1.20 ± 0.20 | 1.05 ± 0.12 | Net effect: approximate cancellation |

- VR_Content: Mann-Whitney U=21, p=0.463 (no domain difference)
- VR_Order: Mann-Whitney U=30, p=0.867 (no domain difference)

#### 3.3.2 Interpretation

- **Content alone (SCRAMBLED) injects noise**: Giving a model content tokens without order makes responses 4-6× more variable than no context. The model has information to work with but interprets it differently each trial — like a doctor reading scattered lab reports.
- **Order suppresses noise**: Putting those tokens in correct sequence reduces variance to ~30% of scrambled. The model locks in on the logical flow — like a doctor reading a properly organized chart.
- **The two forces cancel**: VR_Total ≈ 1 because content-driven amplification and order-driven suppression approximately balance.

#### 3.3.3 Domain Invariance

Unlike ΔRCI Content Fraction (which differs by domain, p=0.0002), the variance decomposition shows **no domain difference** in either VR_Content (p=0.46) or VR_Order (p=0.87). Domains shape *how much* context changes responses (sensitivity), not *how stable* those changes are (variance).

#### 3.3.4 Multiplicative Decomposition

VR_Content × VR_Order ≈ VR_Total holds well for Medical (Product/Total ratio: 0.99-1.15) but breaks down in Philosophy (up to 2.79 for GPT-4o), suggesting cross-terms or non-linear interactions in open-goal domains.

#### 3.3.5 Llama P30 Anomaly Decomposed

The documented Llama P30 Var_Ratio anomaly (Paper 4) decomposes as:

| Model | P30 VR_Total | P30 VR_Content | P30 VR_Order | Interpretation |
|-------|-------------|----------------|-------------|----------------|
| Llama 4 Scout | 7.46 | 16.79 | 0.44 | Extreme content sensitivity |
| Llama 4 Maverick | 2.64 | 6.35 | 0.42 | High content sensitivity |

The anomaly is driven by extreme **content variance amplification** at P30, not by order. COLD responses collapse at P30 (RCI → 0.02-0.11) while SCRAMBLED maintains moderate alignment (~0.62). The safety-related behavior at P30 affects the no-context condition far more than the scrambled-context condition.

### 3.4 Conservation Product K Decomposition

#### 3.4.1 K Has Internal Structure

The conservation product K = ΔRCI × Var_Ratio (Paper 6) decomposes into content and order components:

| Domain | K_total | K_content | K_order |
|--------|---------|-----------|---------|
| Medical (N=8) | 0.437 (CV=0.09) | 1.533 (CV=0.25) | 0.118 (CV=0.29) |
| Philosophy (N=7) | 0.120 (CV=0.58) | 0.681 (CV=0.75) | 0.035 (CV=0.49) |

All three K components are significantly domain-dependent:
- K_total: U=56, p=0.0003
- K_content: U=48, p=0.021
- K_order: U=56, p=0.0003

#### 3.4.2 Interpretation

The conservation constant K is not atomic — it has internal structure. Medical K is larger than Philosophy K because **both tributaries are stronger**: medical content carries more conservation product AND medical order carries more conservation product. The tightness of K_total (CV=0.09 in medical) emerges despite the individual components being more variable (CV=0.25, 0.29) — they compensate each other.

### 3.5 Exploration Arc

#### 3.3.1 Domain Patterns

**Philosophy (open-ended):**
- All models show Arc > 1 (diversity increases)
- Mean Arc ≈ 2.03
- Models explore and expand their response space

**Medical (task-convergent):**
- Mixed: Arc range 0.87-1.26
- Mean Arc ≈ 1.06
- Models maintain stability or slightly converge toward diagnosis

#### 3.3.2 Correlation with Var_Ratio

- Philosophy: r = -0.92, p = 0.08 (strong negative trend, near threshold)
- Medical: r = 0.47, p = 0.24 (weak positive)
- Overall: r = -0.41, p = 0.18

Note: These correlations were computed with limited sample (12 model-domain configurations). Legal domain data will increase statistical power.

### 3.4 Cross-Dimensional Integration

#### 3.4.1 CUD × Content-Order Interaction

**Hypothesis**: Models with high CUD (deep context processing) should show stronger order effects, because they process the full sequential structure. Recency-dominant models (CUD ≈ 1) should be more content-driven, since they ignore the sequence beyond K=1.

Maverick (CUD > 1) in medical: Content Fraction = 58% (lowest among medical models, most order-dependent)
DeepSeek (CUD = 1) in medical: Content Fraction = 62% (more content-driven)

Direction is consistent with hypothesis. Larger CUD sample needed for statistical testing.

#### 3.4.2 CUD × Var_Ratio (Llama Anomaly)

Papers 1-6 identified Llama's high Var_Ratio (up to 7.46 at medical P30). Paper 7 CUD reveals Maverick has the only rising K-curve. These may be two measurements of the same property: **deep context dependence amplifies trial-to-trial variance**.

| CUD Class | Expected Var_Ratio | Models | Confirmed? |
|-----------|-------------------|--------|------------|
| High CUD (deep) | High Var_Ratio | Llama family | Yes (Papers 2-6) |
| Low CUD (recency) | Low Var_Ratio | DeepSeek, Gemini, Qwen3 | Yes (Papers 2-6) |

#### 3.4.3 Exploration Arc × Content-Order

Philosophy shows both high Order Fraction (~60%) and high Exploration Arc (~2.03). This is consistent: models in open-goal domains rely on sequential argument building (order-dependent) and expand their response space as more argument structure is available (exploration).

Medical shows high Content Fraction (~60%) and stable Arc (~1.06). Models converge toward a diagnostic answer (content-driven, non-exploratory).

---

## 4. Discussion

### 4.1 Context as a Multi-Dimensional Property

Prior work treats context as a single dimension: present or absent, long or short. Our decomposition reveals context sensitivity has multiple measurable dimensions:

1. **Depth** (CUD): How far back? Most models are recency-dominant — K=1 captures most of the context effect. This challenges assumptions that longer context windows automatically produce better results.

2. **Structure** (Content/Order): What kind? The dominant component depends on domain goal structure. Closed-goal tasks (medical, legal) are content-driven — the facts matter, not the order. Open-goal tasks (philosophy) are order-driven — the argument structure matters more than the raw content.

3. **Stability** (Variance Decomposition): Content and order have opposite effects on response variance. Content alone (SCRAMBLED) amplifies variance 4-6× — it gives the model something to work with but no structure to anchor on. Order (TRUE vs SCRAMBLED) suppresses variance to ~30% — the sequential structure constrains interpretation. These two forces approximately cancel, explaining why Var_Ratio ≈ 1 for most models.

4. **Trajectory** (Exploration Arc): Where is it going? Open-goal domains produce expanding response diversity. Closed-goal domains produce convergent responses.

### 4.2 The Sensitivity-Stability Dissociation

A key finding is that the Content-Order decomposition tells **different stories** depending on whether you measure sensitivity (ΔRCI) or stability (Var_Ratio):

| | Sensitivity (ΔRCI) | Stability (Var_Ratio) |
|--|---------------------|----------------------|
| Content effect | Increases ΔRCI | Amplifies variance (VR_Content = 4-6×) |
| Order effect | Increases ΔRCI further | Suppresses variance (VR_Order = 0.3×) |
| Domain difference | Yes (p=0.0002) | No (p=0.46, p=0.87) |

Content and order both *increase* sensitivity but have *opposite* effects on stability. This dissociation means you cannot predict variance behavior from sensitivity alone — they are structurally independent dimensions that happen to share the same content/order decomposition.

### 4.3 The Conservation Product Has Internal Structure

Paper 6 established ΔRCI × Var_Ratio ≈ K(domain). We show K is not atomic but decomposes into K_content and K_order, both domain-dependent. Medical K_total's remarkable tightness (CV=0.09) emerges from two individually more variable components (CV=0.25, 0.29) that compensate each other — suggesting a deeper structural constraint governing how content and order contributions balance within each domain.

### 4.4 Practical Implications

#### RAG System Design
- For medical/clinical RAG: prioritize **content retrieval completeness** over document ordering
- For reasoning/argument RAG: prioritize **sequential coherence** of retrieved passages
- Current RAG systems treat all contexts the same — our decomposition suggests domain-specific context assembly strategies

#### Prompt Engineering
- Medical prompts: include all relevant facts, order is secondary (~40% of effect)
- Reasoning prompts: maintain logical sequence, even at cost of some content (~60% of effect)

#### Context Window Optimization
- CUD ≈ 1 for most models means longer context windows have diminishing returns
- Models that genuinely process deep context (Maverick-like) show higher variance — a tradeoff between depth and reliability

#### Clinical AI Deployment
- Content-dominant medical context processing is **robust to information arrival order** — a patient presenting symptoms in any order should receive similar diagnostic reasoning
- This is a desirable property for clinical safety
- However, presenting facts without clinical flow (scrambled) increases response variance 4-6× — ordering reduces this to ~30%, arguing for structured clinical templates even when content dominates sensitivity

### 4.5 The COLD Baseline as Domain Prior

The COLD prior voice analysis (Paper 6, Supplementary) demonstrates that domain structure pre-exists in context-free responses — cross-vendor medical cosine similarity exceeds 0.97 while cross-domain similarity is ~0.18. Domain is a stronger determinant of the COLD baseline than model architecture. The Content-Order decomposition presented here shows how conversational context modulates this domain-shaped prior, with content and order contributing differentially depending on goal structure. The conservation product K is thus not created by context — it is already latent in the domain prior, and context merely activates it.

### 4.6 The Information Hierarchy Revisited

The universal hierarchy TRUE > SCRAMBLED > COLD (Paper 2, 25/25 configurations) now has a mechanistic explanation:

- **COLD → SCRAMBLED**: Adding content (even disordered) provides the largest jump. Content is irreducible — even broken context carries informational signal.
- **SCRAMBLED → TRUE**: Adding order provides a further boost. But the magnitude of this boost depends on domain structure.

The hierarchy itself may be restated as: **context information is irreducible under permutation**. Scrambling degrades but cannot destroy the informational content of a conversation.

### 4.7 Connection to the Conservation Constraint

Paper 6 established ΔRCI × Var_Ratio ≈ K(domain). The Content-Order decomposition explains *why* K differs between domains and reveals K has internal structure (Section 3.4). Both K_content and K_order are larger in medical than philosophy, meaning the conservation product is fed by two stronger tributaries in closed-goal domains.

The legal domain data (in progress) will test whether this decomposition generalizes to a third domain.

### 4.8 Limitations

1. CUD pilot uses only 4 models — a full 14-model CUD study would strengthen conclusions
2. Exploration Arc correlations do not reach significance — legal domain data needed
3. Content-Order decomposition uses final position only — position-level analysis may reveal dynamics
4. Philosophy's low ΔRCI makes all decomposition metrics noisy in that domain
5. The three dimensions may not be fully independent — CUD and Content/Order may interact

---

## 5. Conclusion

Context is not a monolith. We decompose LLM context sensitivity into measurable dimensions — depth, structure, stability, and trajectory — using the MCH program's existing three-condition experimental design. The most striking finding is the Content-Order dissociation: content and order both increase sensitivity (ΔRCI) but have **opposite effects on variance** — content amplifies variance 4-6× while order suppresses it to ~30%. The sensitivity decomposition is domain-dependent (medical: 60% content, philosophy: 60% order, d=1.85, p=0.000239), but the variance decomposition is domain-invariant. The conservation product K = ΔRCI × Var_Ratio itself decomposes into content and order tributaries, revealing that the conservation law has internal structure. Combined with CUD showing most models are recency-dominant and Exploration Arc showing domain-specific diversity trajectories, these dimensions provide a complete structural account of what "context" means to a large language model. The SCRAMBLED condition — often treated as a mere control — emerges as the key to this decomposition, the shadow dagger hiding in the experimental design from the beginning.

---

## Figures (Planned)

1. **Figure 1**: K-curves for 5 pilot models (medical P30) — rising (Maverick) vs flat (others)
2. **Figure 2**: Content-Order ΔRCI decomposition bar chart — medical vs philosophy, all models
3. **Figure 3**: P30 spike comparison — TRUE-COLD vs SCRAMBLED-COLD z-scores
4. **Figure 4**: Variance decomposition — VR_Content (4-6×) vs VR_Order (~0.3×) bar chart, showing opposite roles
5. **Figure 5**: K decomposition — K_total, K_content, K_order by domain (shows conservation product has internal structure)
6. **Figure 6**: Sensitivity-Stability dissociation matrix (content/order × ΔRCI/Var_Ratio)
7. **Figure 7**: Llama P30 anomaly decomposed — VR_Content vs VR_Order at P30
8. **Figure 8**: Exploration Arc scatter plot — Var_Ratio vs Arc, by domain
9. **Figure 9**: Information hierarchy decomposition schematic (TRUE = Content + Order, SCRAMBLED = Content, COLD = baseline)

---

## Tables (Planned)

1. **Table 1**: CUD values and K-curve parameters for all pilot models
2. **Table 2**: Content Fraction by model and domain (full 14-model dataset)
3. **Table 3**: P30 z-scores for TRUE-COLD and SCRAMBLED-COLD conditions
4. **Table 4**: Variance decomposition — VR_Content, VR_Order, VR_Total for all 15 model-domain runs
5. **Table 5**: K decomposition — K_total, K_content, K_order by model and domain
6. **Table 6**: Sensitivity-Stability dissociation summary (domain-level)
7. **Table 7**: Llama P30 anomaly decomposition (VR_Content, VR_Order at P30)
8. **Table 8**: Exploration Arc values by model and domain
9. **Table 9**: Cross-dimensional correlations (CUD × Content Fraction × Arc × Var_Ratio)

---

## Data Availability

All data from MCH Papers 1-6 used in this analysis. CUD pilot data collected independently. Legal domain data (Paper 6 extension) to be included upon completion.

GitHub: LaxmanNandi/MCH-Research
OSF Pre-registration: https://osf.io/dp8nj/

---

## Key Statistics Summary

| Finding | Statistic | Significance |
|---------|-----------|-------------|
| Content Fraction (ΔRCI): Medical > Philosophy | U=133, p=0.000239, d=1.85 | Highly significant |
| Medical mean Content Fraction | 59.8% ± 9.0% | Content-dominant |
| Philosophy mean Content Fraction | 39.8% ± 12.3% | Order-dominant |
| VR_Content (content amplifies variance) | Medical: 4.22 ± 1.20, Phil: 5.69 ± 2.73 | Universal (4-6×) |
| VR_Order (order suppresses variance) | Medical: 0.32 ± 0.07, Phil: 0.32 ± 0.08 | Universal (~0.3×) |
| VR_Content domain difference | U=21, p=0.463 | Not significant |
| VR_Order domain difference | U=30, p=0.867 | Not significant |
| K_total: Medical vs Philosophy | U=56, p=0.0003 | Highly significant |
| K_content: Medical vs Philosophy | U=48, p=0.021 | Significant |
| K_order: Medical vs Philosophy | U=56, p=0.0003 | Highly significant |
| Medical K_total | 0.437 (CV=0.09) | Tight conservation |
| Philosophy K_total | 0.120 (CV=0.58) | Looser conservation |
| Llama Scout P30: VR_Content | 16.79 | Extreme content sensitivity |
| Llama Scout P30: VR_Order | 0.44 | Order partially compensates |
| P30 spike in SCRAMBLED-COLD | z = +0.97 to +4.75 | Present in all models |
| Information hierarchy (legal, Maverick) | 45/45 trials | TRUE > SCRAMBLED > COLD |
| CUD: Maverick depth gradient | K=1: 73% → K=29: 89% | Only rising K-curve |
| Exploration Arc domain contrast | Medical ~1.06, Philosophy ~2.03 | Domain-dependent |
| Paper 4 vs Paper 6 method correlation | VR_Total: r=0.73, p=0.002 | Methods agree |

---

*Working title alternatives:*
- "The Anatomy of Context: Depth, Structure, and Trajectory of Context Sensitivity in Large Language Models"
- "Content Over Order: Decomposing Context Sensitivity Reveals Domain-Dependent Information Processing in LLMs"
- "What Is Context Made Of? A Three-Dimensional Decomposition of LLM Context Sensitivity"
- "Beyond the Information Hierarchy: Depth, Structure, and Diversity in LLM Context Processing"
