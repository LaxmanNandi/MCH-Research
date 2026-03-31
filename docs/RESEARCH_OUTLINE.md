# Cross-Domain AI Behavior Research Program
**From MCH Experiments to Behavioral Science**

---

## Overview

This research program proposes the **Theory of Epistemological Relativity**: the behavioural laws of language models are conserved across architectures but vary across epistemological domains. Validated across four domains (Medical, Philosophy, Legal, Applied Ethics) representing four truth-types (Discovered, Explored, Argued, Felt), with a conservation constraint K = ΔRCI × Var_Ratio that holds across 14+ LLMs from 8 vendors.

---

## Paper Series Structure

### **Paper 1 (Legacy): Context Curves Behavior** [PUBLISHED]
**Status**: Preprints.org (February 2, 2026, corrected version)
**DOI**: 10.20944/preprints202601.1881.v2
**Role**: Foundation - Introduced ΔRCI metric and Epistemological Relativity

**Methodology**:
- 7 closed models (GPT-4o/mini/5.2, Claude Opus/Haiku, Gemini Flash/Pro)
- 2 domains: Philosophy (700 trials) + Medical (300 trials) = 1,000 trials, 90,000 API calls
- Medical: 6 models (Gemini Pro blocked by safety filters)
- Categories: CONVERGENT, NEUTRAL, SOVEREIGN

**Contribution**:
- Introduced ΔRCI metric and three-condition protocol (TRUE/COLD/SCRAMBLED)
- Domain flip: 5/6 models switch behavioral mode between domains (Cohen's d > 2.7)
- GPT-5.2 anomaly: 100% CONVERGENT in both domains
- Vendor signatures: F(2,697)=6.52, p=0.0015
- **Limitation**: Aggregate ΔRCI only (no position-level analysis), closed models only

---

### **Paper 2 (Standardized): Scaling Context Sensitivity** [PUBLISHED]
**Role**: Core Study - Unified methodology, cross-domain validation
**Title**: *Scaling Context Sensitivity: A Standardized Benchmark of ΔRCI Across 25 Model-Domain Runs*
**Status**: Published on Preprints.org — DOI: 10.20944/preprints202602.1114.v2

**Design**: Controlled cross-domain experimental study
- **Models**: 14 unique models, 25 model-domain runs
  - Medical: 13 models (6 closed + 7 open)
  - Philosophy: 12 models (5 closed + 7 open)
- **Trials**: 50 per model (standardized methodology)
- **Data points**: 25 model-domain runs × 50 trials × 90 prompts = 112,500 responses

**Research Questions**:
1. How does domain structure (closed-goal vs open-goal) affect context sensitivity?
2. Do temporal dynamics differ systematically between domains?
3. Are architectural differences (open vs closed models) domain-specific?

**Key Contributions**:
- Establishes standardized 50-trial measurement framework
- Demonstrates domain-specific behavioral signatures
- Validates ΔRCI as robust cross-domain metric
- Provides baseline data for 14 state-of-the-art models across 25 model-domain runs

**Data Status**: ALL COMPLETE (25/25 model-domain runs)

**Extensions & Deep Dives** (build on Paper 2 standardized data):

#### **Paper 3: Temporal Dynamics Analysis** [PUBLISHED]
**Title**: *Domain-Specific Temporal Dynamics of Context Sensitivity in Large Language Models*
**DOI**: 10.20944/preprints202602.1674.v1

**Role**: Extension of Paper 2 - Position-level temporal analysis
- **Dataset**: Paper 2 subset (12 models with response text)
- **Focus**: Domain-specific temporal evolution patterns

**Key Findings**:
1. **Domain-specific temporal patterns** (3-bin aggregation):
   - Philosophy: Mid-conversation peak, late decline (inverted-U in Early/Mid/Late bins)
   - Medical: Diagnostic independence trough, integration rise (U-shape in bins)
2. **Task enablement at P30**: Medical spike (Z > +2.7), philosophy stable
3. **Disruption sensitivity**: Presence > order (context structure matters)
4. **Type 2 scaling law**: ΔRCI ∝ log(context_volume)

#### **Paper 4: Entanglement Mechanism** [PUBLISHED + JMLR UNDER REVIEW]
**Title**: *Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models*
**DOI**: 10.20944/preprints202603.0055.v1

**Role**: Extension of Paper 2 - Information-theoretic mechanism
- **Dataset**: Paper 2 subset (12 models with response text)
- **Innovation**: Variance-based entanglement measure

**Key Findings**:
1. **Entanglement validation**: ΔRCI ~ VRI (r=0.76, p<10⁻⁶⁹)
2. **Bidirectional regimes**:
   - Convergent: Var_Ratio < 1 (context reduces variance)
   - Divergent: Var_Ratio > 1 (context increases variance)
3. **Llama safety anomaly**: Extreme divergence at P30 (Var_Ratio > 7)
4. **Domain architecture**: Medical variance-increasing (1.20), Philosophy neutral (1.01)
5. **Variance sufficiency**: Simple surrogate works (no k-NN needed)

#### **Paper 5: Safety Taxonomy for Clinical Deployment** [PUBLISHED]
**Title**: *Stochastic Incompleteness: A Predictability Taxonomy for Clinical AI Deployment*
**DOI**: 10.20944/preprints202602.2034.v1

**Role**: Application of Papers 2-4 - Deployment framework
- **Dataset**: 8 medical models with response text (P30 summarization, 50 trials each)
- **Innovation**: 2×2 deployment matrix (Var_Ratio × Accuracy)

**Key Findings**:
1. **Falsification**: Var_Ratio does not linearly predict accuracy (r=-0.24, p=0.56, N=8)
2. **Four behavioral classes**:
   - IDEAL (Var_Ratio < 1.2, high accuracy): DeepSeek, Ministral, Mistral, Kimi K2
   - EMPTY (low Var_Ratio, low accuracy): Gemini Flash (safety filter pathology)
   - DIVERGENT (high Var_Ratio, low accuracy): Llama Scout, Llama Maverick
   - RICH (mild divergence, high accuracy): Qwen3 235B
3. **Categorical, not continuous**: Four-class taxonomy captures structure better than any continuous model (quadratic R²=0.11, F-test p=0.72)

#### **Paper 6: Conservation Constraint — Theory of Epistemological Relativity** [CAPSTONE — DRAFT]
**Title**: *Conservation Without Entanglement: A Four-Domain Taxonomy of Context Processing in Large Language Models*
**Working subtitle**: *The Theory of Epistemological Relativity*

**Role**: Capstone — Theoretical unification of entire programme
- **Dataset**: 4 domains, 22+ model-domain runs (8 Med + 6 Phil + 5 Legal + 3+ Ethics)
- **Innovation**: ΔRCI × Var_Ratio ≈ K(domain); four truth-types determine full behavioural mode

**Key Findings**:
1. **Conservation constraint**: K(Medical)=0.429, K(Legal)=0.348, K(Philosophy)=0.301, K(Ethics)=0.193 — CV=0.13-0.19
2. **Four truth-types**: Discovered > Argued > Explored > Felt (K decreases with subjectivity)
3. **Four behavioural modes**: Each truth-type determines entanglement, arc, variance structure, and temporal dynamics
4. **Legal surprise**: Convergence WITHOUT entanglement — third mode not predicted by Papers 2-5
5. **Ethics surprise**: Model-dependent entanglement and arc — domain reveals model personality, doesn't constrain it
6. **Hierarchy**: Conservation > Entanglement > Temporal dynamics
7. **K ⊥ Truth** (from Paper 8): Conservation holds even when semantic encoding degrades to 7%

**Status**: All four domains COMPLETE. Medical (N=8), Philosophy (N=6), Legal (N=5), Ethics (N=5). 24 model-domain runs. Writing begins April 2026.

#### **Paper 7: Content-Order Decomposition** [PUBLISHED]
**Title**: *The Structure and Trajectory of Context Sensitivity in LLMs: Content-Order Decomposition and Variance Dissociation*
**DOI**: 10.20944/preprints202603.1116.v1

**Role**: Theoretical capstone - Decomposes ΔRCI into content/order components
- **Dataset**: Paper 6 conservation-validated subset (N=8 Medical + N=6 Philosophy)
- **Innovation**: Content-order decomposition, exploration arc, CUD pilot

**Key Findings**:
1. **Content fraction**: Medical ~45-55%, Philosophy ~35-55%
2. **Exploration Arc**: Medical 1.72±0.68 (convergent), Philosophy 15.23±16.64 (divergent) — zero domain overlap
3. **P30 spike decomposition**: All N=8 Medical models show z > +2.43 at position 30
4. **Llama P30 anomaly**: Driven entirely by order component, not content
5. **Sensitivity-stability dissociation**: High content fraction ≠ high Var_Ratio

---

## Data Organization

### Complete Datasets (50 trials each)

**Medical Domain** (13 models):
- Open (7): DeepSeek V3.1, Kimi K2, Llama 4 Maverick/Scout, Qwen3 235B, Mistral Small 24B, Ministral 14B
- Closed (6): GPT-4o, GPT-4o-mini, GPT-5.2, Claude Haiku, Claude Opus, Gemini Flash

**Philosophy Domain** (12 models):
- Open (7): DeepSeek V3.1, Kimi K2, Llama 4 Maverick/Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B
- Closed (5): GPT-4o, GPT-4o-mini, GPT-5.2, Claude Haiku, Gemini Flash

### Response Text Availability
- **With text** (18 runs): 4 phil closed + 7 med open + 5 med closed (excl. Claude Opus) + Gemini Flash separate
- **Metrics only** (8 runs): 7 phil open + 1 med closed (Claude Opus, recovered)
- **Use case**: Papers 3 & 4 require response text; Paper 2 uses all 25 runs

---

## Repository Structure

```
mch_experiments/
├── data/
│   ├── medical/
│   │   ├── open_models/      (6 complete, 1 in progress)
│   │   └── closed_models/    (7 complete)
│   └── philosophy/
│       ├── open_models/      (7 complete)
│       └── closed_models/    (4 complete)
├── docs/
│   ├── papers/
│   │   ├── Paper3_Results.md  (Cross-domain temporal dynamics)
│   │   └── Paper4_Results.md  (Entanglement analysis)
│   └── figures/
│       ├── publication/       (Main figures)
│       ├── paper3/           (Domain-specific figures)
│       └── paper4/           (Supplementary figures)
├── results/
│   └── tables/               (CSV metrics for all 14 models)
├── scripts/
│   ├── experiments/          (Data collection)
│   └── analysis/            (Figure generation)
└── archive/                 (Historical materials)
```

---

## Next Steps

1. ~~Complete Ethics domain~~ — **DONE** (N=5, K=0.193, CV=0.134)
2. **Write Paper 6 capstone** — All data complete. Four domains, four truth-types, Theory of Epistemological Relativity
3. **Paper 8 venue** — Seeking appropriate venue for submission
4. **Paper 4 JMLR review** — Under review since March 4, 2026

---

**Last Updated**: March 30, 2026
**Status**: 8 papers (6 published, 1 seeking venue, 1 capstone ready to write). All four domains complete (24 model-domain runs). Theory of Epistemological Relativity: validated.
