# Cross-Domain AI Behavior Research Program
**From MCH Experiments to Behavioral Science**

---

## Overview

This research program investigates how domain structure shapes AI context sensitivity through controlled cross-domain experiments comparing medical (closed-goal) vs philosophy (open-goal) reasoning.

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

### **Paper 2 (Standardized): Cross-Domain AI Behavior Framework** [IN PREPARATION]
**Role**: Core Study - Unified methodology, cross-domain validation
**Title**: *Cross-Domain Measurement of Context Sensitivity in Large Language Models: Medical vs Philosophical Reasoning*

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

#### **Paper 3: Temporal Dynamics Analysis** [DRAFT COMPLETE]
**Title**: *How Philosophical vs Medical Reasoning Shapes Context Sensitivity Dynamics in Large Language Models*

**Role**: Extension of Paper 2 - Position-level temporal analysis
- **Dataset**: Paper 2 subset (10 models with response text)
- **Focus**: Domain-specific temporal evolution patterns

**Key Findings**:
1. **Domain-specific temporal patterns**:
   - Philosophy: Inverted-U (mid-conversation peak)
   - Medical: U-shaped (diagnostic independence trough)
2. **Task enablement at P30**: Medical spike (Z > +3.5), philosophy stable
3. **Disruption sensitivity**: Presence > order (context structure matters)
4. **Type 2 scaling law**: ΔRCI ∝ log(context_volume)

**Status**: Figures complete, ready for submission

#### **Paper 4: Entanglement Mechanism** [DRAFT COMPLETE]
**Title**: *Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling*

**Role**: Extension of Paper 2 - Information-theoretic mechanism
- **Dataset**: Paper 2 subset (11 models with response text)
- **Innovation**: Variance-based entanglement measure

**Key Findings**:
1. **Entanglement validation**: ΔRCI ~ MI_proxy (r=0.76, p<10⁻⁶²)
2. **Bidirectional regimes**:
   - Convergent: Var_Ratio < 1 (context reduces variance)
   - Divergent: Var_Ratio > 1 (context increases variance)
3. **Llama safety anomaly**: Extreme divergence at P30 (Var_Ratio > 7)
4. **Domain architecture**: Medical variance-increasing (1.23), Philosophy neutral (1.01)
5. **Variance sufficiency**: Simple surrogate works (no k-NN needed)

**Status**: Figures complete, ready for submission

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
│   └── tables/               (CSV metrics for all 24 models)
├── scripts/
│   ├── experiments/          (Data collection)
│   └── analysis/            (Figure generation)
└── archive/                 (Historical materials)
```

---

## Next Steps

1. ~~Complete Kimi K2 medical~~ **DONE** (50/50 trials, dRCI=0.417)
2. **Generate Paper 2 figures** (cross-domain comparison, all 25 model-domain runs)
3. **Write Paper 2 draft** (standardized framework)
4. **Prepare submission packages** (Papers 2, 3, 4)

---

**Last Updated**: February 12, 2026
**Status**: ALL COMPLETE (25/25 model-domain runs), Paper 2 outline ready
