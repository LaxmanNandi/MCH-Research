# Paper 3: Cross-Domain Temporal Dynamics

**Status**: DRAFT COMPLETE
**Title**: *How Philosophical vs Medical Reasoning Shapes Context Sensitivity Dynamics in Large Language Models*

## Overview
Extension of Paper 2 analyzing position-level temporal evolution patterns. **Central contribution: Two Cognitive Architectures (Type 1/Type 2) framework showing that goal structure—not domain content—determines context sensitivity patterns. Architectures distinguished by temporal dynamics and task enablement, not by ΔRCI magnitude.**

## Two Cognitive Architectures Framework

Paper 3's core theoretical finding: AI context sensitivity varies systematically by **goal structure**, producing two distinct cognitive architectures distinguished by temporal pattern and task enablement behavior.

| Type | Architecture | Domain | Goal Structure | Temporal Pattern | P30 Behavior | Evidence |
|------|--------------|--------|----------------|------------------|--------------|----------|
| **Type 1** | Open-Goal | Philosophy | Unbounded | Inverted-U (mid-conversation peak) | No spike (summarization feasible) | 4/4 models (100%) |
| **Type 2** | Closed-Goal | Medical | Guideline-bounded | U-shaped (diagnostic independence trough) | **Extreme spike** (Z > +3.5, task enablement) | 6/6 models (100%) |

**ASCII Visualization - Temporal Pattern Distinction:**
```
Type 1 (Open-Goal):     /‾‾‾\     Inverted-U, mid-peak at P15-20
Type 2 (Closed-Goal):   \_____/   U-shaped trough, then P30 spike ↑↑↑
```

**Key Insight**: Architectures are distinguished by **temporal dynamics and P30 task enablement**, not by ΔRCI magnitude (ranges overlap: Philosophy 0.27-0.33, Medical 0.29-0.35). Goal structure—not domain content—determines context sensitivity pattern.

### Why This Matters

1. **Generalizable Framework**: Type 1 vs Type 2 distinction explains domain differences and predicts behavior for ANY task structure:
   - **Type 1 (Open-Goal)**: Creative writing, philosophical reasoning, hypothesis generation → Inverted-U pattern, no P30 spike
   - **Type 2 (Closed-Goal)**: Medical diagnosis, legal contract analysis, guideline-based review → U-shaped pattern, extreme P30 spike

2. **Safety Implications**: Type 2 tasks show **task enablement** (not just performance enhancement) at summarization positions:
   - Medical P30: Z > +3.5 (all 6 models show extreme spike)
   - Philosophy P30: Z ≈ +0.25 (no spike)
   - **Critical**: Models cannot execute Type 2 summarization without context—produce refusals or generic templates in COLD condition

3. **Deployment Predictions**: ANY guideline-anchored task will show Type 2 pattern:
   - Diagnostic independence trough (positions 10-25)
   - U-shaped temporal dynamics
   - Extreme task enablement at summarization prompts

4. **Type 2 Scaling Law**: Figure 6 (`fig6_type2_scaling.png`) shows ΔRCI ∝ log(context_volume)
   - P10 summarization: Z = -0.59 (insufficient context)
   - P30 summarization: Z = +2.01 (sufficient context)
   - Context dependence is graded, not binary

## Key Findings (Through Type 1/Type 2 Framework)

1. **Two Cognitive Architectures Identified**:
   - **Type 1 (Open-Goal)**: Philosophy shows inverted-U temporal pattern, mid-conversation peak (ΔRCI = 0.331 at P15-20), 4/4 models consistent
   - **Type 2 (Closed-Goal)**: Medical shows U-shaped pattern, diagnostic independence trough (ΔRCI = 0.292 at P10-25), 6/6 models consistent
   - **Architecture is determined by goal structure, not domain content**
   - **Distinction is by temporal pattern + P30 behavior, NOT by ΔRCI magnitude** (ranges overlap)

2. **Type 2 Task Enablement Characteristic**:
   - Medical (Type 2): Extreme P30 spike (Z > +3.5) at summarization—models cannot execute task without context
   - Philosophy (Type 1): Stable baseline (no spike)—summarization feasible even without context
   - **This is task enablement, not performance enhancement**

3. **Type 2 Scaling Law**: ΔRCI ∝ log(context_volume)
   - Graded (not binary) context dependence
   - P10 summarization: Z = -0.59 (insufficient context)
   - P30 summarization: Z = +2.01 (sufficient context)
   - Evidence: `fig6_type2_scaling.png`

4. **Disruption Sensitivity**: Context presence > order (14/15 runs, DS < 0)
   - Framework-general finding across both architectures

## Dataset
- **Models**: 10 (subset of Paper 2 with response text)
  - Philosophy: 4 closed (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
  - Medical: 6 open (DeepSeek, Llama 4 Maverick/Scout, Mistral Small, Ministral, Qwen3 235B)
- **Data source**: Paper 2 standardized dataset
- **Location**: `/data/` (shared, no duplication)

## Contents
- `figures/`: All Paper 3 figures (domain comparison, position patterns, P30 analysis)
- `MODEL_LIST.md`: 10-model subset details
- `Paper3_Results.md`: Complete results and discussion

## Figures
1. Position-level ΔRCI by domain
2. Domain grand mean comparison
3. Position 30 Z-score analysis
4. Three-bin temporal analysis (P1-29)
5. Disruption sensitivity patterns
6. Position-specific disruption effects
7. Type 2 scaling validation

## Related Documents
- Parent study: `papers/paper2_standardized/`
- Companion analysis: `papers/paper4_entanglement/`

---

**Status**: Ready for submission
