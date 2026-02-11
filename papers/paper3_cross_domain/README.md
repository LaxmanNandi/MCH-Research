# Paper 3: Cross-Domain Temporal Dynamics

**Status**: DRAFT COMPLETE
**Title**: *How Philosophical vs Medical Reasoning Shapes Context Sensitivity Dynamics in Large Language Models*

## Overview
Extension of Paper 2 analyzing position-level temporal evolution patterns. Demonstrates domain-specific temporal signatures: medical U-shaped dynamics vs philosophy inverted-U patterns.

## Key Findings
1. **Domain-specific temporal patterns**:
   - Philosophy (open-goal): Inverted-U curve (mid-conversation peak at P15-20)
   - Medical (closed-goal): U-shaped curve (diagnostic independence trough at P10-25)

2. **Position 30 task enablement**:
   - Medical: Extreme spike (Z > +3.5) at summarization prompt
   - Philosophy: Stable baseline (no spike)

3. **Disruption sensitivity**: Context presence matters more than order
4. **Type 2 scaling law**: ΔRCI ∝ log(context_volume)

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
