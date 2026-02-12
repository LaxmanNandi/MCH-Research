# Paper 2 (Standardized): Cross-Domain AI Behavior Framework

**Status**: IN PREPARATION
**Title**: *Cross-Domain Measurement of Context Sensitivity in Large Language Models: Medical vs Philosophical Reasoning*

## Overview
Core standardized study with unified methodology across 14 unique models and 2 domains (25 model-domain runs). Demonstrates domain-specific behavioral signatures and validates ΔRCI as robust cross-domain metric.

## Key Contributions
1. **Unified methodology**: Corrected trial definition, 50 trials all models
2. **Cross-domain validation**: Medical (closed-goal) vs Philosophy (open-goal)
3. **Architectural diversity**: Open + closed models in both domains
4. **Domain-specific patterns**: U-shaped (medical) vs inverted-U (philosophy)
5. **Large-scale baseline**: 14 models, 25 model-domain runs, 112,500 responses

## Research Questions
1. How does domain structure (closed vs open goal) affect context sensitivity?
2. Do temporal dynamics differ systematically between domains?
3. Are architectural effects (open vs closed models) domain-dependent?
4. Can ΔRCI generalize across task domains?

## Methodology
- **Domains**: Medical (STEMI diagnosis) + Philosophy (consciousness)
- **Models**: 14 unique (13 medical + 12 philosophy = 25 model-domain runs)
- **Trials**: 50 per model (standardized)
- **Total responses**: 112,500
- **Conditions**: 3 per trial (TRUE, COLD, SCRAMBLED)

## Progress
- **Complete**: ALL 25/25 model-domain runs

## Contents
- `figures/`: 6 cross-domain comparison figures
- `MODEL_LIST.md`: Complete model inventory with status
- `Paper2_Draft.md`: Main manuscript draft

## Data Location
All data in `/data/` directory (single source of truth):
- Medical: `/data/medical/open_models/`, `/data/medical/closed_models/`
- Philosophy: `/data/philosophy/open_models/`, `/data/philosophy/closed_models/`

## Extensions
This dataset serves as the foundation for:
- **Paper 3**: Temporal dynamics analysis (10 models with text)
- **Paper 4**: Entanglement mechanism (11 models with text)

## Related Documents
- `docs/RESEARCH_OUTLINE.md`: Complete research program
- `docs/PAPER_COMPARISON.md`: Paper 1 vs Paper 2 evolution

---

**Target**: Nature Machine Intelligence, Science Advances, PNAS
