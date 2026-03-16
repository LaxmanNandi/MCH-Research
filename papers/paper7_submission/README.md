# Paper 7: Content-Order Decomposition and Variance Dissociation

**Status**: PUBLISHED (March 16, 2026)
**DOI**: 10.20944/preprints202603.1116.v1
**Title**: *The Structure and Trajectory of Context Sensitivity in LLMs: Content-Order Decomposition and Variance Dissociation*
**Author**: Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

## Overview

Decomposes ΔRCI into content and order components using the SCRAMBLED condition, revealing how much of context sensitivity comes from message content alone versus sequential ordering. Introduces Content Fraction, Exploration Arc, and K decomposition as new analytical tools.

## Key Findings

1. **Content-Order Decomposition**: Content accounts for ~45-55% of ΔRCI in Medical, ~35-55% in Philosophy
2. **Exploration Arc**: Medical 1.72±0.68 (convergent), Philosophy 15.23±16.64 (divergent) — zero domain overlap
   - Arc thresholds: convergent < 3.0, divergent > 5.0
3. **P30 Spike Decomposition**: All N=8 Medical models show z > +2.43 at position 30; content and order both contribute
4. **Llama Safety Anomaly**: P30 variance spike driven entirely by order component, not content
5. **Sensitivity-Stability Dissociation**: High content fraction ≠ high Var_Ratio

## Model Set

Uses Paper 6's conservation-validated subset (N=14 model-domain runs):
- **Medical (8)**: Gemini Flash, DeepSeek V3.1, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B
- **Philosophy (6)**: Claude Haiku, DeepSeek V3.1, Gemini Flash, GPT-4o, GPT-4o-mini, Llama 4 Maverick

## Contents

- `paper7.tex` — LaTeX source
- `paper7.pdf` — Final published PDF
- `figures/` — 9 figures (PNG)
- `archive/` — Prior manuscript versions, generation scripts, verification scripts

## Figures

1. Content-Order Decomposition bars (N=14)
2. P30 Spike Decomposition (N=8 Medical, z-scores)
3. Variance Decomposition (VR_Content vs VR_Order)
4. K Decomposition
5. Sensitivity-Stability scatter
6. Llama P30 Anomaly
7. Exploration Arc (log scale)
8. Information Hierarchy schematic

## Supplementary

CUD (Context Utilization Depth) pilot in Supplementary S1 — moved from main paper due to noise. Concept docs and pilot analysis in `archive/`.

## Related Papers

- **Paper 6**: Conservation constraint (ΔRCI × Var_Ratio ≈ K) — provides the validated model set
- **Paper 4**: Entanglement (VRI correlation) — variance reduction that Paper 7 decomposes
- **Paper 2**: Foundation data (14 models, 112,500 responses)

---

**Published**: March 16, 2026
**Preprints.org**: DOI 10.20944/preprints202603.1116.v1
