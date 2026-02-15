# Paper 6: Conservation Constraint for Context Sensitivity and Output Variance

**Version:** 2.0
**Date:** February 15, 2026
**Status:** Draft complete
**Author:** Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka

---

## Abstract

This paper reports an empirical conservation constraint governing the relationship between context sensitivity (ΔRCI) and output variance (Var_Ratio) in large language models: the product ΔRCI × Var_Ratio is approximately constant within a domain, across architectures. Across 14 model-domain runs spanning 8 vendors and 11 architectures, the product clusters tightly within each domain (Medical K = 0.429, CV = 0.170; Philosophy K = 0.301, CV = 0.166) while differing significantly between domains (U = 46, p = 0.003; Cohen's d = 2.06). The constraint implies an inverse trade-off consistent with domain-specific resource allocation: increased context sensitivity correlates with decreased output variance, and vice versa.

---

## Research Program Context

| Paper | Question | Core Finding | Status |
|-------|----------|-------------|--------|
| Paper 1 | Does context matter? | ΔRCI validated | Published |
| Paper 2 | How does it vary? | 14-model benchmark | Published |
| Paper 3 | What patterns exist? | Type 1 vs Type 2 architectures | Draft |
| Paper 4 | Why does it happen? | VRI and information theory | Draft |
| Paper 5 | How do we deploy safely? | Four-class predictability taxonomy | Draft |
| **Paper 6** | **Is there a conservation constraint?** | **ΔRCI × VR ≈ K(domain)** | **Draft** |

---

## Key Finding

```
ΔRCI × Var_Ratio ≈ K(domain)
```

| Domain | N | K | SD | CV | 95% CI |
|--------|---|---|----|----|--------|
| Medical | 8 | 0.429 | 0.073 | 0.170 | [0.368, 0.490] |
| Philosophy | 6 | 0.301 | 0.050 | 0.166 | [0.248, 0.353] |

**Domain difference:** Mann-Whitney U = 46, p = 0.003; Welch t = 3.91, p = 0.002; Cohen's d = 2.06

---

## Interpretation

The constraint is consistent with domain-specific resource allocation. The task structure (closed-goal medical vs open-goal philosophy) shapes the domain-specific scaling factor K. Each architecture allocates capacity differently — some toward context sensitivity, others toward output variance — but the product remains approximately constant within a domain. The regularity behaves like a conservation law, pending broader validation across additional domains.

---

## Models Tested

11 unique models from 8 vendors:

| Vendor | Models | Runs |
|--------|--------|------|
| DeepSeek | DeepSeek V3.1 | Med + Phil |
| Google | Gemini Flash | Med + Phil |
| Meta | Llama 4 Scout, Llama 4 Maverick | Med + Phil (Maverick) |
| Mistral | Mistral Small 24B, Ministral 14B | Med |
| Moonshot | Kimi K2 | Med |
| Alibaba | Qwen3 235B | Med |
| Anthropic | Claude Haiku | Phil |
| OpenAI | GPT-4o, GPT-4o Mini | Phil |

---

## Figures

| Figure | Content | File |
|--------|---------|------|
| Fig 1 | Conservation constraint with hyperbolas | fig1_conservation_law_hyperbolas.png |
| Fig 2 | Product distribution by domain | fig2_product_distribution.png |
| Fig 3 | Domain scaling factors comparison | fig3_domain_constants.png |
| Fig 4 | Predictability taxonomy overlay | fig4_taxonomy_overlay.png |

---

## Integration with Paper 5

The four-class predictability taxonomy (IDEAL, EMPTY, HIGH-VARIANCE, RICH) maps onto the conservation constraint. All classes follow the hyperbolic relationship — they represent different allocation strategies within the same domain-specific capacity:

- **IDEAL** (DeepSeek, Kimi K2): Balanced allocation — moderate ΔRCI, moderate VR
- **EMPTY** (Gemini Flash): High ΔRCI, convergent but incomplete outputs
- **HIGH-VARIANCE** (Llama Scout/Maverick): Lower ΔRCI, capacity allocated toward variance
- **RICH** (Qwen3): Moderate ΔRCI with excess variance that remains accurate

---

## Data Location

| Resource | Path |
|----------|------|
| Conservation product CSV | data/paper6/conservation_product_test.csv |
| MI verification results | data/paper6/conservation_law_verification/ |
| Figures | docs/figures/paper6/ |
| Scripts | scripts/analysis/paper6_*.py |

---

**Document Version:** 2.0
**Last Updated:** February 15, 2026
