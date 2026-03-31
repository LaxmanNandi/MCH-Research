# Paper 6: The Theory of Epistemological Relativity

**Version:** 3.0
**Date:** March 30, 2026
**Status:** Capstone — Draft in preparation
**Author:** Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka

---

## Abstract

We propose the **Theory of Epistemological Relativity**: the behavioural laws of language models are conserved across architectures but vary across epistemological domains. We report an empirical conservation constraint — the product of context sensitivity (ΔRCI) and output variance (Var_Ratio) is approximately constant within a domain — validated across four domains representing four truth-types: Medical (Discovered, K=0.429), Legal (Argued, K=0.348), Philosophy (Explored, K=0.301), and Applied Ethics (Felt, K=0.193). The conservation holds across 24 model-domain runs spanning 14 architectures from 8 vendors, with within-domain CV = 0.13–0.19 in all four domains. Each truth-type determines a distinct behavioural mode — including entanglement structure, exploration arc, variance signature, and temporal dynamics — yielding a four-mode taxonomy of context processing. The conservation law is more fundamental than entanglement (which is absent in Legal) and persists across all four truth-types with increasing precision (Ethics CV = 0.134, the tightest of any domain).

---

## Research Program Context

| Paper | Question | Core Finding | Status |
|-------|----------|-------------|--------|
| Paper 1 | Does context matter? | ΔRCI validated; Epistemological Relativity named | Published |
| Paper 2 | How does it vary? | 14-model benchmark, 112,500 responses | Published |
| Paper 3 | What temporal patterns? | U-shape vs inverted-U (3-bin aggregation) | Published |
| Paper 4 | What mechanism? | Entanglement: ΔRCI~VRI r=0.76 | Published + JMLR |
| Paper 5 | How to deploy safely? | IDEAL/EMPTY/DIVERGENT/RICH taxonomy | Published |
| Paper 7 | What's the decomposition? | Content-order; exploration arc | Published |
| Paper 8 | Where does it break? | EFI=0.07; K⊥Truth; Coherent Misalignment | Seeking venue |
| **Paper 6** | **What's the law?** | **K = ΔRCI × VR ≈ constant; four truth-types** | **Capstone** |

---

## Key Finding: Conservation Constraint

```
ΔRCI × Var_Ratio ≈ K(domain)
```

| Domain | Truth Type | N | K | CV | Entangled | Arc |
|--------|-----------|---|------|------|-----------|-----|
| Medical | Discovered | 8 | 0.429 | 0.170 | Yes (r=0.76) | Convergent (1.72) |
| Legal | Argued | 5 | 0.348 | 0.192 | No (all ns) | Convergent |
| Philosophy | Explored | 6 | 0.301 | 0.166 | Yes (r=0.76) | Divergent (15.23) |
| Ethics | Felt | 5 | 0.193 | 0.134 | Mixed | Mixed |

**K ordering: Discovered > Argued > Explored > Felt.**
**CV ≈ 0.17 in all four domains** — conservation equally tight regardless of K value.
**Domain difference (Med vs Phil):** U=46, p=0.003, Cohen's d=2.06

---

## Four-Mode Taxonomy

| Mode | Domain | K | Entangled | Arc | Content Fraction | Var_Ratio | Signature |
|------|--------|------|-----------|-----|-----------------|-----------|-----------|
| **Discovered** | Medical | 0.429 | Yes | Convergent | 45-55% | >1 | P30 spike, U-shape, universal pattern |
| **Argued** | Legal | 0.348 | No | Convergent | ~70-80% | >1 | No P30 spike, mixed temporal, structure constrains |
| **Explored** | Philosophy | 0.301 | Yes | Divergent | 35-55% | ~1 | Inverted-U, response space expands |
| **Felt** | Ethics | 0.193 | Mixed | Mixed | 80-91% | Mixed | Context individuates, model personality visible |

---

## Theoretical Hierarchy

```
Conservation (K) > Entanglement (ΔRCI~VRI coupling) > Temporal dynamics
```

- K holds in all four domains — the deepest constraint
- Entanglement holds in Medical and Philosophy, absent in Legal, mixed in Ethics
- Temporal patterns vary by domain and are surface manifestations of K

---

## Integration with Papers 1-8

| Paper | Contribution to Paper 6 |
|-------|------------------------|
| 1 | Named Epistemological Relativity — the seed |
| 2 | Standardised protocol, 14-model foundation |
| 3 | Temporal dynamics that K explains |
| 4 | Entanglement — a special case of conservation |
| 5 | Safety taxonomy — K predicts deployment class |
| 7 | Content-order decomposition — how K distributes |
| 8 | Boundary condition — K holds when semantics break (K⊥Truth) |

---

## Models Tested (22+ model-domain runs)

| Domain | Models | Vendors |
|--------|--------|---------|
| Medical | 8 | DeepSeek, Google, Meta, Mistral, Moonshot, Alibaba |
| Philosophy | 6 | Anthropic, DeepSeek, Google, OpenAI, Meta |
| Legal | 5 | DeepSeek, Meta, Alibaba, Mistral |
| Ethics | 5 | DeepSeek, Meta (Maverick + 70B), Alibaba, Mistral |

---

## Figures (planned)

1. Four-domain conservation constraint with hyperbolas
2. K gradient: Discovered > Argued > Explored > Felt
3. Four-mode taxonomy comparison (entanglement × arc)
4. Content fraction gradient across domains
5. Var_Ratio signature by truth-type
6. Theoretical hierarchy diagram

---

## Data Location

| Resource | Path |
|----------|------|
| Medical data | data/medical/ |
| Philosophy data | data/philosophy/ |
| Legal data | data/legal/open_models/ |
| Ethics data | data/ethics/open_models/ |
| Conservation CSV | data/paper6/conservation_product_test.csv |
| Scripts | scripts/analysis/paper6_*.py |

---

**Document Version:** 3.0
**Last Updated:** March 30, 2026
