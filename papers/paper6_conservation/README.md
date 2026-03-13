# Paper 6: Conservation Constraint for Context Sensitivity

**Status**: DRAFT COMPLETE
**Title**: *An Empirical Conservation Constraint on Context Sensitivity and Output Variance: Evidence Across LLM Architectures*

## Overview
Capstone paper of the MCH Research Program. Reports that the product of context sensitivity (ΔRCI) and output variance (Var_Ratio) is approximately constant within a domain, across all architectures tested. This conservation constraint connects findings from all five prior papers under a single quantitative relationship.

## Key Findings
1. **Conservation constraint**: ΔRCI × Var_Ratio ≈ K(domain)
   - Medical K = 0.429 (CV = 0.170, N = 8)
   - Philosophy K = 0.301 (CV = 0.166, N = 6)

2. **Domain scaling factors differ significantly**:
   - Mann-Whitney U = 46, p = 0.003
   - Welch's t = 3.91, p = 0.002
   - Cohen's d = 2.06 (very large)

3. **Resource allocation interpretation**: Context sensitivity and output variance trade off within a domain-specific capacity shaped by task structure.

4. **Predictability taxonomy integration**: The four classes from Paper 5 represent different allocation strategies within the same domain-specific capacity.

5. **MI-based test**: Negative result — KSG estimator failed in high-dimensional space. Conservation constraint established via direct product test.

6. **Embedding robustness**: Conservation holds under alternative embedding (all-mpnet-base-v2, 768D):
   - Medical K = 0.402 (CV = 0.154) vs original 0.429 (CV = 0.170)
   - Philosophy K = 0.282 (CV = 0.141) vs original 0.301 (CV = 0.166)
   - Both CVs *improved* under mpnet — constraint is tighter, not weaker
   - Shared embedding space objection refuted

## Dataset
- **Configurations**: 14 model-domain runs (8 Medical, 6 Philosophy)
- **Models**: 11 unique architectures from 8 vendors
- **Data source**: Paper 2 standardized dataset (models with embedding-based Var_Ratio)
- **Location**: `/data/paper6/` (conservation product CSV + MI verification)

## Contents
- `Paper6_Draft.md`: Complete manuscript (v2.0)
- `Paper6_Definition.md`: Paper definition and scope
- `figures/`: All Paper 6 figures (4 main + variant renderings)

## Main Figures
1. Conservation constraint with domain hyperbolas (14 model-domain runs)
2. Product distribution by domain (within-domain clustering)
3. Domain scaling factors comparison (K_med vs K_phil with 95% CI)
4. Predictability taxonomy overlay on conservation constraint

## Legal Domain Extension — Emerging Insight (March 13, 2026)

### Reasoning Topology Hypothesis
Early legal data (3 models: DeepSeek, Maverick, Qwen3) suggests K(Legal) ≈ 0.30, clustering with Philosophy (K=0.301) rather than Medical (K=0.429). This contradicts the pre-registered prediction of K ≈ 0.41.

**Proposed reframing**: K is not domain-specific but **reasoning-topology-specific**:
- **Convergent reasoning** (single correct answer) → K ≈ 0.43 (Medical: symptoms → diagnosis)
- **Divergent reasoning** (interpretive, argued positions) → K ≈ 0.30 (Philosophy: open inquiry; Legal: fixed rules + ambiguous application)

Legal reasoning has rigid structure (statutes, precedent) but the *application* of that structure to facts is interpretive — more like philosophical argument than diagnostic convergence. The models treat legal as open-goal despite its professional/structured surface.

**Implication**: If confirmed across all 7 legal models, Paper 6 shifts from "domain-specific K" to "topology-specific K" — a stronger, more general claim. The failed pre-registered prediction becomes the paper's strongest finding.

### Legal Trial Status (March 13, 2026)
| Model | ΔRCI (cold) | Trials | Status |
|-------|------------|--------|--------|
| DeepSeek V3.1 | 0.276 | 50/50 | COMPLETE |
| Llama 4 Maverick | 0.209 | 50/50 | COMPLETE |
| Qwen3 235B | 0.265 | 42/50 | RUNNING |
| Llama 4 Scout | — | 0/50 | QUEUED |
| Mistral Small | — | 0/50 | QUEUED |
| Ministral 14B | — | 0/50 | QUEUED |
| Kimi K2 | — | 0/50 | QUEUED |

All 3 completed models show information hierarchy: TRUE > SCRAMBLED > COLD.

## Pre-registration
- **OSF Project**: https://osf.io/7954v/
- **OSF Registration**: https://osf.io/dp8nj/
- **Date**: March 6, 2026
- **Scope**: Prospective replication in 3 new domains (Legal, Technical, Applied Ethics)

## Scripts
- `scripts/analysis/paper6_conservation_law.py` — MI-based conservation test
- `scripts/analysis/paper6_conservation_product.py` — Direct product test
- `scripts/analysis/paper6_figures.py` — Figure generation
- `scripts/analysis/paper6_verify.py` — Statistical verification
- `scripts/analysis/paper6_robustness_embedding.py` — Embedding robustness check (mpnet 768D)

## Supplementary: COLD Prior Voice Analysis
- **Location**: `/data/paper7/`
- **Figures**: `/docs/figures/paper6_supplementary/`
- Domain forces cross-vendor centroid convergence (cos > 0.97 in Medical) — grounds K(domain) at the baseline level
- Same model cross-domain = nearly orthogonal voices (cos ~0.18) — Epistemological Relativity at the prior level
- Llama compressed spring: tightest Var_COLD → highest Var_Ratio with context

## Related Documents
- Paper 4 (entanglement): Provides ΔRCI ~ VRI correlation that conservation constraint quantifies
- Paper 5 (predictability): Taxonomy maps onto hyperbolic constraint

---

**Status**: Ready for submission
