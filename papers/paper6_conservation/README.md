# Paper 6: Conservation Constraint for Context Sensitivity

**Status**: SUBMITTED — Preprints.org ID: 207501 (April 9, 2026)
**Title**: *A Conservation Constraint in LLM Context Processing Across Four Epistemological Domains*

## Overview
Capstone paper of the MCH Research Program. Reports that the product of context sensitivity (ΔRCI) and output variance (Var_Ratio) is approximately constant within a domain — K(domain) — across all architectures tested. Extended to 4 domains (Medical, Philosophy, Legal, Applied Ethics) revealing a four-mode taxonomy of context processing determined by truth-type. Legal domain reveals convergence WITHOUT entanglement — a third mode not predicted by Papers 2-5. K also functions as a stability predictor: higher K correlates with lower variance degradation across trials (rho=-0.600).

## Key Findings

### 1. Conservation constraint: ΔRCI × Var_Ratio ≈ K(domain)
| Domain | K | CV | N | Status |
|--------|------|------|---|--------|
| Medical (discovered truth) | 0.429 | 0.170 | 8 | Complete |
| Legal (argued truth) | 0.348 | 0.214 | 5 | Complete |
| Philosophy (explored truth) | 0.301 | 0.166 | 6 | Complete |
| Ethics (felt truth) | 0.223 | 0.162 | 5 | Complete |

*All four domains complete. 24 model-domain runs total.

### 2. Four-mode taxonomy of context processing
| Mode | Domain | Entangled | Arc | Truth type |
|------|--------|-----------|-----|-----------|
| **Discovered** | Medical | Yes (r=0.76) | Convergent (1.72) | Fixed answer exists |
| **Argued** | Legal | **No** (all r ns) | Convergent | Constructed through structure |
| **Explored** | Philosophy | Yes (r=0.76) | Divergent (15.23) | Open-ended inquiry |
| **Felt** | Ethics | **Mixed** (model-dependent) | Mixed (0.93–5.20) | Morally weighted, position-taking |

Key discoveries: Legal reveals **convergence WITHOUT entanglement** — a third mode not predicted by Papers 2-5. Ethics reveals **model-dependent entanglement and arc** — a fourth mode where domain doesn't force topology.

### 3. Content-Order decomposition (Legal domain)
- VR_Order < 0.36 for all models — order **constrains** variance
- VR_Content > 3.7 for all models — content alone **explodes** variance
- SCRAMBLED ≈ COLD in Ethics — order IS the reasoning in moral domains
- Legal argument structure is the constraint, not truth content

### 4. K as stability predictor (Hot Mess connection)
| Domain | K | Trial VR | Stability |
|--------|------|----------|-----------|
| Medical | 0.429 | 1.197 | STABLE |
| Legal | 0.348 | 1.131 | STABLE |
| Philosophy | 0.301 | 1.350 | MILD DEGRADATION |
| Ethics | 0.223 | mixed | MIXED (VR ranges 0.79-1.19, model-dependent) |

K vs Trial Variance Ratio: rho=-0.600 (higher K = more stable, N=4 domains).
Conservation constraint may function as structural immune system against reasoning degradation.

### 5. Domain scaling factors differ significantly
- Medical vs Philosophy: Mann-Whitney U=46, p=0.003, Cohen's d=2.06

### 6. Embedding robustness
- Conservation holds under mpnet (768D): Medical K=0.402, Philosophy K=0.282
- CVs improved — constraint is tighter, not weaker

## Dataset (Updated April 2026)
- **Configurations**: 24 model-domain runs (8 Med, 6 Phil, 5 Legal, 5 Ethics)
- **Models**: 14 unique architectures from 8 vendors
- **Data sources**:
  - `/data/paper6/` — conservation product CSV, MI verification, manuscript data JSON
  - `/data/paper6/robustness/` — MiniLM, MPNet, LaBSE robustness results
  - `/data/legal/open_models/` — legal domain results + metrics
  - `/data/ethics/open_models/` — ethics domain results (all 5 models complete)
  - `/data/medical/`, `/data/philosophy/` — original domains

## Contents
- `Paper6_Draft.md`: Current manuscript draft
- `figures/`: Legacy figures (2-domain, Papers 2-5 era)
- `figures_v2/`: Updated figures (4-domain, all 24 runs)
- `paper6_supplementary.tex/.pdf`: Verified supplementary material
- `v1_submission/`: Preprint submission package
- `archive/`: Old planning docs (Definition, Draft_v1, experimental design)

## Main Figures (figures_v2/)
1. `fig1_conservation_scatter.png` — Conservation constraint with 4 domain hyperbolas (24 runs)
2. `fig2_k_by_domain.png` — K by domain bar chart with CV
3. `fig3_four_mode_taxonomy.png` — Four-mode taxonomy quadrant
4. `fig4_three_embedding.png` — Three-embedding robustness (MiniLM, MPNet, LaBSE)
5. `fig5_content_order.png` — Content-order decomposition
6. `fig6_hierarchy.png` — Theoretical hierarchy (nested diagram)

## Legal Domain — Final Results (N=5)

### K(Legal) = 0.348 (CV=0.214, N=5)
Sits between Philosophy (K=0.301) and Medical (K=0.429). Pre-registered prediction was K≈0.41 — actual value lower, reflecting the constructed nature of legal truth.

### Discovered Truth vs Constructed Truth (March 15, 2026)

**Key insight**: Entanglement (ΔRCI~VRI coupling) depends on whether truth in a domain is discovered or constructed.

- **Medical** (discovered truth — molecular pathways, fixed diagnoses): Context converges the answer → ΔRCI and VRI couple → r=0.76, p=2.37×10⁻⁶⁸ → P30 spike → U-shape temporal dynamics
- **Legal** (constructed truth — dynamic rules, argued verdicts): Context expands argument space → ΔRCI and VRI decouple → r=-0.033, p=0.722 → no P30 spike → mixed temporal dynamics

Per-model legal correlations (N=30 each): DeepSeek r=0.29 (p=0.13), Maverick r=-0.14 (p=0.46), Qwen3 r=0.02 (p=0.92), Mistral Small r=-0.13 (p=0.48). Zero significant — entanglement genuinely absent, not a sample size artifact.

**Conservation holds despite decoupling**: K is more fundamental than entanglement. K constrains the product regardless of whether components are correlated. Hierarchy: Conservation > Entanglement > Temporal dynamics.

**Implication for Paper 5 taxonomy**: IDEAL/EMPTY/DIVERGENT/RICH classification loses predictive power in decoupled domains. In entangled domains, ΔRCI alone predicts safety class. In decoupled domains, both ΔRCI and VRI must be measured independently.

### Legal Temporal Dynamics (3-bin)
| Model | Early | Mid | Late | Pattern |
|-------|-------|-----|------|---------|
| DeepSeek V3.1 | 0.285 | 0.258 | 0.285 | U-shape |
| Llama 4 Maverick | 0.222 | 0.216 | 0.189 | Declining |
| Qwen3 235B | 0.242 | 0.263 | 0.289 | Rising |
| Mistral Small | 0.277 | 0.208 | 0.272 | U-shape |

No consensus temporal pattern — unlike Medical (all U-shape) or Philosophy (all inverted-U). Legal is a "weak situation" that doesn't constrain model temporal behavior.

### Legal Trial Status (Updated March 21, 2026)
| Model | ΔRCI (cold) | Var_Ratio | K | Trials | Status |
|-------|------------|-----------|------|--------|--------|
| DeepSeek V3.1 | 0.276 | 1.225 | 0.338 | 50/50 | COMPLETE |
| Llama 4 Maverick | 0.209 | 1.412 | 0.295 | 50/50 | COMPLETE |
| Qwen3 235B | 0.265 | 1.796 | 0.476 | 50/50 | COMPLETE |
| Mistral Small | 0.252 | 1.338 | 0.338 | 50/50 | COMPLETE |
| Llama 3.3 70B Turbo | 0.206 | 1.428 | 0.294 | 50/50 | COMPLETE |
| Kimi K2.5 | 0.509 | 1.633 | 0.831 | 50/50 | EXCLUDED — COLD refusal + 21% empty |
| GLM-5 | — | — | — | 13/50 | EXCLUDED — 86% empty responses |
| Ministral 14B | — | — | — | 0/50 | UNAVAILABLE (Together AI) |
| Llama 4 Scout | — | — | — | 0/50 | UNAVAILABLE (Together AI) |
| Kimi K2 | — | — | — | 0/50 | UNAVAILABLE (Together AI) |

**N=5 valid models.** K(Legal) = 0.348 (range 0.294–0.476). All 5 show information hierarchy: TRUE > SCRAMBLED > COLD. No P30 spike. No entanglement (all r non-significant). All convergent (arc < 3.0).

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
- Same model cross-domain = nearly orthogonal voices (cos ~0.18) — domain shapes prior voice
- Llama compressed spring: tightest Var_COLD → highest Var_Ratio with context

## Ethics Domain — Fourth Mode: "Felt Truth" (March 2026)

### Trial Status (Updated March 31, 2026)
| Model | ΔRCI | Var_Ratio | K | Ent r | Arc | Trials | Status |
|-------|------|-----------|------|-------|-----|--------|--------|
| DeepSeek V3.1 | 0.257 | 0.920 | 0.236 | 0.23 (ns) | 0.93 | 50/50 | COMPLETE |
| Llama 4 Maverick | 0.181 | 0.925 | 0.167 | 0.63 (p=0.0002) | 5.20 | 50/50 | COMPLETE |
| Qwen3 235B | 0.211 | 0.794 | 0.168 | 0.50 (p=0.005) | 2.23 | 50/50 | COMPLETE |
| Mistral Small 24B | 0.183 | 1.027 | 0.188 | 0.67 (p=0.0001) | 2.96 | 50/50 | COMPLETE |
| Llama 3.3 70B Turbo | 0.172 | 1.191 | 0.205 | 0.60 (p=0.0005) | 2.33 | 50/50 | COMPLETE |

**N=5 valid models (all re-embedded).** K(Ethics) = 0.223 (range 0.167–0.236, CV=0.162).

### K Values — Four-Domain Comparison
| Domain | Truth Type | K | CV | N | Entangled | Arc |
|--------|-----------|------|------|---|-----------|-----|
| Medical | Discovered | 0.429 | 0.170 | 8 | Yes (r=0.76) | Convergent (1.72) |
| Legal | Argued | 0.348 | 0.214 | 5 | No (all ns) | Convergent |
| Philosophy | Explored | 0.301 | 0.166 | 6 | Yes (r=0.76) | Divergent (15.23) |
| Ethics | Felt | 0.223 | 0.162 | 5 | Mixed (model-dependent) | Mixed |

**K ordering: Discovered > Argued > Explored > Felt.** The more subjective the truth-type, the lower K.
**CV = 0.16-0.21 in all four domains** — conservation is equally tight regardless of K value. Ethics has the tightest CV (0.162).

### Ethics-Specific Findings
- **SCRAMBLED ≈ COLD**: Content fraction 80-91% — highest of any domain. Order IS the reasoning.
- **Var_Ratio mixed**: DeepSeek 0.92, Maverick 0.93, Qwen3 0.79 (context reduces variance), Mistral 1.03 (neutral), Llama 70B 1.19 (context increases variance). No consensus — unique to ethics.
- **Entanglement is model-dependent**: Mistral (r=0.67***), Maverick (r=0.63***), Llama 70B (r=0.60***), Qwen3 (r=0.50**), DeepSeek (r=0.23 ns). Four of five entangled, one not — unlike Medical/Philosophy (universally entangled) or Legal (universally decoupled). The "Felt" mode allows model personality to determine coupling.
- **Arc is model-dependent**: Maverick divergent (5.20), Mistral borderline (2.96), Llama 70B convergent (2.33), Qwen3 convergent (2.23), DeepSeek strongly convergent (0.93). Domain doesn't force topology — model personality does.
- **Context individuates, not converges**: Re-embedded TRUE responses are LESS similar to each other than COLD responses (negative re-embedded ΔRCI). Each trial builds a unique moral position. Context doesn't narrow toward an answer — it commits to one.
- **Model personality visible**: DeepSeek evolves framework (dRCI=0.257, "Constrained Maximization"). Maverick rigid (dRCI=0.181). Qwen3 balanced (dRCI=0.211, "Pluralistic Deontology").
- **Framework flip**: Qwen3 TRUE P30 → deontology grounded in care. Qwen3 COLD P30 → rule-consequentialism. Context changes what the model believes.

## EEG Pilot — Biological Neural Network Validation

### MCH framework applied to human brain (Sleep-EDF, 20 subjects)
- **Deep Sleep > REM > Wake** in neural coherence
- Deep > Wake: p=0.0000009, Cohen's d=2.10
- **Hierarchy INVERTS** relative to LLMs (TRUE > COLD becomes Wake < Deep)
- Aligns with Upanishadic consciousness model (Prajna > Taijasa > Vaishvanara)
- Script: `scripts/experiments/eeg_pilot_rci.py`
- Data: `data/eeg_pilot/sleep_rci_pilot.json`

## Related Documents
- Paper 4 (entanglement): Provides ΔRCI ~ VRI correlation that conservation constraint quantifies
- Paper 5 (predictability): Taxonomy maps onto hyperbolic constraint
- Paper 7 (content-order): Decomposition method applied to legal and ethics domains
- Paper 8 (encoding fidelity): K ⊥ Truth — conservation orthogonal to semantic correctness

---

**Status**: SUBMITTED to Preprints.org (ID: 207501, April 9, 2026). 24 model-domain runs, three embedding robustness checks, Table 4 verified against authoritative JSON.
