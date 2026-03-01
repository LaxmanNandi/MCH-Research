# Papers Directory: Research Lineage

This directory organizes research outputs by **paper lineage** rather than file type.

## Structure

```
papers/
├── paper1_legacy/           Paper 1 ✅ Published (Preprints.org)
├── paper2_standardized/     Paper 2 ✅ Published v2 (Preprints.org)
├── paper3_cross_domain/     Paper 3 ✅ Published (Preprints.org)
├── paper4_entanglement/     Paper 4 ✅ Published (Preprints.org) + TMLR under review
├── paper5_safety/           Paper 5 ✅ Published (Preprints.org)
├── paper6_conservation/     Paper 6 📄 Draft complete
└── paper7_cud/              Paper 7 🔬 Pilot complete (4 models)
```

## Paper Lineage

```
Paper 1 (Legacy) → Paper 2 (Standardized) → Papers 3 & 4 (Extensions) → Paper 5 (Safety) → Paper 6 (Conservation Law) → Paper 7 (Mechanism Independence)
```

### Paper 1: Context Curves Behavior
- **Status**: Published (Preprints.org, February 2, 2026, corrected version)
- **DOI**: 10.20944/preprints202601.1881.v2
- **Role**: Foundation -- Introduced ΔRCI metric and Epistemological Relativity
- **Models**: 7 closed models (GPT-4o/mini/5.2, Claude Opus/Haiku, Gemini Flash/Pro)
- **Domains**: 2 (Philosophy: 700 trials + Medical: 300 trials = 1,000 total)
- **Location**: `paper1_legacy/`

### Paper 2: Scaling Context Sensitivity
- **Status**: Published -- Preprints.org (ID: 198770, February 12, 2026; v2 correction submitted)
- **DOI**: 10.20944/preprints202602.1114.v2
- **v2 Correction**: Gemini Flash Medical ΔRCI corrected -0.133 → +0.427 (alignment method fix)
- **Role**: Core study -- Unified methodology, cross-domain validation
- **Models**: 14 unique models, 25 model-domain runs (13 medical + 12 philosophy)
- **Methodology**: Standardized 50 trials, corrected trial definition
- **Location**: `paper2_standardized/`

### Paper 3: Domain-Specific Temporal Dynamics
- **Status**: ✅ Published (Preprints.org, February 16, 2026)
- **DOI**: 10.20944/preprints202602.1674.v1
- **Role**: Extension of Paper 2 -- Position-level analysis across 30 conversation positions
- **Key Finding**: Domain-specific temporal signatures (U-shape medical, inverted-U philosophy in 3-bin aggregation)
- **Dataset**: Paper 2 data (25 model-domain runs)
- **Location**: `paper3_cross_domain/`

### Paper 4: Engagement as Entanglement
- **Status**: ✅ Published on Preprints.org (ID: 199894, February 22, 2026) + Submitted to TMLR (February 26, 2026)
- **DOI**: Pending assignment
- **Role**: Extension of Paper 2 -- Variance-based entanglement mechanism
- **Key Finding**: ΔRCI ~ VRI correlation r=0.76, p=2.37×10⁻⁶⁸ (N=360), ESI metric for instability prediction
- **Dataset**: Paper 2 subset (12 models with response text for variance computation)
- **Location**: `paper4_entanglement/` (includes `tmlr_submission/` folder)
- **Journal**: TMLR (Transactions on Machine Learning Research) - under review

### Paper 5: Stochastic Incompleteness
- **Status**: ✅ Published on Preprints.org (ID: 200695, February 28, 2026)
- **DOI**: 10.20944/preprints202602.2034.v1
- **Title**: *Stochastic Incompleteness: A Predictability Taxonomy for Clinical AI Deployment*
- **Role**: Application -- Deployment framework based on accuracy verification
- **Key Finding**: Four behavioral classes (IDEAL, EMPTY, DIVERGENT, RICH) based on 2×2 Var_Ratio × Accuracy matrix
- **Dataset**: 8 medical models with response text (P30 summarization, 50 trials each)
- **Location**: `paper5_safety/`

### Paper 6: Conservation Constraint
- **Status**: 📄 Draft complete
- **Role**: Unifying theory -- Conservation constraint across all prior papers
- **Key Finding**: ΔRCI × Var_Ratio ≈ K(domain). Medical K=0.429, Philosophy K=0.301 (Mann-Whitney p=0.003, Cohen's d=2.06)
- **Dataset**: 14 model-domain runs across 11 architectures, 8 vendors
- **Location**: `paper6_conservation/`

### Paper 7: Context Utilization Depth (CUD)
- **Status**: 🔬 Pilot complete (4 models × 2 domains)
- **Role**: Mechanistic validation -- Tests whether conservation constraint is architecture-dependent
- **Key Finding**: CUD (mechanism) is orthogonal to K (capacity). Immediate processors (CUD=1) and deep integrators (CUD=10) converge to same K constraint.
- **Models**: DeepSeek V3.1, Gemini Flash, Llama 4 Maverick, Qwen3 235B
- **Dataset**: 18 JSON files with K-curve measurements, cud_summary.csv
- **Location**: `paper7_cud/` + `/scripts/experiments/paper7_pilot/results/`
- **Analysis**: PAPER7_ANALYSIS_SUMMARY.md (246 lines)

## Each Paper Folder Contains

- `Paper[X]_Manuscript.tex` or `Paper[X]_Draft.md` -- Paper content
- `figures/` -- All figures specific to this paper
- `README.md` -- Overview and key findings
- `archive/` -- Legacy drafts and superseded versions

## Data Location

**All experimental data is stored in `/data/` directory** (single source of truth).

```
/data/
├── medical/
│   ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 ×2, Mistral ×2, Qwen)
│   └── closed_models/    6 models (GPT ×3, Claude ×2, Gemini)
├── philosophy/
│   ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 ×2, Mistral ×2, Qwen)
│   └── closed_models/    5 models (GPT ×3, Claude, Gemini)
├── paper5/               Accuracy verification and Llama deep-dive data
└── paper6/               Conservation law test data and MI verification
```

## Quick Navigation

| Document | Location |
|----------|----------|
| Research outline | `/docs/RESEARCH_OUTLINE.md` |
| Paper 1 vs 2 comparison | `/docs/PAPER_COMPARISON.md` |
| Paper 1 figures | `paper1_legacy/figures/` |
| Paper 2 manuscript | `paper2_standardized/Paper2_Manuscript.tex` |
| Paper 3 manuscript | `paper3_cross_domain/paper3_temporal_dynamics.tex` |
| Paper 4 manuscript | `paper4_entanglement/Paper4_Manuscript.tex` |
| Paper 5 manuscript | `paper5_safety/Paper5_Final_Corrected.tex` |
| Paper 6 draft | `paper6_conservation/Paper6_Draft.md` |
| Conservation data | `/data/paper6/conservation_product_test.csv` |

---

**Last Updated**: March 1, 2026
**Data Status**:
- ✅ Foundation data: 25/25 model-domain runs complete
- ✅ Paper 5 accuracy verification complete
- ✅ Paper 6 conservation constraint data complete
- ✅ Paper 7 CUD pilot complete (4 models)
**Publications**: 5 published (Papers 1-5 on Preprints.org), Paper 4 also under TMLR journal review
