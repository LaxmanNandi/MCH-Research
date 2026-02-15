# Papers Directory: Research Lineage

This directory organizes research outputs by **paper lineage** rather than file type.

## Structure

```
papers/
├── paper1_legacy/           Paper 1 (Published, Preprints.org)
├── paper2_standardized/     Paper 2 (Published, Preprints.org)
├── paper3_cross_domain/     Paper 3 (Draft complete)
├── paper4_entanglement/     Paper 4 (Draft complete)
├── paper5_safety/           Paper 5 (Draft complete)
└── paper6_conservation/     Paper 6 (Draft complete)
```

## Paper Lineage

```
Paper 1 (Legacy) → Paper 2 (Standardized) → Papers 3 & 4 (Extensions) → Paper 5 (Safety) → Paper 6 (Conservation Law)
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

### Paper 3: Cross-Domain Temporal Dynamics
- **Status**: Draft complete
- **Role**: Extension of Paper 2 -- Position-level analysis across 30 conversation positions
- **Key Finding**: Domain-specific temporal signatures (U-shape medical, inverted-U philosophy in 3-bin aggregation)
- **Dataset**: Paper 2 subset (12 models with response text)
- **Location**: `paper3_cross_domain/`

### Paper 4: Entanglement and Variance Reduction
- **Status**: Draft complete
- **Role**: Extension of Paper 2 -- Information-theoretic mechanism
- **Key Finding**: ΔRCI ~ VRI correlation r=0.76, p=1.5×10⁻⁶² (N=330)
- **Dataset**: Paper 2 subset (12 models with response text)
- **Location**: `paper4_entanglement/`

### Paper 5: Predictability as Safety Metric
- **Status**: Draft complete
- **Role**: Application -- Deployment framework based on accuracy verification
- **Key Finding**: Four behavioral classes (IDEAL, EMPTY, DANGEROUS, RICH) based on 2×2 Var_Ratio × Accuracy matrix
- **Dataset**: 8 medical models with response text (P30 summarization, 50 trials each)
- **Location**: `paper5_safety/`

### Paper 6: Conservation Law
- **Status**: Draft complete
- **Role**: Unifying theory -- Conservation constraint across all prior papers
- **Key Finding**: ΔRCI × Var_Ratio ≈ K(domain). Medical K=0.429, Philosophy K=0.301 (p=0.003)
- **Dataset**: 14 model-domain runs across 11 architectures, 8 vendors
- **Location**: `paper6_conservation/`

## Each Paper Folder Contains

- `Paper[X]_Definition.md` or `Paper[X]_Results.md` -- Paper content
- `figures/` -- All figures specific to this paper
- `README.md` and `MODEL_LIST.md` -- Overview and model inventory (Papers 1-4)

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
| Paper 3 results | `paper3_cross_domain/Paper3_Results.md` |
| Paper 4 results | `paper4_entanglement/Paper4_Results.md` |
| Paper 5 definition | `paper5_safety/Paper5_Definition.md` |
| Paper 6 draft | `paper6_conservation/Paper6_Draft.md` |
| Conservation data | `/data/paper6/conservation_product_test.csv` |

---

**Last Updated**: February 15, 2026
**Data Status**: ALL COMPLETE (25/25 model-domain runs) + Paper 5 accuracy + Paper 6 conservation
