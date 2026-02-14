# Papers Directory: Research Lineage

This directory organizes research outputs by **paper lineage** rather than file type.

## Structure

```
papers/
├── paper1_legacy/           Paper 1 (Preprints.org, Feb 2026)
├── paper2_standardized/     Paper 2 (ACCEPTED, core study)
├── paper3_cross_domain/     Paper 3 (COMPLETE, extension)
├── paper4_entanglement/     Paper 4 (COMPLETE, extension)
└── paper5_safety/           Paper 5 (DEFINED, deployment framework)
```

## Paper Lineage

```
Paper 1 (Legacy) → Paper 2 (Standardized) → Papers 3 & 4 (Extensions) → Paper 5 (Safety)
```

### Paper 1 (Legacy): Context Curves Behavior
- **Status**: Published (Preprints.org, February 2, 2026, corrected version)
- **DOI**: 10.20944/preprints202601.1881.v2
- **Role**: Foundation - Introduced ΔRCI metric and Epistemological Relativity
- **Models**: 7 closed models (GPT-4o/mini/5.2, Claude Opus/Haiku, Gemini Flash/Pro)
- **Domains**: 2 (Philosophy: 700 trials + Medical: 300 trials = 1,000 total)
- **Location**: `paper1_legacy/`

### Paper 2 (Standardized): Scaling Context Sensitivity
- **Status**: ACCEPTED — Preprints.org (ID: 198770, February 12, 2026)
- **Role**: Core study - Unified methodology, cross-domain validation
- **Models**: 14 unique models, 25 model-domain runs (13 medical + 12 philosophy)
- **Methodology**: Standardized 50 trials, corrected trial definition
- **Location**: `paper2_standardized/`

### Paper 3: Cross-Domain Temporal Dynamics
- **Status**: DRAFT COMPLETE
- **Role**: Extension of Paper 2 - Position-level analysis
- **Dataset**: Paper 2 subset (11 models with response text)
- **Location**: `paper3_cross_domain/`

### Paper 4: Entanglement Mechanism
- **Status**: DRAFT COMPLETE
- **Role**: Extension of Paper 2 - Information-theoretic mechanism
- **Dataset**: Paper 2 subset (12 models with response text)
- **Location**: `paper4_entanglement/`

### Paper 5: Safety Taxonomy for Clinical Deployment
- **Status**: DEFINED
- **Role**: Application - Deployment framework based on accuracy verification
- **Dataset**: 8 medical models with response text (P30 summarization, 50 trials each)
- **Innovation**: 2×2 deployment matrix (Var_Ratio × Accuracy), four behavioral classes (IDEAL, EMPTY, DANGEROUS, RICH)
- **Location**: `paper5_safety/`

## Each Paper Folder Contains

- `README.md` - Paper overview and key findings
- `MODEL_LIST.md` - Complete model inventory with methodology
- `figures/` - All figures specific to this paper
- `Paper[X]_Results.md` - Complete results and discussion (Papers 3 & 4)

## Data Location

**All experimental data is stored in `/data/` directory** (single source of truth).

Papers 3 and 4 use subsets of Paper 2's data - **NO data duplication**.

```
/data/
├── medical/
│   ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 2x, Mistral 2x, Qwen)
│   └── closed_models/    6 models (GPT 3x, Claude 2x, Gemini)
└── philosophy/
    ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 2x, Mistral 2x, Qwen)
    └── closed_models/    5 models (GPT 3x, Claude, Gemini)
```

## Quick Navigation

| Document | Location |
|----------|----------|
| Research outline | `/docs/RESEARCH_OUTLINE.md` |
| Paper 1 vs 2 comparison | `/docs/PAPER_COMPARISON.md` |
| Paper 1 figures | `paper1_legacy/figures/` |
| Paper 2 draft | `paper2_standardized/Paper2_Draft.md` (TBD) |
| Paper 3 results | `paper3_cross_domain/Paper3_Results.md` |
| Paper 4 results | `paper4_entanglement/Paper4_Results.md` |
| Paper 5 definition | `paper5_safety/Paper5_Definition.md` |

## Git Tags

- `paper1-arxiv` - Marks Paper 1 publication (Preprints.org, tag name retained for historical consistency)

---

**Last Updated**: February 14, 2026
**Data Status**: ALL COMPLETE (25/25 model-domain runs), Paper 5 accuracy data generated
