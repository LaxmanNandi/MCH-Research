# Papers Directory: Research Lineage

This directory organizes research outputs by **paper lineage** rather than file type.

## Structure

```
papers/
├── paper1_legacy/           Paper 1 (arXiv, Jan 2026)
├── paper2_standardized/     Paper 2 (IN PREP, core study)
├── paper3_cross_domain/     Paper 3 (COMPLETE, extension)
└── paper4_entanglement/     Paper 4 (COMPLETE, extension)
```

## Paper Lineage

```
Paper 1 (Legacy) → Paper 2 (Standardized) → Papers 3 & 4 (Extensions)
```

### Paper 1 (Legacy): Multi-turn Conversational Helpfulness
- **Status**: Published (arXiv preprint, January 2026)
- **Role**: Foundation - Introduced ΔRCI metric
- **Models**: 8 philosophy models (mixed methodology)
- **Location**: `paper1_legacy/`

### Paper 2 (Standardized): Cross-Domain AI Behavior Framework
- **Status**: IN PREPARATION
- **Role**: Core study - Unified methodology, cross-domain validation
- **Models**: 24 models (13 medical, 11 philosophy)
- **Methodology**: Standardized 50 trials, corrected trial definition
- **Location**: `paper2_standardized/`

### Paper 3: Cross-Domain Temporal Dynamics
- **Status**: DRAFT COMPLETE
- **Role**: Extension of Paper 2 - Position-level analysis
- **Dataset**: Paper 2 subset (10 models with response text)
- **Location**: `paper3_cross_domain/`

### Paper 4: Entanglement Mechanism
- **Status**: DRAFT COMPLETE
- **Role**: Extension of Paper 2 - Information-theoretic mechanism
- **Dataset**: Paper 2 subset (11 models with response text)
- **Location**: `paper4_entanglement/`

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
│   ├── open_models/      6 models (DeepSeek, Llama 4 2x, Mistral 2x, Qwen)
│   └── closed_models/    7 models (GPT 3x, Claude 2x, Gemini 2x)
└── philosophy/
    ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 2x, Mistral 2x, Qwen)
    └── closed_models/    4 models (GPT 2x, Claude, Gemini)
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

## Git Tags

- `paper1-arxiv` - Marks Paper 1 (arXiv) publication

---

**Last Updated**: February 12, 2026
**Data Status**: 23/24 models complete (Kimi K2 medical in progress)
