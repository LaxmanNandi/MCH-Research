# Cross-Domain AI Behavior: Medical vs Philosophical Reasoning

## Standardized Measurement of Context Sensitivity Across 14 LLMs

[![Preprints.org](https://img.shields.io/badge/Preprints.org-10.20944%2Fpreprints202601.1881.v2-blue.svg)](https://www.preprints.org/manuscript/202601.1881/v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **TL;DR:** Cross-domain experimental study measuring how domain structure shapes context sensitivity in 14 LLMs across 112,500 responses. Medical (closed-goal) reasoning shows diagnostic independence troughs and task enablement spikes; philosophy (open-goal) shows mid-conversation peaks with late decline. Standardized 50-trial methodology validates ΔRCI as robust metric (r=0.76 with VRI, p<10⁻⁶²). Significant vendor signatures persist even excluding outliers (p=0.017). Identifies safety-critical divergence in medical summarization tasks.

**Research Program**: Paper 1 (legacy) → **Paper 2 (standardized framework, 14 models)** → Papers 3 & 4 (deep dives) → Paper 5 (safety taxonomy)

*Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka, India*

**Keywords:** Cross-domain AI evaluation · Context sensitivity · Medical vs philosophical reasoning · GPT-4o · Claude · Gemini · Llama 4 · DeepSeek · Qwen · Mistral · AI safety · Domain-specific behavior · Reproducible research

---

## Paper Series

All papers organized in `/papers/` directory by lineage:

### **Paper 1: Context Curves Behavior** [PUBLISHED - Preprints.org]
**Role**: Legacy foundation - Introduced ΔRCI metric and Epistemological Relativity
- 7 closed models, 2 domains (philosophy + medical), 1,000 trials (90,000 API calls)
- **Location**: `papers/paper1_legacy/`
- **Status**: [Preprints.org](https://www.preprints.org/manuscript/202601.1881/v2) - DOI:10.20944/preprints202601.1881.v2 (February 2, 2026, corrected version)

### **Paper 2: Scaling Context Sensitivity** [ACCEPTED - Preprints.org]
**Title**: *Scaling Context Sensitivity: A Standardized Benchmark of ΔRCI Across 25 Model-Domain Runs*
**Role**: Core study - Standardized methodology, cross-domain validation
- 25 model-domain runs × 50 trials = **112,500 responses** (14 unique models)
- Medical (13 models: 6 closed + 7 open) + Philosophy (12 models: 5 closed + 7 open)
- **Location**: `papers/paper2_standardized/`
- **Status**: Preprints.org (ID: 198770, accepted February 12, 2026)

### **Papers 3 & 4: Extensions** [DRAFTS COMPLETE]
**Role**: Deep dives using Paper 2's standardized dataset
- **Paper 3**: Temporal dynamics → `papers/paper3_cross_domain/Paper3_Results.md`
- **Paper 4**: Entanglement mechanism → `papers/paper4_entanglement/Paper4_Results.md`

### **Paper 5: Safety Taxonomy** [DEFINED]
**Role**: Deployment framework - Accuracy verification and behavioral classification
- 8 medical models scored against 16-element clinical accuracy rubric (50 trials each)
- Four behavioral classes: IDEAL, EMPTY, DANGEROUS, RICH
- 2×2 deployment matrix (Var_Ratio × Accuracy)
- **Location**: `papers/paper5_safety/Paper5_Definition.md`

**Complete Structure**: See `papers/README.md`

**Why this matters:** Domain structure fundamentally shapes how LLMs use context. Medical (closed-goal) tasks show diagnostic independence troughs and task enablement spikes; philosophical (open-goal) tasks show recursive accumulation. This repository provides the first **standardized cross-domain framework** to measure these effects across 14 models (25 model-domain runs) with unified methodology—critical for understanding deployment risks in medical and safety-relevant applications.

**Featured Finding:** Medical P30 task enablement reveals safety-critical divergence classes. While convergent models (DeepSeek, Gemini) stabilize under context (Var_Ratio < 0.6), Llama models show extreme variance explosion (Var_Ratio up to 7.5), producing unpredictable outputs precisely when task completion requires context integration.

![Featured figure: Medical P30 entanglement spike](docs/figures/publication/entanglement_validation.png?v=2)
*Caption: Complete 12-model analysis showing P30 medical summarization divergence. Llama models show extreme variance explosion (Var_Ratio=2.6-7.5), while DeepSeek/Kimi K2/Gemini show convergent entanglement (Var_Ratio<1.0). Analysis validates ΔRCI as VRI surrogate (r=0.76, p=8.2×10⁻⁶⁹, N=360).*

---

## Key Findings

### 1. Epistemological Relativity v2.0
Domain shapes temporal dynamics of context sensitivity:

| Domain | Temporal Pattern (3-bin aggregation) |
|--------|------------------|
| Philosophy (open-goal) | Mid-conversation peak, late decline (inverted-U in Early/Mid/Late bins) |
| Medical (closed-goal) | Diagnostic independence trough, integration rise (U-shape in bins) + Type-2 spike at P30 |

### 2. Vendor Signatures
Significant vendor-level differences in context utilization (F=90.65, p<0.0001).

### 3. Variance Reduction Entanglement
Strong correlation (r=0.76, p=1.5×10⁻⁶²) between ΔRCI and VRI (Variance Reduction Index) across 330 position-level measurements (11 model-domain runs), validating information-theoretic interpretation.

---

## Repository Structure

**Organized by paper lineage** (not file type):

```
mch_experiments/
├── papers/                 # Research outputs organized by paper
│   ├── paper1_legacy/     # Paper 1 (Preprints.org, mixed methodology)
│   ├── paper2_standardized/  # Paper 2 (IN PREP, core study)
│   ├── paper3_cross_domain/  # Paper 3 (temporal dynamics)
│   ├── paper4_entanglement/  # Paper 4 (information theory)
│   └── paper5_safety/        # Paper 5 (deployment safety taxonomy)
├── data/                   # Experiment data (single source of truth)
│   ├── medical/           # Medical domain (STEMI case)
│   │   ├── closed_models/ # 7 models (GPT, Claude, Gemini)
│   │   └── open_models/   # 7 models (DeepSeek, Kimi, Llama, Mistral, Qwen)
│   └── philosophy/        # Philosophy domain (consciousness)
│       ├── closed_models/ # 5 models (GPT, Claude, Gemini)
│       └── open_models/   # 7 models (DeepSeek, Kimi, Llama, Mistral, Qwen)
├── results/               # Analysis outputs
│   └── tables/           # Data tables (CSV)
├── scripts/               # Code
│   ├── experiments/      # Run experiments
│   └── analysis/         # Analyze data
├── docs/                  # Documentation
│   ├── RESEARCH_OUTLINE.md
│   └── PAPER_COMPARISON.md
└── archive/              # Historical materials
```

**Key principle**: Data stored once in `/data/`, papers organized by lineage in `/papers/`.


---

## Quick Start

```bash
# Clone
git clone https://github.com/LaxmanNandi/MCH-Experiments.git
cd MCH-Experiments

# Install
pip install -r requirements.txt

# Run analysis
python scripts/analysis/compute_trial_drci.py

# Generate figures
python scripts/analysis/generate_paper3_figures.py
```

---

## Methodology

### Delta Relational Coherence Index (ΔRCI)

ΔRCI measures how context affects response consistency:

```
ΔRCI = mean(RCI_TRUE) - mean(RCI_COLD)
```

Where:
- **RCI_TRUE**: Self-similarity of responses within true context (≈1.0)
- **RCI_COLD**: Cross-similarity between true and scrambled context responses

**Interpretation:**
- ΔRCI > 0: Context increases coherence (positive dependence)
- ΔRCI ≈ 0: Context-independent generation
- ΔRCI < 0: Context decreases coherence (rare; suggests instability)

### Task Types

- **Type 1 (Open-goal):** Philosophy prompts, no single correct answer
- **Type 2 (Closed-goal):** Medical reasoning, diagnostic/therapeutic targets

### Embedding Model
All semantic similarity computed using `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

---

## Models Tested

### Closed (API-based)
- GPT-4o, GPT-4o-mini, GPT-5.2
- Claude Opus, Claude Haiku
- Gemini Flash

### Open (Self-hosted)
- DeepSeek V3.1, Qwen3 235B
- Llama 4 Maverick, Llama 4 Scout
- Mistral Small 24B, Ministral 14B
- Kimi K2

---

## Citation

```bibtex
@article{laxman2026context,
  title={Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI},
  author={Laxman, M M},
  journal={Preprints.org},
  doi={10.20944/preprints202601.1881.v2},
  year={2026},
  note={Corrected version, published February 2, 2026}
}
```

---

## Acknowledgments

See `CONTRIBUTORS.md` for collaborator roles and contributions.

Developed using Distributed Intelligence Architecture (DIA) with Claude Code and GPT-5.2 Codex assistance.

---

## License

MIT License - see `LICENSE` for details.

---

## Contact

**Dr. Laxman M M, MBBS**
Primary Health Centre Manchi, Karnataka, India
Email: barlax5377@gmail.com
GitHub: [@LaxmanNandi](https://github.com/LaxmanNandi)
