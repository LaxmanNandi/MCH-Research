# Cross-Domain AI Behavior: Medical vs Philosophical Reasoning

## Standardized Measurement of Context Sensitivity Across 24 LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **TL;DR:** Cross-domain experimental study measuring how domain structure shapes context sensitivity in 24 major LLMs across 99,000+ responses. Medical (closed-goal) reasoning shows U-shaped dynamics with task enablement spikes; philosophy (open-goal) shows inverted-U patterns. Standardized 50-trial methodology validates ΔRCI as robust metric (r=0.76 with MI proxy, p<10⁻⁶²). Identifies safety-critical divergence in medical summarization tasks.

**Research Program**: Paper 1 (legacy) → **Paper 2 (standardized framework, 24 models)** → Papers 3 & 4 (deep dives)

*Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka, India*

**Keywords:** Cross-domain AI evaluation · Context sensitivity · Medical vs philosophical reasoning · GPT-4o · Claude · Gemini · Llama 4 · DeepSeek · Qwen · Mistral · AI safety · Domain-specific behavior · Reproducible research

---

## Paper Series

### **Paper 1: Multi-turn Conversational Helpfulness (MCH)** [PUBLISHED - arXiv]
**Role**: Legacy foundation - Introduced ΔRCI metric and MCH framework
- Philosophy domain only (8 models, mixed methodology)
- **Status**: [arXiv preprint](https://arxiv.org/) (January 2026)

### **Paper 2: Cross-Domain AI Behavior Framework** [IN PREPARATION]
**Role**: Core study - Standardized methodology, cross-domain validation
- 24 models × 2 domains × 50 trials = **99,000+ responses**
- Medical (13 models) + Philosophy (11 models)
- **Documents**: `docs/RESEARCH_OUTLINE.md`, `docs/PAPER_COMPARISON.md`

### **Papers 3 & 4: Extensions** [DRAFTS COMPLETE]
**Role**: Deep dives using Paper 2's standardized dataset
- **Paper 3**: Temporal dynamics analysis → `docs/papers/Paper3_Results.md`
- **Paper 4**: Entanglement mechanism → `docs/papers/Paper4_Results.md`

**Complete Research Outline**: See `docs/RESEARCH_OUTLINE.md`

**Why this matters:** Domain structure fundamentally shapes how LLMs use context. Medical (closed-goal) tasks show diagnostic independence troughs and task enablement spikes; philosophical (open-goal) tasks show recursive accumulation. This repository provides the first **standardized cross-domain framework** to measure these effects across 24 models with unified methodology—critical for understanding deployment risks in medical and safety-relevant applications.

**Featured Finding:** Medical P30 task enablement reveals safety-critical divergence classes. While convergent models (DeepSeek, Gemini) stabilize under context (Var_Ratio < 0.6), Llama models show extreme variance explosion (Var_Ratio up to 7.5), producing unpredictable outputs precisely when task completion requires context integration.

![Featured figure: Medical P30 entanglement spike](docs/figures/publication/entanglement_validation.png)
*Caption: Complete 11-model analysis showing P30 medical summarization divergence. Llama models show extreme variance explosion (Var_Ratio=2.6-7.5), while DeepSeek/Gemini show convergent entanglement (Var_Ratio<0.6). Analysis validates ΔRCI as mutual information proxy (r=0.76, p=1.5×10⁻⁶², N=330).*

---

## Key Findings

### 1. Epistemological Relativity v2.0
Domain shapes temporal dynamics of context sensitivity:

| Domain | Temporal Pattern |
|--------|------------------|
| Philosophy (open-goal) | Inverted-U curve (positions 1-29) |
| Medical (closed-goal) | U-shaped curve (positions 1-29) + Type-2 spike at P30 |

### 2. Vendor Signatures
Significant vendor-level differences in context utilization (F=90.65, p<0.0001).

### 3. Mutual Information Entanglement
Strong correlation (r=0.76, p=1.5×10⁻⁶²) between ΔRCI and mutual information proxy across 330 position-level measurements (11 model-domain runs), validating information-theoretic interpretation.

---

## Repository Structure

```
mch_experiments/
├── data/                    # Experiment data
│   ├── medical/            # Medical domain (STEMI case)
│   │   ├── closed_models/
│   │   ├── open_models/
│   │   └── gemini_flash/
│   └── philosophy/         # Philosophy domain (consciousness)
│       ├── closed_models/
│       ├── open_models/
│       └── original/
├── results/                # Analysis outputs
│   ├── figures/           # Publication figures
│   ├── tables/            # Data tables (CSV)
│   └── metrics/           # Computed metrics
├── docs/                   # Documentation
│   └── papers/            # Paper manuscripts
├── scripts/                # Code
│   ├── experiments/       # Run experiments
│   └── analysis/          # Analyze data
├── README.md
├── LICENSE
├── requirements.txt
└── CONTRIBUTORS.md
```

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
- Gemini Flash, Gemini 2.5 Pro

### Open (Self-hosted)
- DeepSeek V3.1, Qwen3 235B
- Llama 4 Maverick, Llama 4 Scout
- Mistral Small 24B, Ministral 14B
- Kimi K2

---

## Citation

```bibtex
@article{nandi2026context,
  title={Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI},
  author={Nandi, Laxman M M},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
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
