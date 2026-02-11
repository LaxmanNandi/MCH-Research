# LLM Context Sensitivity Benchmark: 22 Models, 99K+ Responses

## Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **TL;DR:** Reproducible benchmark measuring how 22 major LLMs (GPT-4o, Claude Opus, Gemini 2.5 Pro, Llama 4, DeepSeek V3, Qwen3 235B, Mistral, Kimi K2) handle context across 99,000+ responses. Reveals position-dependent behavior patterns, domain-specific dynamics, and safety-critical divergence in medical reasoning tasks. Strong validation: ΔRCI correlates with mutual information (r=0.76, p=1.5×10⁻⁶²).

**Large-scale analysis: 22 model-domain configurations, 99,000+ responses**

*Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka, India*

**Keywords:** LLM evaluation · Context sensitivity · GPT-4 benchmarking · Claude analysis · Llama 4 testing · Gemini evaluation · DeepSeek benchmark · AI safety · Medical reasoning · Sentence embeddings · Mutual information · Reproducible research · NLP evaluation

---

## Start Here

- **Temporal Dynamics Results:** `docs/papers/Paper3_Results.md`
- **Entanglement Analysis Results:** `docs/papers/Paper4_Results.md`
- **Safety Anomaly Note (Llama P30):** `docs/papers/Llama_Safety_Note.md`
- **Data Availability Index:** `docs/data_availability_index.md`

**Why this matters:** Context effects in LLMs are not uniform; they change by position, domain, and task type. This repository provides reproducible evidence and a structured framework (taxonomy + dual-axis metrics) to distinguish when context improves reliability versus when it destabilizes it—critical for medical and safety-relevant applications.

**Featured Finding:** Position-dependent entanglement spike in medical summarization tasks reveals model-specific divergence patterns.

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

![Effect Sizes](figures/fig2_effect_sizes_ci.png?v=2)

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
