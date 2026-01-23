# Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**1,000 Trials | 90,000 API Calls | 7 Models | 2 Domains**

*Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka, India*

---

## Key Findings

### 1. Epistemological Relativity
AI behavior "curves" based on knowledge structure. The same model exhibits different behavioral modes depending on domain:

| Domain | Epistemology | Dominant Pattern |
|--------|--------------|------------------|
| Philosophy | Open-ended, uncertain | SOVEREIGN/NEUTRAL |
| Medicine | Guideline-anchored, certain | CONVERGENT |

**Effect size: Cohen's d > 3.0** (massive domain modulation)

### 2. GPT-5.2: The Outlier
**100% CONVERGENT in BOTH domains** (150 trials, only model to achieve this)
- Philosophy: ΔRCI = +0.310 (σ=0.014)
- Medical: ΔRCI = +0.379 (σ=0.021)

### 3. Vendor Signatures
Significant vendor-level differences in context utilization (F=6.52, p=0.0015)

![Effect Sizes](figures/fig2_effect_sizes_ci.png)

---

## Results Summary

### Philosophy Domain (700 trials: 7 models × 100 trials)

| Model | Mean ΔRCI | Pattern | Conv% |
|-------|-----------|---------|-------|
| GPT-4o | -0.005 | NEUTRAL | 45% |
| GPT-4o-mini | -0.009 | NEUTRAL | 50% |
| **GPT-5.2** | **+0.310** | **CONVERGENT** | **100%** |
| Claude Opus | -0.036 | SOVEREIGN | 36% |
| Claude Haiku | -0.011 | NEUTRAL | 46% |
| Gemini 2.5 Pro | -0.067 | SOVEREIGN | 31% |
| Gemini 2.5 Flash | -0.038 | SOVEREIGN | 28% |

### Medical Domain (300 trials: 6 models × 50 trials)

| Model | Mean ΔRCI | Pattern | Conv% |
|-------|-----------|---------|-------|
| GPT-4o | +0.299 | CONVERGENT | 100% |
| GPT-4o-mini | +0.319 | CONVERGENT | 100% |
| **GPT-5.2** | **+0.379** | **CONVERGENT** | **100%** |
| Claude Opus | +0.339 | CONVERGENT | 100% |
| Claude Haiku | +0.340 | CONVERGENT | 100% |
| Gemini 2.5 Flash | -0.133 | SOVEREIGN | 0% |

*Note: Gemini 2.5 Pro blocked by safety filters for medical prompts*

---

## Quick Start

```bash
# Clone
git clone https://github.com/LaxmanNandi/MCH-Experiments.git
cd MCH-Experiments

# Install
pip install -r requirements.txt

# Run interactive explorer (no API keys needed)
cd app
streamlit run app.py
```

---

## Repository Structure

```
MCH-Experiments/
├── MCH_Paper1_arXiv.tex       # Paper source (LaTeX)
├── MCH_Paper1_arXiv.pdf       # Compiled paper
├── figures/                   # Publication figures (7 figures)
├── app/
│   ├── app.py                 # Interactive Streamlit explorer
│   └── data/                  # All trial data (JSON)
│       ├── philosophy/        # 700 trials (7 models × 100)
│       └── medical/           # 300 trials (6 models × 50)
├── data/
│   ├── philosophy_results/    # Raw philosophy data
│   └── medical_results/       # Raw medical data
├── scripts/
│   └── verify_prompts.py      # Prompt uniformity verification
└── analysis/                  # Statistical analysis scripts
```

---

## The ΔRCI Metric

**Delta Relational Coherence Index** measures context sensitivity:

```
ΔRCI = RCI_TRUE - RCI_COLD
```

Where RCI = cosine similarity between prompt and response embeddings.

### Three-Condition Protocol
1. **TRUE**: Full coherent conversation history
2. **COLD**: No history (fresh start each prompt)
3. **SCRAMBLED**: History present but randomized

### Pattern Classification
- **CONVERGENT** (ΔRCI > 0): History helps
- **NEUTRAL** (ΔRCI ≈ 0): History irrelevant
- **SOVEREIGN** (ΔRCI < 0): History hurts

---

## Citation

```bibtex
@article{laxman2026context,
  title={Context Curves Behavior: Measuring AI Relational Dynamics with {$\Delta$RCI}},
  author={Laxman, M M},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- GitHub: [@LaxmanNandi](https://github.com/LaxmanNandi)
- Email: barlax5377@gmail.com
- ORCID: [0009-0009-0405-6531](https://orcid.org/0009-0009-0405-6531)
