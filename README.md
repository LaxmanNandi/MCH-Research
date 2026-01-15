# Model Coherence Hypothesis (MCH)

**Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment**

Dr. Laxman M M, MBBS
Primary Health Centre Manchi, Bantwal Taluk, Karnataka, India

---

## Overview

This repository contains experimental data and code for the Model Coherence Hypothesis (MCH) research, introducing the **Delta Relational Coherence Index (ΔRCI)** - a metric for quantifying how LLMs utilize conversational context.

## Key Findings

| Finding | Result |
|---------|--------|
| **Vendor Effect** | Significant (F=6.566, p=0.0015) |
| **Tier Effect** | Not significant (F=2.571, p=0.109) |

### Relational Patterns by Vendor

| Vendor | Pattern | Meaning |
|--------|---------|---------|
| Google | SOVEREIGN | Performs worse with history |
| OpenAI | NEUTRAL | No history effect |
| Anthropic | Mixed | Haiku: Neutral, Opus: Sovereign |

## Repository Structure

```
MCH-Experiments/
├── README.md
├── data/
│   ├── philosophy_results/    # 100 trials × 6 models
│   └── medical_results/       # 50 trials × 6 models
├── scripts/
│   ├── mch_experiment_*.py    # Experiment runners
│   └── streamlit_explorer.py  # Interactive data explorer
├── figures/
│   └── figure1-3.png          # Publication figures
└── paper/
    └── MCH_Paper1_Final_Manuscript.pdf
```

## Models Tested

| Model | Vendor | Model ID |
|-------|--------|----------|
| GPT-4o-mini | OpenAI | gpt-4o-mini |
| GPT-4o | OpenAI | gpt-4o |
| Gemini Flash | Google | gemini-2.5-flash |
| Gemini Pro | Google | gemini-2.5-pro |
| Claude Haiku | Anthropic | claude-haiku-4-5-20251001 |
| Claude Opus | Anthropic | claude-opus-4-5-20250120 |

## Running the Streamlit Explorer

```bash
pip install streamlit pandas numpy plotly scipy
cd scripts
streamlit run streamlit_explorer.py
```

## Citation

```bibtex
@article{laxman2026mch,
  title={Differential Relational Dynamics in Large Language Models},
  author={Laxman, M M},
  year={2026}
}
```

## License

MIT License
