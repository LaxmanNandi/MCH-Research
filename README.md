# MCH Experiments - Model Coherence Hypothesis

Research repository for studying differential relational dynamics in Large Language Models.

## Paper

**Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment**

Dr. Laxman M M, MBBS
Primary Health Centre Manchi, Bantwal Taluk, Karnataka, India

## Overview

This repository contains the experimental code and data for the Model Coherence Hypothesis (MCH) research, which introduces the Delta Relational Coherence Index (ΔRCI) metric for quantifying how models utilize conversational context.

## Key Findings

- **Vendor Effect**: Significant (F=6.566, p=0.0015) - architectural decisions at vendor level determine relational behavior
- **Tier Effect**: Not significant (F=2.571, p=0.109) - model scale doesn't predict relational patterns
- **Patterns Discovered**:
  - Google models: SOVEREIGN (negative ΔRCI)
  - OpenAI models: NEUTRAL (no history effect)
  - Anthropic: Tier-differentiated (Haiku: Neutral, Opus: Sovereign)

## Repository Structure

```
mch_experiments/
├── mch_experiment_*.py      # Individual model experiment scripts
├── mch_medical_*.py         # Medical domain experiments
├── mch_results_*.json       # Philosophy domain results (100 trials each)
├── medical_results/         # Medical domain results (50 trials each)
├── publication_analysis/    # Final manuscripts
└── app.py                   # Streamlit data explorer
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
streamlit run app.py
```

## Citation

```bibtex
@article{laxman2026mch,
  title={Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment},
  author={Laxman, M M},
  year={2026},
  journal={arXiv preprint}
}
```

## License

MIT License

## Contact

Dr. Laxman M M, MBBS
GitHub: [@LaxmanNandi](https://github.com/LaxmanNandi)
