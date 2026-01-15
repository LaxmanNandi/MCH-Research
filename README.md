# Model Coherence Hypothesis (MCH)

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment**

*Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka, India*

---

## Key Finding

**Vendor architecture determines relational behavior** (F=6.566, p=0.0015), **not model scale** (F=2.571, p=0.109).

| Vendor | Pattern | Meaning |
|--------|---------|---------|
| Google | **SOVEREIGN** | Performs worse with conversation history |
| OpenAI | **NEUTRAL** | No history effect |
| Anthropic | **Mixed** | Haiku: Neutral, Opus: Sovereign |

![Effect Sizes](figures/figure2_effect_sizes.png)

## Quick Start

```bash
# Clone
git clone https://github.com/LaxmanNandi/MCH-Experiments.git
cd MCH-Experiments

# Install
pip install -r requirements.txt

# Explore existing data (no API keys needed)
streamlit run scripts/streamlit_explorer.py

# Run minimal replication (~30 min, ~$2.50)
cp config/api_keys.yaml.example config/api_keys.yaml
# Add your API keys
python scripts/reproduce/minimal_replication.py
```

## Repository Structure

```
MCH-Experiments/
├── data/
│   ├── philosophy_results/    # 600 trials (100 × 6 models)
│   └── medical_results/       # 300 trials (50 × 6 models)
├── scripts/
│   ├── reproduce/             # Replication scripts
│   ├── validate/              # Data validation
│   └── streamlit_explorer.py  # Interactive explorer
├── figures/                   # Publication figures
├── paper/                     # Manuscript PDF
├── config/                    # Configuration files
└── docs/                      # Documentation
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [API Setup](docs/API_SETUP.md)

## Models Tested

| Model | Vendor | Tier | ΔRCI | Pattern |
|-------|--------|------|------|---------|
| GPT-4o-mini | OpenAI | Efficient | -0.009 | Neutral |
| GPT-4o | OpenAI | Flagship | -0.005 | Neutral |
| Gemini Flash | Google | Efficient | -0.038 | Sovereign |
| Gemini Pro | Google | Flagship | -0.067 | Sovereign |
| Claude Haiku | Anthropic | Efficient | -0.011 | Neutral |
| Claude Opus | Anthropic | Flagship | -0.036 | Sovereign |

## Citation

```bibtex
@article{laxman2026mch,
  title={Differential Relational Dynamics in Large Language Models:
         Cross-Vendor Analysis of History-Dependent Response Alignment},
  author={Laxman, M M},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- GitHub: [@LaxmanNandi](https://github.com/LaxmanNandi)
- Paper: [MCH_Paper1_Final_Manuscript.pdf](paper/MCH_Paper1_Final_Manuscript.pdf)
