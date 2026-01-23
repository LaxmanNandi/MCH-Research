# MCH Experiments - 10-Minute Quickstart

## Prerequisites

- Python 3.10+
- API keys for at least one vendor (OpenAI, Google, or Anthropic)
- ~$2.50 for minimal replication

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/LaxmanNandi/MCH-Experiments.git
cd MCH-Experiments

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp config/api_keys.yaml.example config/api_keys.yaml
# Edit config/api_keys.yaml and add your API keys
```

## Quick Exploration (No API keys needed)

```bash
# View existing results with interactive explorer
streamlit run scripts/streamlit_explorer.py
```

This launches a web interface to explore all 700 philosophy trials and 300 medical trials.

## Minimal Replication (~30 minutes, ~$2.50)

```bash
# Run 10 trials per model to verify methodology
python scripts/reproduce/minimal_replication.py
```

This produces a summary report comparing your results against published findings.

## Full Replication (~8-10 hours, ~$25)

```bash
# Run full 100 trials per model
python scripts/reproduce/full_pipeline.py --trials 100
```

## Validate Existing Data

```bash
# Verify data integrity
python scripts/validate/check_data_integrity.py

# Verify statistical calculations
python scripts/validate/verify_statistics.py
```

## Key Files

| File | Description |
|------|-------------|
| `data/philosophy_results/` | 100 trials x 7 models |
| `data/medical_results/` | 50 trials x 6 models |
| `MCH_Paper1_arXiv.pdf` | Published manuscript |
| `figures/` | Publication figures |

## Need Help?

- Open an issue on GitHub
- Check `docs/API_SETUP.md` for detailed API configuration
- Check `docs/EXTENDING.md` for adding new models/domains
