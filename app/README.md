# MCH Dataset Explorer

Interactive Streamlit application for exploring the Model Coherence Hypothesis (MCH) experimental results.

## Dataset Overview

- **Total Trials**: 1000
- **Philosophy Domain**: 700 trials (7 models × 100 trials)
- **Medical Domain**: 300 trials (6 models × 50 trials)

### Models Tested

| Vendor | Models |
|--------|--------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-5.2 |
| Anthropic | Claude Opus, Claude Haiku |
| Google | Gemini 2.5 Pro, Gemini 2.5 Flash |

## Key Findings

- **GPT-5.2**: Uniquely CONVERGENT in both domains
- **Gemini Flash**: Persistently SOVEREIGN in both domains
- **Domain Effect**: Most models flip from SOVEREIGN/NEUTRAL (philosophy) to CONVERGENT (medical)

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
cd app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

1. **Overview Dashboard** - Summary statistics and violin plots
2. **Model Explorer** - Deep dive into individual model performance
3. **Trial Viewer** - Examine specific trials with full details
4. **Model Comparison** - Side-by-side comparison with t-tests
5. **Domain Analysis** - Philosophy vs Medical behavior analysis
6. **Vendor Analysis** - Performance grouped by vendor with ANOVA
7. **Export Data** - Download filtered data as CSV/JSON

## Data Structure

Each trial contains:
- `trial_id`: Unique identifier
- `model`: Model name
- `domain`: philosophy or medical
- `prompt`: The question asked
- `delta_rci`: ΔRCI = alignment_true - alignment_cold
- `pattern`: CONVERGENT (>0.01), SOVEREIGN (<-0.01), or NEUTRAL
- `align_true/cold/scrambled`: Embedding alignments for each condition
- `insight_quality`: Quality metric (philosophy only)
- `entanglement`: Contextual entanglement score (philosophy only)

## Citation

```
MCH Experiments - Model Coherence Hypothesis
Testing context sensitivity across frontier AI models
January 2026
```
