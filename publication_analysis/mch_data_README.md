# MCH Complete Dataset

## Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment

### Overview

This dataset contains the complete experimental results from the MCH (Model Coherence Hypothesis) study, testing how 6 large language models from 3 vendors utilize conversation history.

### Dataset Statistics

- **Total Trials:** 600 (100 per model)
- **Models Tested:** 6
- **Vendors:** OpenAI, Google, Anthropic
- **Date Generated:** January 2026

### Models Included

| Model Name | Vendor | Tier | Model ID |
|------------|--------|------|----------|
| GPT-4o-mini | OpenAI | Efficient | gpt-4o-mini |
| GPT-4o | OpenAI | Flagship | gpt-4o |
| Gemini Flash | Google | Efficient | gemini-1.5-flash |
| Gemini Pro | Google | Flagship | gemini-1.5-pro |
| Claude Haiku | Anthropic | Efficient | claude-3-haiku |
| Claude Opus | Anthropic | Flagship | claude-3-opus |

### File Structure

```
mch_complete_dataset.json
├── metadata
│   ├── title
│   ├── description
│   ├── version
│   ├── created_date
│   ├── author
│   ├── affiliation
│   ├── total_trials
│   ├── trials_per_model
│   ├── models[]
│   ├── schema_version
│   └── experiment_parameters
└── models
    └── [Model Name]
        ├── info (vendor, tier, model_id)
        ├── trial_count
        ├── summary
        │   ├── drci_cold (mean, std, min, max)
        │   └── drci_scrambled (mean, std, min, max)
        └── trials[]
            ├── trial_id
            ├── prompt
            ├── model
            ├── alignments (true, cold, scrambled)
            ├── delta_rci (cold, scrambled)
            ├── entanglement
            └── response_lengths (true, cold, scrambled)
```

### Schema Definitions

#### Trial Object

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | integer | Unique trial identifier (1-100) |
| `prompt` | string | The philosophical prompt used |
| `model` | string | Model identifier |
| `alignments.true` | float | Cosine similarity (True condition, with history) |
| `alignments.cold` | float | Cosine similarity (Cold condition, no history) |
| `alignments.scrambled` | float | Cosine similarity (Scrambled condition) |
| `delta_rci.cold` | float | ΔRCI = Alignment(True) - Alignment(Cold) |
| `delta_rci.scrambled` | float | ΔRCI = Alignment(True) - Alignment(Scrambled) |
| `entanglement` | float | Cumulative entanglement value |
| `response_lengths.true` | integer | Response length (True condition) |
| `response_lengths.cold` | integer | Response length (Cold condition) |
| `response_lengths.scrambled` | integer | Response length (Scrambled condition) |

### Computing ΔRCI

The Delta Relational Coherence Index (ΔRCI) is the primary metric:

```
ΔRCI(cold) = Alignment(True) - Alignment(Cold)
ΔRCI(scrambled) = Alignment(True) - Alignment(Scrambled)
```

Where Alignment is computed as cosine similarity between response embeddings and prompt embeddings using sentence-transformers (all-MiniLM-L6-v2).

**Interpretation:**
- ΔRCI > 0 (significant): **Convergent** - history improves response quality
- ΔRCI ≈ 0: **Neutral** - history has no significant effect
- ΔRCI < 0 (significant): **Sovereign** - history degrades response quality

### Experimental Protocol

1. **Philosophical Dialogue:** Each trial presents prompts exploring consciousness, identity, and philosophy
2. **Three Conditions:**
   - **True:** Full conversation history maintained
   - **Cold:** No history, each prompt answered independently
   - **Scrambled:** Randomized history order
3. **100 Trials:** Each model tested 100 times to ensure statistical power

### API Parameters

- Temperature: 0.7
- Top-p: 0.95
- Delay between calls: 5 seconds
- Retry delay: 10 seconds
- Max retries: 3

### Key Findings

| Model | ΔRCI Mean | 95% CI | Pattern |
|-------|-----------|--------|---------|
| GPT-4o-mini | -0.0091 | [-0.033, +0.015] | Neutral |
| GPT-4o | -0.0051 | [-0.027, +0.017] | Neutral |
| Gemini Flash | -0.0377 | [-0.062, -0.013] | Sovereign |
| Gemini Pro | -0.0665 | [-0.099, -0.034] | Sovereign |
| Claude Haiku | -0.0106 | [-0.034, +0.013] | Neutral |
| Claude Opus | -0.0357 | [-0.057, -0.015] | Sovereign |

### Statistical Analysis

- **Vendor Effect:** F = 6.566, p = 0.0015 (significant)
- **Tier Effect:** F = 2.571, p = 0.109 (not significant)
- **Within-Vendor Correlation:** r = 0.189
- **Cross-Vendor Correlation:** r = 0.002

### Citation

If you use this dataset, please cite:

```bibtex
@article{laxman2026differential,
  title={Differential Relational Dynamics in Large Language Models: Cross-Vendor Analysis of History-Dependent Response Alignment},
  author={Laxman, M M},
  journal={arXiv preprint},
  year={2026}
}
```

### Author

**Dr. Laxman M M, MBBS**
Government Duty Medical Officer, Primary Health Centre Manchi
Bantwal Taluk, Dakshina Kannada, Karnataka, India

### License

This dataset is released for research purposes. Please contact the author for commercial use.

### Acknowledgments

This research was conducted using human-AI collaborative methods with Claude (Anthropic), ChatGPT (OpenAI), Deepseek, and Claude Code.

### Data Integrity

- All 6 files validated for 100 trials each
- No API keys or sensitive data included
- ΔRCI calculations verified
- Schema standardized across all models
