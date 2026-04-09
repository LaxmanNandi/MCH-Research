# Paper 1 (Legacy): Models and Results

**Status**: Published - Preprints.org (February 2, 2026, corrected version)
**DOI**: 10.20944/preprints202601.1881.v2
**Git Tag**: `paper1-arxiv`

## Models (7 total - Closed models, 3 vendors)

### Philosophy Domain (7 models × 100 trials = 700 trials)

| Model | Vendor | Mean ΔRCI | 95% CI | Pattern | Conv% |
|-------|--------|-----------|--------|---------|-------|
| GPT-4o | OpenAI | -0.005 | [-0.027, 0.017] | NEUTRAL | 45% |
| GPT-4o-mini | OpenAI | -0.009 | [-0.033, 0.015] | NEUTRAL | 50% |
| **GPT-5.2** | OpenAI | **+0.310** | **[0.307, 0.313]** | **CONVERGENT** | **100%** |
| Claude Opus | Anthropic | -0.036 | [-0.057, -0.015] | SOVEREIGN | 36% |
| Claude Haiku | Anthropic | -0.011 | [-0.034, 0.013] | NEUTRAL | 46% |
| Gemini 2.5 Pro | Google | -0.067 | [-0.099, -0.034] | SOVEREIGN | 31% |
| Gemini 2.5 Flash | Google | -0.038 | [-0.062, -0.013] | SOVEREIGN | 28% |

### Medical Domain (6 models × 50 trials = 300 trials)

| Model | Vendor | Mean ΔRCI | 95% CI | Pattern | Conv% |
|-------|--------|-----------|--------|---------|-------|
| GPT-4o | OpenAI | +0.299 | [0.296, 0.302] | CONVERGENT | 100% |
| GPT-4o-mini | OpenAI | +0.319 | [0.316, 0.322] | CONVERGENT | 100% |
| **GPT-5.2** | OpenAI | **+0.379** | **[0.373, 0.385]** | **CONVERGENT** | **100%** |
| Claude Opus | Anthropic | +0.339 | [0.334, 0.344] | CONVERGENT | 100% |
| Claude Haiku | Anthropic | +0.340 | [0.337, 0.343] | CONVERGENT | 100% |
| Gemini 2.5 Flash | Google | -0.133 | [-0.140, -0.126] | SOVEREIGN | 0% |

**Note**: Gemini 2.5 Pro was blocked by safety filters for medical prompts.

### Cross-Domain Shift

| Model | Philosophy | Medical | Shift | Cohen's d |
|-------|-----------|---------|-------|-----------|
| GPT-4o | -0.005 (NEUTRAL) | +0.299 (CONV) | +0.304 | 2.78 |
| GPT-4o-mini | -0.009 (NEUTRAL) | +0.319 (CONV) | +0.328 | 2.71 |
| GPT-5.2 | +0.310 (CONV) | +0.379 (CONV) | +0.069 | 3.82 |
| Claude Haiku | -0.011 (NEUTRAL) | +0.340 (CONV) | +0.351 | 4.25 |
| Claude Opus | -0.036 (SOV) | +0.339 (CONV) | +0.375 | 4.02 |
| Gemini Flash | -0.038 (SOV) | -0.133 (SOV) | -0.095 | 0.42 |

## Totals
- **1,000 trials** = 90,000 API calls
- **Philosophy**: 700 trials (7 models × 100)
- **Medical**: 300 trials (6 models × 50)
- **Vendor analysis**: F(2,697)=6.52, p=0.0015

## Behavioral Categories
- **CONVERGENT**: ΔRCI > 0, p < α — context helps
- **NEUTRAL**: ΔRCI ≈ 0, p ≥ α — context irrelevant
- **SOVEREIGN**: ΔRCI < 0, p < α — context hurts
