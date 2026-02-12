# Paper 1 (Legacy): Context Curves Behavior

**Title**: *Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI*
**Status**: Published - Preprints.org (February 2, 2026, corrected version)
**DOI**: 10.20944/preprints202601.1881.v2
**Git Tag**: `paper1-arxiv` (note: tag name retained for historical consistency)

## Overview
Introduced ΔRCI (Delta Relational Coherence Index), a cosine-similarity-based metric measuring context sensitivity through a three-condition protocol (TRUE/COLD/SCRAMBLED). Demonstrated that domain fundamentally alters context utilization across 7 models.

## Key Contributions
- Introduced ΔRCI metric and three-condition protocol
- Discovered **Epistemological Relativity**: AI behavior curves based on knowledge structure
- **Domain flip**: 5/6 models switch from SOVEREIGN/NEUTRAL in philosophy to CONVERGENT in medicine (Cohen's d > 2.7)
- **GPT-5.2 anomaly**: Unique 100% CONVERGENT in both domains (150 trials, σ=0.014-0.021)
- **Vendor signatures**: Systematic differences in context utilization (F=6.52, p=0.0015)
- Categorized models as **CONVERGENT, NEUTRAL, SOVEREIGN**

## Methodology
- **Domains**: 2 (Philosophy + Medical)
- **Models**: 7 closed models (3 vendors: OpenAI, Anthropic, Google)
- **Trials**: 1,000 total = 90,000 API calls
  - Philosophy: 7 models × 100 trials = 700 trials
  - Medical: 6 models × 50 trials = 300 trials (Gemini 2.5 Pro blocked by safety filters)
- **Embedding**: all-MiniLM-L6-v2 (384D)
- **Temperature**: 0.7

## Key Results
| Domain | Pattern | Notable |
|--------|---------|---------|
| Philosophy | Mostly NEUTRAL/SOVEREIGN | GPT-5.2 sole CONVERGENT (+0.310) |
| Medical | Mostly CONVERGENT (+0.30 to +0.38) | Gemini Flash sole SOVEREIGN (-0.133) |
| Cross-domain | Cohen's d > 2.7 for 5/6 models | Domain determines behavioral mode |

## Limitations (Addressed in Paper 2)
1. **Data collection evolved**: Philosophy trials captured additional metrics not systematically collected in medical
2. **Closed models only**: No open-source/self-hosted models
3. **Aggregate ΔRCI**: No position-level temporal analysis
4. **Single embedding model**: No robustness check

## Contents
- `figures/`: Legacy figures from Paper 1 analysis
- `MODEL_LIST.md`: Complete model list with results

## Data Location
- Philosophy data: `/data/philosophy/closed_models/`
- Medical data: `/data/medical/closed_models/`

## Next Steps
See **Paper 2** for standardized cross-domain analysis with 14 models (25 model-domain runs, 112,500 responses), position-level dynamics, and corrected methodology.

---

**Citation**: DOI:10.20944/preprints202601.1881.v2 (February 2, 2026, corrected version)
**URL**: https://www.preprints.org/manuscript/202601.1881/v2
**Original version** (v1): DOI:10.20944/preprints202601.1881.v1 (January 26, 2026)
