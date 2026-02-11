# Paper 1 (Legacy): Multi-turn Conversational Helpfulness

**Status**: Published - Preprints.org (February 2, 2026, corrected version)
**DOI**: 10.20944/preprints202601.1881.v2
**Git Tag**: `paper1-arxiv` (note: tag name retained for historical consistency)

## Overview
Exploratory study introducing the ΔRCI metric and MCH framework. Demonstrated that context effects exist and vary by model architecture.

## Key Contribution
- Introduced Delta Relational Coherence Index (ΔRCI)
- Demonstrated context effects vary by position
- Categorized models as ALIGNED, RESISTANT, SOVEREIGN

## Methodology
- **Domain**: Philosophy only (consciousness prompts)
- **Models**: 8 closed models
- **Trials**: Mixed (GPT-5.2: 100 standard, others: ~50 flawed)
- **Total responses**: ~15,000

## Limitations (Fixed in Paper 2)
1. **Flawed trial methodology**: Only GPT-5.2 used correct script
2. **Single domain**: Cannot generalize beyond philosophy
3. **Closed models only**: No architectural diversity
4. **Inconsistent trials**: Reduced cross-model comparability

## Contents
- `figures/`: Legacy figures from Paper 1 analysis
- `MODEL_LIST.md`: Complete model list with methodology notes

## Data Location
Data for Paper 1 models is in `/data/philosophy/closed_models/`
**Note**: Only use data with correct methodology (see Paper 2)

## Next Steps
See **Paper 2** for standardized cross-domain analysis using corrected methodology.

---

**Citation**: DOI:10.20944/preprints202601.1881.v2 (February 2, 2026, corrected version)
**URL**: https://www.preprints.org/manuscript/202601.1881/v2
**Original version** (v1): DOI:10.20944/preprints202601.1881.v1 (January 26, 2026)
