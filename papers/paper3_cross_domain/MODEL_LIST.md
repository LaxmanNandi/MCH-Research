# Paper 3 (Cross-Domain Temporal Dynamics): Models and Dataset

**Status**: DRAFT COMPLETE
**Role**: Extension of Paper 2 - Position-level temporal analysis
**Dataset**: Paper 2 subset (10 models with response text)

## Models (10 total - Cross-domain subset)

### Philosophy Domain (4 models - Closed only)

| Model | Domain | Trials | Response Text | Status |
|-------|--------|--------|---------------|--------|
| GPT-4o | Philosophy | 50 | ✓ Saved | Complete |
| GPT-4o-mini | Philosophy | 50 | ✓ Saved | Complete |
| Claude Haiku | Philosophy | 50 | ✓ Saved | Complete |
| Gemini Flash | Philosophy | 50 | ✓ Saved | Complete |

### Medical Domain (6 models - Open only)

| Model | Domain | Trials | Response Text | Status |
|-------|--------|--------|---------------|--------|
| DeepSeek V3.1 | Medical | 50 | ✓ Saved | Complete |
| Llama 4 Maverick | Medical | 50 | ✓ Saved | Complete |
| Llama 4 Scout | Medical | 50 | ✓ Saved | Complete |
| Mistral Small 24B | Medical | 50 | ✓ Saved | Complete |
| Ministral 14B | Medical | 50 | ✓ Saved | Complete |
| Qwen3 235B | Medical | 50 | ✓ Saved | Complete |

## Data Source
All data comes from Paper 2's standardized dataset.
**Location**: `/data/` (shared with Paper 2, no duplication)

## Why This Subset?
Response text is required for:
- Qualitative validation of ΔRCI patterns
- Position-level temporal analysis
- Domain-specific behavioral signatures

Only 10 of Paper 2's 24 models have complete response text preserved.

## Key Findings
1. **Domain-specific temporal patterns**:
   - Philosophy (closed): Inverted-U curve (mid-conversation peak)
   - Medical (open): U-shaped curve + P30 task enablement spike

2. **Position 30 anomaly**: Medical models show Z > +3.5 spike at summarization prompt

## Figures
All Paper 3 figures stored in `papers/paper3_cross_domain/figures/`
