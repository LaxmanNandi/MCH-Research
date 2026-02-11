# Data Organization

## Current Authoritative Data (50 Trials Each)

All current analysis uses **50-trial datasets** collected with consistent methodology.

### Medical Domain (STEMI Case)
- **Location**: `data/medical/open_models/` and `data/medical/closed_models/`
- **Status**: COMPLETE (7 models, 50 trials each)
- **Models**:
  - Open: DeepSeek V3.1, Llama 4 Maverick, Llama 4 Scout, Qwen3 235B, Mistral Small 24B, Ministral 14B
  - Closed: GPT-4o, GPT-4o-mini, Claude Haiku, Claude Opus, Gemini Flash, GPT-5.2
- **Exception**: Kimi K2 at 33/50 trials (checkpoint file - incomplete)

### Philosophy Domain (Consciousness Case)
- **Location**: `data/philosophy/open_models/` and `data/philosophy/closed_models/`
- **Status**: COMPLETE (11 models, 50 trials each)
- **Models**: All 11 models complete

## Legacy Data (100 Trials, Pre-Rerun)
- **Location**: `data/philosophy/original/`
- **Note**: Original Paper 1 data with 100 trials per model
- **Use**: Archived for reference, NOT used in current Paper 3/4 analysis

## File Naming Convention
```
mch_results_{model}_{domain}_{trials}trials.json
```

Examples:
- `mch_results_deepseek_v3_1_medical_50trials.json` ✓ Current
- `mch_results_claude_opus_100trials.json` ⚠ Legacy (100 trials)
- `mch_results_claude_opus_medical_43trials_recovered.json` ⚠ Legacy (incomplete)

## Data Integrity
- All 50-trial files validated
- Response text saved starting with medical rerun (Feb 2026)
- Embedding model: `all-MiniLM-L6-v2` (384D)
- Temperature: 0.7 (consistent across all runs)
