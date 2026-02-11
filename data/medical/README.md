# Medical Domain Data (STEMI Case)

## Current: 50-Trial Datasets

### Closed Models (API-based)
**Location**: `closed_models/`
- GPT-4o (50 trials) ✓
- GPT-4o-mini (50 trials, rerun) ✓
- GPT-5.2 (50 trials) ✓
- Claude Opus (50 trials) ✓
- Claude Haiku (50 trials) ✓

### Open Models (Self-hosted via Together AI)
**Location**: `open_models/`
- DeepSeek V3.1 (50 trials) ✓
- Llama 4 Maverick (50 trials) ✓
- Llama 4 Scout (50 trials) ✓
- Qwen3 235B (50 trials) ✓
- Mistral Small 24B (50 trials) ✓
- Ministral 14B (50 trials) ✓
- Kimi K2 (33/50 trials) ⚠ INCOMPLETE

### Gemini Flash Rerun
**Location**: `gemini_flash/`
- Gemini Flash (50 trials) ✓

## Legacy Files
- `mch_results_claude_opus_medical_43trials_recovered.json` - Incomplete early run
- `gemini_pro_safety_blocked.json` - Safety filter test

## Data Integrity
✓ All 50-trial files validated
✓ Response text saved for all models
✓ Embedding: all-MiniLM-L6-v2 (384D)
✓ Temperature: 0.7

**Total**: 6 complete open models + 6 complete closed models = 12 models
**Status**: Medical analysis ready for Paper 3 & 4
