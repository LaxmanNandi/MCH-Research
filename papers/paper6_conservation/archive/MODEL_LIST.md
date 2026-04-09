# Paper 6 (Conservation Constraint): Models and Dataset

**Status**: DRAFT COMPLETE
**Role**: Capstone — Conservation constraint across all architectures
**Dataset**: Paper 2 subset with embedding-based Var_Ratio (14 model-domain runs)

## Models (11 unique, 14 model-domain runs)

### Medical Domain (8 runs)

| Model | Vendor | Parameters | ΔRCI | Var_Ratio | Product |
|-------|--------|-----------|------|-----------|---------|
| Gemini Flash | Google | Undisclosed | 0.427 | 1.287 | 0.549 |
| Llama 4 Scout | Meta | 109B (17B active) | 0.323 | 1.610 | 0.521 |
| Qwen3 235B | Alibaba | 235B (22B active) | 0.328 | 1.334 | 0.437 |
| Ministral 14B | Mistral | 14B | 0.391 | 1.080 | 0.423 |
| Kimi K2 | Moonshot | ~1T (32B active) | 0.417 | 1.006 | 0.420 |
| Llama 4 Maverick | Meta | 400B (17B active) | 0.316 | 1.213 | 0.384 |
| Mistral Small 24B | Mistral | 24B | 0.365 | 0.985 | 0.359 |
| DeepSeek V3.1 | DeepSeek | 671B (37B active) | 0.320 | 1.071 | 0.343 |

**Medical K = 0.429 (SD = 0.073, CV = 0.170)**

### Philosophy Domain (6 runs)

| Model | Vendor | Parameters | ΔRCI | Var_Ratio | Product |
|-------|--------|-----------|------|-----------|---------|
| Gemini Flash | Google | Undisclosed | 0.338 | 1.120 | 0.378 |
| Claude Haiku | Anthropic | Undisclosed | 0.331 | 1.012 | 0.334 |
| DeepSeek V3.1 | DeepSeek | 671B (37B active) | 0.302 | 1.034 | 0.312 |
| GPT-4o | OpenAI | Undisclosed | 0.283 | 0.950 | 0.269 |
| GPT-4o Mini | OpenAI | Undisclosed | 0.269 | 0.968 | 0.260 |
| Llama 4 Maverick | Meta | 400B (17B active) | 0.266 | 0.939 | 0.250 |

**Philosophy K = 0.301 (SD = 0.050, CV = 0.166)**

## Dual-Domain Models (3 models)
- DeepSeek V3.1: Med + Phil
- Gemini Flash: Med + Phil
- Llama 4 Maverick: Med + Phil

## Data Source
- Conservation product CSV: `/data/paper6/conservation_product_test.csv`
- MI verification: `/data/paper6/conservation_law_verification/`
- Raw model data: `/data/medical/` and `/data/philosophy/`

## Why 14 Runs (Not 25)?
Paper 2 has 25 model-domain runs across 14 unique models. Paper 6 uses the 14 runs that have embedding-based Var_Ratio computed from saved response text. The remaining 11 runs have metrics-only files without raw embeddings.
