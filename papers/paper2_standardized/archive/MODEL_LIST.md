# Paper 2 (Standardized): Models and Methodology

**Status**: ACCEPTED — Preprints.org (ID: 198770, February 12, 2026)
**Methodology**: Unified standard - 50 trials per model, corrected trial definition

## All Models (14 unique, 25 model-domain runs)

### Medical Domain (13 models)

**Closed (6)**:
| Model | Vendor | Trials | Status |
|-------|--------|--------|--------|
| GPT-4o | OpenAI | 50 | Complete |
| GPT-4o-mini | OpenAI | 50 | Complete |
| GPT-5.2 | OpenAI | 50 | Complete |
| Claude Haiku | Anthropic | 50 | Complete |
| Claude Opus | Anthropic | 50 | Complete (43 recovered + 7 re-run, metrics only) |
| Gemini Flash | Google | 50 | Complete |

**Open (7)**:
| Model | Vendor | Trials | Status |
|-------|--------|--------|--------|
| DeepSeek V3.1 | DeepSeek | 50 | Complete |
| Kimi K2 | Moonshot | 50 | Complete |
| Llama 4 Maverick | Meta | 50 | Complete |
| Llama 4 Scout | Meta | 50 | Complete |
| Qwen3 235B | Alibaba | 50 | Complete |
| Mistral Small 24B | Mistral | 50 | Complete |
| Ministral 14B | Mistral | 50 | Complete |

### Philosophy Domain (12 models)

**Closed (5)**:
| Model | Vendor | Trials | Status |
|-------|--------|--------|--------|
| GPT-4o | OpenAI | 50 | Complete |
| GPT-4o-mini | OpenAI | 50 | Complete |
| GPT-5.2 | OpenAI | 50 | Complete (first 50 of legacy 100) |
| Claude Haiku | Anthropic | 50 | Complete |
| Gemini Flash | Google | 50 | Complete |

**Open (7)**:
| Model | Vendor | Trials | Status |
|-------|--------|--------|--------|
| DeepSeek V3.1 | DeepSeek | 50 | Complete |
| Kimi K2 | Moonshot | 50 | Complete |
| Llama 4 Maverick | Meta | 50 | Complete |
| Llama 4 Scout | Meta | 50 | Complete |
| Qwen3 235B | Alibaba | 50 | Complete |
| Mistral Small 24B | Mistral | 50 | Complete |
| Ministral 14B | Mistral | 50 | Complete |

## Data Scale
- **Unique models**: 14
- **Model-domain runs**: 25 (13 medical + 12 philosophy)
- **Trials per run**: 50
- **Total trials**: 1,250
- **Total responses**: 112,500 (1,250 trials x 3 conditions x 30 prompts)

## Response Text Availability
- **With text** (18 runs): 5 phil closed + 7 med open + 5 med closed (excl. Claude Opus) + Gemini Flash
- **Metrics only** (7 runs): 7 phil open (metrics-only JSON files)
- **Recovered** (1 run): Claude Opus medical (metrics recovered, no response text)

## Methodology Upgrade from Paper 1
- **Unified standard**: Each trial = independent TRUE + COLD + SCRAMBLED run
- **Fixed trial definition**: Corrected measurement errors from Paper 1
- **Cross-domain**: Medical (closed-goal) + Philosophy (open-goal)
- **Architectural balance**: Open + closed models in both domains

## Data Location
All data stored in `/data/` directory (single source of truth):
- `data/medical/open_models/`
- `data/medical/closed_models/`
- `data/philosophy/open_models/`
- `data/philosophy/closed_models/`
