# Paper 2 (Standardized): Models and Methodology

**Status**: IN PREPARATION
**Methodology**: Unified standard - 50 trials per model, corrected trial definition

## All Models (24 total - Cross-domain)

### Medical Domain (13 models)

**Closed (7)**:
| Model | Trials | Status |
|-------|--------|--------|
| GPT-4o | 50 | ✓ Complete |
| GPT-4o-mini | 50 | ✓ Complete |
| GPT-5.2 | 50 | ✓ Complete |
| Claude Haiku | 50 | ✓ Complete |
| Claude Opus | 50 | ✓ Complete |
| Gemini Flash | 50 | ✓ Complete (rerun) |
| Gemini 2.5 Pro | 50 | ✓ Complete |

**Open (6)**:
| Model | Trials | Status |
|-------|--------|--------|
| DeepSeek V3.1 | 50 | ✓ Complete |
| Kimi K2 | 50 | 🔄 In progress (32/50) |
| Llama 4 Maverick | 50 | ✓ Complete |
| Llama 4 Scout | 50 | ✓ Complete |
| Mistral Small 24B | 50 | ✓ Complete |
| Ministral 14B | 50 | ✓ Complete |

### Philosophy Domain (11 models)

**Closed (4)**:
| Model | Trials | Status |
|-------|--------|--------|
| GPT-4o | 50 | ✓ Complete |
| GPT-4o-mini | 50 | ✓ Complete |
| Claude Haiku | 50 | ✓ Complete |
| Gemini Flash | 50 | ✓ Complete |

**Open (7)**:
| Model | Trials | Status |
|-------|--------|--------|
| DeepSeek V3.1 | 50 | ✓ Complete |
| Kimi K2 | 50 | ✓ Complete |
| Llama 4 Maverick | 50 | ✓ Complete |
| Llama 4 Scout | 50 | ✓ Complete |
| Ministral 14B | 50 | ✓ Complete |
| Mistral Small 24B | 50 | ✓ Complete |
| Qwen3 235B | 50 | ✓ Complete |

## Data Scale
- **Total runs**: 2,400 (24 models × 2 domains × 50 trials)
- **Total measurements**: ~72,000 (2,400 runs × 30 positions)
- **Total responses**: ~99,000 (TRUE + COLD + SCRAMBLED conditions)

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
