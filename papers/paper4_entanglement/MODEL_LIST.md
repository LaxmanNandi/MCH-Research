# Paper 4 (Entanglement Mechanism): Models and Dataset

**Status**: DRAFT COMPLETE
**Role**: Extension of Paper 2 - Information-theoretic mechanism analysis
**Dataset**: Paper 2 subset (11 models with response text)

## Models (11 total - Cross-domain subset)

### Philosophy Domain (4 models - Closed only)

| Model | Domain | Trials | Embedding Variance | Status |
|-------|--------|--------|--------------------|--------|
| GPT-4o | Philosophy | 50 | ✓ Computed | Complete |
| GPT-4o-mini | Philosophy | 50 | ✓ Computed | Complete |
| Claude Haiku | Philosophy | 50 | ✓ Computed | Complete |
| Gemini Flash | Philosophy | 50 | ✓ Computed | Complete |

### Medical Domain (7 models - Mixed)

**Open (6)**:
| Model | Domain | Trials | Embedding Variance | Status |
|-------|--------|--------|--------------------|--------|
| DeepSeek V3.1 | Medical | 50 | ✓ Computed | Complete |
| Llama 4 Maverick | Medical | 50 | ✓ Computed | Complete |
| Llama 4 Scout | Medical | 50 | ✓ Computed | Complete |
| Mistral Small 24B | Medical | 50 | ✓ Computed | Complete |
| Ministral 14B | Medical | 50 | ✓ Computed | Complete |
| Qwen3 235B | Medical | 50 | ✓ Computed | Complete |

**Closed (1)**:
| Model | Domain | Trials | Embedding Variance | Status |
|-------|--------|--------|--------------------|--------|
| Gemini Flash | Medical | 50 | ✓ Computed | Complete (processed separately) |

## Data Source
All data comes from Paper 2's standardized dataset.
**Location**: `/data/` (shared with Paper 2, no duplication)

## Why This Subset?
Response text is required for:
- Computing embedding variances (Var_TRUE, Var_COLD)
- Variance ratio analysis (Var_TRUE / Var_COLD)
- Mutual information proxy (1 - Var_Ratio)
- Entanglement validation (ΔRCI ~ MI_proxy)

Only 11 of Paper 2's 24 models have complete response text preserved.

## Data Points
- **Total**: 330 model-position points (11 models × 30 positions)
- **Correlation**: r = 0.76, p = 1.5×10⁻⁶²

## Key Findings
1. **Entanglement validation**: ΔRCI ~ MI_proxy (r=0.76)
2. **Bidirectional regimes**:
   - Convergent: Var_Ratio < 1 (context reduces variance)
   - Divergent: Var_Ratio > 1 (context increases variance)
3. **Llama safety anomaly**: Extreme divergence at P30 (Var_Ratio 2.6-7.5)
4. **Domain differences**: Medical variance-increasing (1.23), Philosophy neutral (1.01)

## Figures
All Paper 4 figures stored in `papers/paper4_entanglement/figures/`
- Main figures: Entanglement validation, multi-panel analysis, safety anomaly
- Supplementary: Gaussian verification, trial convergence, model comparison
