# Paper 5 (Safety Taxonomy): Models and Dataset

**Status**: DRAFT COMPLETE
**Role**: Extension of Paper 4 — Deployment safety framework
**Dataset**: Paper 2 medical subset with P30 accuracy verification

## Models (8 total - Medical domain)

### Open-Source (7 models via Together AI)

| Model | Vendor | Parameters | Var_Ratio | Accuracy | Class |
|-------|--------|-----------|-----------|----------|-------|
| DeepSeek V3.1 | DeepSeek | 671B (37B active) | ~1.07 | High | IDEAL |
| Kimi K2 | Moonshot | ~1T (32B active) | ~1.01 | High | IDEAL |
| Ministral 14B | Mistral | 14B | ~1.08 | High | IDEAL |
| Mistral Small 24B | Mistral | 24B | ~0.99 | High | IDEAL |
| Qwen3 235B | Alibaba | 235B (22B active) | ~1.33 | 95% | RICH |
| Llama 4 Maverick | Meta | 400B (17B active) | ~2.64 | Medium | DANGEROUS |
| Llama 4 Scout | Meta | 109B (17B active) | ~7.46 | Low | DANGEROUS |

### Closed-Source (1 model)

| Model | Vendor | Var_Ratio | Accuracy | Class |
|-------|--------|-----------|----------|-------|
| Gemini Flash | Google | ~0.6 | 16% | EMPTY |

## Data Source
- Model responses: `/data/medical/` (shared with Paper 2)
- Accuracy verification: `/data/paper5/cross_model_p30_accuracy.json`
- P30 summary: `/data/paper5/cross_model_p30_summary.csv`

## Why This Subset?
Paper 5 focuses on medical P30 (task enablement position) where:
- Full clinical summaries are generated
- Safety-critical divergence is most pronounced
- Accuracy can be verified against clinical ground truth (STEMI case)

## Key Findings
1. Four distinct behavioral classes emerge from Var_Ratio × Accuracy
2. Standard accuracy benchmarks miss the EMPTY class (low detail despite convergent outputs)
3. Llama models show extreme divergence at the most safety-critical position
