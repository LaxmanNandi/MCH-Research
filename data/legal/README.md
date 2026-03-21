# Legal Domain Experiment Data

Third domain extension of the MCH conservation framework (Paper 6).
Employment law dispute with whistleblower retaliation — 30 positions, 50 trials per model.

## Models

| Model | Trials | dRCI | Var_Ratio | K | Status |
|-------|--------|------|-----------|------|--------|
| DeepSeek V3.1 | 50/50 | 0.276 | 1.225 | 0.338 | COMPLETE |
| Llama 4 Maverick | 50/50 | 0.209 | 1.412 | 0.295 | COMPLETE |
| Qwen3 235B | 50/50 | 0.265 | 1.796 | 0.476 | COMPLETE |
| Mistral Small 24B | 50/50 | 0.252 | 1.338 | 0.338 | COMPLETE |
| Llama 3.3 70B Turbo | 50/50 | 0.206 | 1.428 | 0.294 | COMPLETE |
| Kimi K2.5 | 50/50 | 0.509 | 1.633 | 0.831 | EXCLUDED — COLD refusal + 21% empty |
| Ministral 14B | 0/50 | — | — | — | EMPTY — non-functional |

## Key Results (N=5 valid)

- **K(Legal) = 0.348** (range 0.294–0.476)
- **Information hierarchy**: TRUE > SCRAMBLED > COLD confirmed in all 5 models
- **No entanglement**: All ΔRCI~VRI correlations non-significant
- **All convergent**: Exploration arc < 3.0 for all models
- **Content dominates**: VR_Order < 0.3 for all models — order constrains variance in legal domain

## Conservation Law Comparison

| Domain | K | N |
|--------|------|---|
| Medical | 0.429 | 8 |
| Legal | 0.348 | 5 |
| Philosophy | 0.301 | 6 |

## File Structure

```
legal/
└── open_models/
    ├── mch_results_deepseek_v3_1_legal_50trials.json
    ├── mch_results_llama_4_maverick_legal_50trials.json
    ├── mch_results_qwen3_235b_legal_50trials.json
    ├── mch_results_mistral_small_24b_legal_50trials.json
    ├── mch_results_llama_3_3_70b_turbo_legal_50trials.json
    ├── mch_results_kimi_k2_5_legal_50trials.json (excluded)
    └── mch_results_ministral_14b_legal_50trials.json (empty)
```
