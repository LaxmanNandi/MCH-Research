# Paper 4 Figure Data Verification

## Main Figures (Entanglement Analysis)

### Data Source: `entanglement_position_data.csv`
- **Models**: 11 (verified ✓)
- **Data points**: 330 (11 models × 30 positions)
- **Breakdown**:
  - Philosophy: 4 closed models (GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash)
  - Medical: 7 models (6 open + Gemini Flash closed)

### Figures Using This Data:
1. **Figure 1** (entanglement_validation.png) - 11 models ✓
2. **Figure 2** (fig4_entanglement_multipanel.png) - 11 models ✓
3. **Figure 3** (fig7_llama_safety_anomaly.png) - 2 Llama models (subset) ✓
4. **Figure 4** (fig5_independence_rci_var.png) - 11 models ✓

**Status**: All main figures correctly use 11-model entanglement dataset

---

## Supplementary Figures

### Data Source: `trial_level_drci.csv` (filtered to trial ≤ 50, cleaned)
- **Unique model names**: 14
- **Model-domain runs**: 22 (some models tested in both domains)
- **Breakdown**:
  - Medical: 10 model runs
  - Philosophy: 12 model runs
  - Overlap: 8 models tested in BOTH domains

### The 14 Unique Models:
1. claude_haiku
2. claude_opus
3. deepseek_v3_1
4. gemini_flash
5. gpt4o
6. gpt4o_mini
7. gpt4o_mini_rerun
8. gpt_5_2
9. kimi_k2
10. llama_4_maverick
11. llama_4_scout
12. ministral_14b
13. mistral_small_24b
14. qwen3_235b

### Models in BOTH Domains (8):
- claude_haiku, deepseek_v3_1, gpt4o, gpt_5_2
- llama_4_maverick, llama_4_scout, mistral_small_24b, qwen3_235b

### Figures Using This Data:
- **Figure S2** (figure8_trial_convergence.png) - 14 unique models across 22 runs ✓
- **Figure S3** (figure9_model_comparison.png) - 14 unique models across 22 runs ✓

**Status**: Supplementary figures correctly use 14-model trial-level dataset

---

## Additional Open Models in Supplementary Figures

Compared to the 11-model entanglement analysis, Figures S2/S3 add:

**Philosophy Open Models (7 added):**
- deepseek_v3_1
- kimi_k2
- llama_4_maverick
- llama_4_scout
- ministral_14b
- mistral_small_24b
- qwen3_235b

**Medical Closed Models (6 added):**
- claude_haiku
- claude_opus
- gpt4o
- gpt4o_mini (+ rerun)
- gpt_5_2

---

## Verification Summary

✓ **Main entanglement figures (1-4)**: Correctly use 11 models with response text
✓ **Supplementary figures (S2-S3)**: Correctly use 14 unique models (22 domain runs) from trial-level data
✓ **Data consistency**: All figures match their documented model counts
✓ **Methodology**: Main analysis requires response text; supplementary only needs trial metrics

**Conclusion**: All figures accurately reflect the data they claim to use.
