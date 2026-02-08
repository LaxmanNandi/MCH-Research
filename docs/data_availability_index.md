# Data Availability Index (Paper 3/4)

This index summarizes the current data and analysis assets for Paper 3/4 submission prep.

---

## 1. COMPLETED model runs (raw response JSONs)

### Philosophy (50-trial reruns)
- `data/closed_model_philosophy_rerun/mch_results_gpt4o_mini_philosophy_50trials.json`
- `data/closed_model_philosophy_rerun/mch_results_gpt4o_philosophy_50trials.json`
- `data/closed_model_philosophy_rerun/mch_results_claude_haiku_philosophy_50trials.json`
- `data/closed_model_philosophy_rerun/mch_results_gemini_flash_philosophy_50trials.json`
- `data/open_model_results/mch_results_deepseek_v3_1_philosophy_50trials.json`
- `data/open_model_results/mch_results_llama_4_maverick_philosophy_50trials.json`
- `data/open_model_results/mch_results_llama_4_scout_philosophy_50trials.json`
- `data/open_model_results/mch_results_qwen3_235b_philosophy_50trials.json`
- `data/open_model_results/mch_results_mistral_small_24b_philosophy_50trials.json`
- `data/open_model_results/mch_results_ministral_14b_philosophy_50trials.json`
- `data/open_model_results/mch_results_kimi_k2_philosophy_50trials.json`

### Medical (closed models, 50 trials)
- `data/medical_results/mch_results_gpt4o_medical_50trials.json`
- `data/medical_results/mch_results_gpt4o_mini_medical_50trials.json`
- `data/medical_results/mch_results_claude_haiku_medical_50trials.json`
- `data/medical_results/mch_results_gemini_flash_medical_50trials.json`
- `data/medical_results/mch_results_claude_opus_medical_50trials.json`
- `data/medical_results/mch_results_gpt_5_2_medical_50trials.json`

### Medical (open / reruns, 50 trials)
- `data/gemini_flash_medical_rerun/mch_results_gemini_flash_medical_50trials.json`
- `data/open_medical_rerun/mch_results_deepseek_v3_1_medical_50trials.json`
- `data/open_medical_rerun/mch_results_llama_4_maverick_medical_50trials.json`
- `data/open_medical_rerun/mch_results_llama_4_scout_medical_50trials.json`

---

## 2. IN-PROGRESS model runs (expected paths + status)

- `data/open_medical_rerun/mch_results_qwen3_235b_medical_50trials.json` — **in progress** (47/50 trials)
- `data/open_medical_rerun/mch_results_mistral_small_24b_medical_50trials.json` — **queued next**
- `data/open_medical_rerun/mch_results_ministral_14b_medical_50trials.json` — **pending**
- `data/open_medical_rerun/mch_results_kimi_k2_medical_50trials.json` — **pending**

---

## 3. Key analysis outputs (CSVs)

- Entanglement (Paper 4)
  - `analysis/entanglement_position_data.csv`
  - `analysis/entanglement_correlations.csv`
  - `analysis/entanglement_variance_summary.csv`
- Independence tests (Paper 4)
  - `analysis/independence_test_results.csv`
  - `analysis/independence_test_per_model.csv`
  - `analysis/independence_var_ratio_results.csv`
  - `analysis/independence_var_ratio_per_model.csv`
- Position dynamics (Paper 3)
  - `analysis/position_analysis_summary.csv`
  - `analysis/position_drci_data.csv`
  - `analysis/position30_analysis/position30_outlier_analysis.csv`
  - `analysis/position30_analysis/position30_bin_analysis.csv`
  - `analysis/position30_analysis/position30_trend_comparison.csv`
- Type 2 scaling
  - `analysis/type2_scaling_validation.csv`
  - `analysis/type2_scaling_points_TEMPLATE.csv`

---

## 4. Key scripts (analysis + validation)

- `scripts/mch_position_analysis.py` — position-dependent ?RCI and DS extraction
- `scripts/position_analysis_position30_excluded.py` — P30 exclusion and slope changes
- `scripts/paper3_generate_figures.py` — Paper 3 figures
- `scripts/validate/test_entanglement_theory.py` — variance/MI proxy entanglement test
- `scripts/validate/test_independence.py` — ?RCI vs RCI_COLD (invalid) check
- `scripts/validate/test_independence_var_ratio.py` — RCI_COLD vs Var_Ratio independence
- `scripts/validate_type2_scaling.py` — Type 2 scaling check
- `scripts/validate/test_deepseek_simple.py` — DeepSeek V3.1 spot checks
- `scripts/validate/test_deepseek_theory.py` — DeepSeek V3.1 theory validation

---

## Notes
- Embedding files are not stored; embeddings are recomputed by scripts as needed.
- Legacy Paper 1 explorer data are archived separately in `docs/Paper1_Archived_Summary.md`.
