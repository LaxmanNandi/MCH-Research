# Data Validation Report

## Complete (50 trials, usable for Paper 3/4)

| Model | Domain | File |
|-------|--------|------|
| claude_haiku | medical_reasoning | data\medical_results\mch_results_claude_haiku_medical_50trials.json |
| claude_opus | medical_reasoning | data\medical_results\mch_results_claude_opus_medical_50trials.json |
| deepseek_v3_1 | medical_reasoning | data\open_medical_rerun\mch_results_deepseek_v3_1_medical_50trials.json |
| gemini_flash | medical_reasoning | data\gemini_flash_medical_rerun\mch_results_gemini_flash_medical_50trials.json |
| gemini_flash | medical_reasoning | data\medical_results\mch_results_gemini_flash_medical_50trials.json |
| gpt4o | medical_reasoning | data\medical_results\mch_results_gpt4o_medical_50trials.json |
| gpt4o_mini | medical_reasoning | data\medical_results\mch_results_gpt4o_mini_medical_50trials.json |
| gpt4o_mini_rerun | medical_reasoning | data\medical_results\mch_results_gpt4o_mini_rerun_medical_50trials.json |
| gpt_5_2 | medical_reasoning | data\medical_results\mch_results_gpt_5_2_medical_50trials.json |
| llama_4_maverick | medical_reasoning | data\open_medical_rerun\mch_results_llama_4_maverick_medical_50trials.json |
| llama_4_scout | medical_reasoning | data\open_medical_rerun\mch_results_llama_4_scout_medical_50trials.json |
| mistral_small_24b | medical_reasoning | data\open_medical_rerun\mch_results_mistral_small_24b_medical_50trials.json |
| qwen3_235b | medical_reasoning | data\open_medical_rerun\mch_results_qwen3_235b_medical_50trials.json |
| claude_haiku | philosophy | data\closed_model_philosophy_rerun\mch_results_claude_haiku_philosophy_50trials.json |
| deepseek_v3_1 | philosophy | data\open_model_results\mch_results_deepseek_v3_1_philosophy_50trials.json |
| gemini_flash | philosophy | data\closed_model_philosophy_rerun\mch_results_gemini_flash_philosophy_50trials.json |
| gpt4o | philosophy | data\closed_model_philosophy_rerun\mch_results_gpt4o_philosophy_50trials.json |
| gpt4o_mini | philosophy | data\closed_model_philosophy_rerun\mch_results_gpt4o_mini_philosophy_50trials.json |
| kimi_k2 | philosophy | data\open_model_results\mch_results_kimi_k2_philosophy_50trials.json |
| llama_4_maverick | philosophy | data\open_model_results\mch_results_llama_4_maverick_philosophy_50trials.json |
| llama_4_scout | philosophy | data\open_model_results\mch_results_llama_4_scout_philosophy_50trials.json |
| ministral_14b | philosophy | data\open_model_results\mch_results_ministral_14b_philosophy_50trials.json |
| mistral_small_24b | philosophy | data\open_model_results\mch_results_mistral_small_24b_philosophy_50trials.json |
| qwen3_235b | philosophy | data\open_model_results\mch_results_qwen3_235b_philosophy_50trials.json |

## In Progress (Open Medical Rerun)

| Model | Status | Notes |
|-------|--------|-------|
| DeepSeek V3.1 | 50/50 complete | Rerun complete |
| Qwen3 235B | 50/50 complete | Rerun complete |
| Llama 4 Maverick | 50/50 complete | Rerun complete |
| Llama 4 Scout | 50/50 complete | Rerun complete |
| Ministral 14B | 10/50 in progress | Checkpoint saved |
| Kimi K2 | 0/50 queued | Pending |

## Excluded

| File | Reason |
|------|--------|
| data/medical_results/gemini_pro_safety_blocked.json | Safety blocked (0 trials) |
| data/open_medical_rerun/mch_results_mistral_small_24b_medical_checkpoint.json | Prompt mismatch (checkpoint; superseded by rerun) |
| data/medical_results/mch_results_claude_opus_medical_43trials_recovered.json | Recovered partial (43 trials); superseded by 50-trial file |

## Note

Open medical rerun ETA: 2-3 days (remaining queued runs)
Mistral Small 24B medical rerun completed (50 trials, corrected prompts).
