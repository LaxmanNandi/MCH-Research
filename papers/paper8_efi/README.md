# Paper 8: Encoding Fidelity and Coherent Misalignment

**Title:** Encoding Fidelity and Coherent Misalignment: Why Shannon's Channel Model Breaks for Non-English Clinical AI

**Author:** Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

**Date:** March 20, 2026

**Status:** Submitted to Preprints.org (Preprints ID: 204266, March 20, 2026)

---

## Folder Structure

```
papers/paper8_efi/
├── paper8.tex                  # Main LaTeX manuscript
├── README.md                   # This file
├── drafts/
│   ├── deepseek_outline_prompt.md    # Original prompt sent to DeepSeek for outline
│   └── paper8_outline_refined.md     # Refined outline with verified citations
├── scripts/
│   ├── paper8_efi_test.py            # Exp 1: EFI battery (Kannada + Tamil, 15 sentences)
│   ├── paper8_efi_hindi.py           # Exp 2: Hindi comparison
│   ├── paper8_efi_robustness.py      # Exp 3: MPNet cross-encoder robustness
│   ├── paper8_llm_confusion.py       # Exp 4: LLM language ID + translation (original)
│   ├── paper8_llm_multitrial.py      # Exp 5: LLM multi-trial variance (original, 5 trials)
│   ├── run_exp4_5_with_responses.py  # Exp 4-5 rerun with full response saving
│   ├── run_exp5_15trials.py          # Exp 5: 15-trial variance (final, 600 API calls)
│   └── save_all_results.py           # Utility to re-run Exp 1-3 and save JSON
├── results/
│   ├── exp1_2_efi_battery.json       # EFI scores: Kn=0.099, Ta=0.069, Hi=0.076
│   ├── exp1_efi_battery_output.txt   # Raw terminal output from Exp 1
│   ├── exp2_hindi_output.txt         # Raw terminal output from Exp 2
│   ├── exp3_mpnet_robustness.json    # MPNet: Kn=0.125, Ta=0.106
│   ├── exp3_robustness_output.txt    # Raw terminal output from Exp 3
│   ├── exp4_language_translation_accuracy.json  # LLM language ID + translation (with responses)
│   ├── exp5_multitrial_variance.json # 5-trial variance (with responses, 2 models)
│   └── exp5_15trials_variance.json   # 15-trial variance (FINAL, 600 responses, 1.4MB)
└── figures/                          # (to be generated)
```

## Experiments Summary

| Exp | Description | Key Finding |
|-----|-------------|-------------|
| 1 | EFI clinical battery (Kn, Ta vs En) | EFI = 0.069-0.099, all p < 1e-13 vs English |
| 2 | Hindi comparison | Hindi EFI = 0.076, no significant difference from Dravidian |
| 3 | MPNet robustness check | Confirmed with second embedding model (r > 0.99) |
| 4 | LLM language ID + translation | DeepSeek/Mistral confusion on Dravidian medical terms |
| 5 | Clinical variance (15 trials) | Kannada VR=1.72-2.05x (p<0.05 both models) |

## Key Results (Exp 5, 15 trials, 2 models)

### Variance Ratios
| Language | DeepSeek VR | Mistral VR | Replicated? |
|----------|-------------|------------|-------------|
| Kannada  | 1.72x (p=0.048) | 2.05x (p=0.008) | YES |
| Tamil    | 1.06x (ns)      | 1.63x (p=0.016) | Partial |
| Hindi    | 1.38x (ns)      | 0.91x (ns)      | No amplification |

### Code-Switching (% of responses starting in English)
| Model | Kannada | Tamil | Hindi |
|-------|---------|-------|-------|
| DeepSeek | 84% | 40% | 79% |
| Mistral  | 45% | 48% | 59% |

### Novel Findings
1. **Coherent misalignment**: Models produce medically correct but linguistically wrong responses
2. **Dravidian-specific variance amplification**: Kannada/Tamil amplify, Hindi does not
3. **Language identity confusion**: DeepSeek responds to Kannada in English 84% of the time
4. **Orthographic corruption**: Mistral misspells Kannada medical terms (e.g., cough)
5. **Scenario-dependent switching**: Complex scenarios trigger different language choices
6. **K-orthogonal truth**: Conservation law holds regardless of encoding fidelity

## Original Scripts Location
Original experiment scripts also exist at: `scripts/analysis/paper8_*.py`
These are the source scripts; copies in `scripts/` folder are identical.

## Dependencies
- sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- scipy, numpy
- together (Together.ai API for Exp 4-5)
- API key: TOGETHER_API_KEY environment variable

## Cost
- Together.ai API: ~$5-10 for all Exp 4-5 runs combined
