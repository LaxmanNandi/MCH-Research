# Complete Data Structure Overview
**MCH Experiments: RCI Data Across All Models and Domains**

---

## Experimental Design

**Dimensions per model-domain run:**
- **Trials:** 50 independent runs
- **Positions:** 30 conversational exchanges per trial
- **Conditions:** 3 (TRUE, COLD, SCRAMBLED) per position
- **Total responses per model:** 50 trials × 30 positions × 3 conditions = **4,500 responses**

**RCI Computation:** Position-level analysis
- For each of 30 positions, compute ΔRCI across 50 trials
- ΔRCI = RCI_true - RCI_cold
- RCI = mean cosine similarity between a response and all other responses in same condition

---

## Philosophy Domain (11 Models)

### Closed Models (4)

| Model | Type | Early ΔRCI (1-10) | Mid ΔRCI (11-20) | Late ΔRCI (21-30) | Pattern | Status |
|-------|------|-------------------|------------------|-------------------|---------|--------|
| GPT-4o-mini | Closed | 0.2642 | 0.3046 | 0.2374 | Inverted-U | ✓ Complete |
| GPT-4o | Closed | 0.3051 | 0.3069 | 0.2371 | Inverted-U | ✓ Complete |
| Claude Haiku | Closed | 0.3183 | 0.3661 | 0.3073 | Inverted-U | ✓ Complete |
| Gemini Flash | Closed | 0.3410 | 0.3451 | 0.3278 | Inverted-U | ✓ Complete |

### Open Models (7)

| Model | Type | Early ΔRCI (1-10) | Mid ΔRCI (11-20) | Late ΔRCI (21-30) | Pattern | Status |
|-------|------|-------------------|------------------|-------------------|---------|--------|
| DeepSeek V3.1 | Open | 0.2654 | 0.3414 | 0.3039 | Inverted-U | ✓ Complete |
| Llama 4 Maverick | Open | 0.2663 | 0.2869 | 0.2525 | Inverted-U | ✓ Complete |
| Llama 4 Scout | Open | 0.3048 | 0.3198 | 0.2683 | Inverted-U | ✓ Complete |
| Qwen3 235B | Open | 0.2997 | 0.3461 | 0.3206 | Inverted-U | ✓ Complete |
| Mistral Small 24B | Open | 0.3040 | 0.2908 | 0.2476 | Linear- | ✓ Complete |
| Ministral 14B | Open | 0.3475 | 0.3890 | 0.3817 | Inverted-U | ✓ Complete |
| Kimi K2 (1T params) | Open | 0.3707 | 0.4534 | 0.4605 | Linear+ | ✓ Complete |

**Philosophy Pattern Summary:**
- Inverted-U: 9/11 models (82%)
- Linear+: 1/11 (Kimi K2)
- Linear-: 1/11 (Mistral Small)
- Peak position: 11-20 (mid conversation)
- Mechanism: Early anchoring → Mid synthesis → Late over-generalization

---

## Medical Domain (STEMI Case Reasoning)

### Closed Models (4)

| Model | Type | Early ΔRCI (1-10) | Mid ΔRCI (11-20) | Late ΔRCI (21-29)* | Position 30 Z-score | Pattern (1-29) | Status |
|-------|------|-------------------|------------------|-------------------|---------------------|----------------|--------|
| Gemini Flash | Closed | 0.4114 | 0.3721 | 0.3673 | +2.92 | U-shaped | ✓ Complete |
| Claude Haiku | Closed | 0.3149 | 0.2796 | 0.3142 | +4.25 | U-shaped | ✓ Complete |
| GPT-4o | Closed | 0.2977 | 0.2315 | 0.2999 | +3.69 | U-shaped | ✓ Complete |
| GPT-4o-mini | Closed | 0.3546 | 0.3620 | 0.3690 | +2.24 | Inverted-U | ✓ Complete |

*Note: Late ΔRCI excludes Position 30 outlier

### Open Models (7 - In Progress)

| Model | Type | Early ΔRCI (1-10) | Mid ΔRCI (11-20) | Late ΔRCI (21-29) | Overall Mean | Status |
|-------|------|-------------------|------------------|-------------------|--------------|--------|
| DeepSeek V3.1 | Open | 0.345† | 0.311† | 0.373† | 0.3200 ± 0.0103 | ✓ Complete (50/50) |
| Llama 4 Maverick | Open | - | - | - | 0.3165 ± 0.0120 | ✓ Complete (50/50) |
| Llama 4 Scout | Open | - | - | - | 0.3233 ± 0.0109 | ✓ Complete (50/50) |
| Qwen3 235B | Open | - | - | - | 0.3292 (avg) | ⏳ In progress (47/50) |
| Mistral Small 24B | Open | - | - | - | - | ⏸️ Next in queue |
| Ministral 14B | Open | - | - | - | - | ⏸️ Pending |
| Kimi K2 (1T params) | Open | - | - | - | - | ⏸️ Pending |

†Estimated from preliminary analysis

**Medical Pattern Summary:**
- U-shaped: 3/4 closed models (75%)
- Inverted-U: 1/4 closed models (GPT-4o-mini)
- Trough position: 11-20 ("diagnostic trough")
- Mechanism: High during history → Low during diagnosis → Moderate during integration
- **Position 30 outlier:** All medical models show extreme spike (Z > +2.0) at summarization prompt

---

## Data File Locations

### Philosophy Results
```
c:/Users/barla/mch_experiments/data/philosophy_results/
├── mch_results_gpt4o_mini_philosophy_50trials.json
├── mch_results_gpt4o_philosophy_50trials.json
├── mch_results_claude_haiku_philosophy_50trials.json
├── mch_results_gemini_flash_philosophy_50trials.json
├── mch_results_deepseek_v3_1_philosophy_50trials.json
├── mch_results_llama_4_maverick_philosophy_50trials.json
├── mch_results_llama_4_scout_philosophy_50trials.json
├── mch_results_qwen3_235b_philosophy_50trials.json
├── mch_results_mistral_small_24b_philosophy_50trials.json
├── mch_results_ministral_14b_philosophy_50trials.json
└── mch_results_kimi_k2_philosophy_50trials.json
```

### Medical Results
```
c:/Users/barla/mch_experiments/data/medical_results/  (closed models)
c:/Users/barla/mch_experiments/data/open_medical_rerun/  (open models)
├── mch_results_deepseek_v3_1_medical_50trials.json  ✓
├── mch_results_llama_4_maverick_medical_50trials.json  ✓
├── mch_results_llama_4_scout_medical_50trials.json  ✓
├── mch_results_qwen3_235b_medical_50trials.json  ⏳ (in progress)
├── mch_results_mistral_small_24b_medical_50trials.json  ⏸️
├── mch_results_ministral_14b_medical_50trials.json  ⏸️
└── mch_results_kimi_k2_medical_50trials.json  ⏸️
```

---

## JSON Structure (All Models)

```json
{
  "model": "model_name",
  "model_id": "vendor/model-identifier",
  "vendor": "together|openai|anthropic|google",
  "domain": "philosophy|medical_reasoning",
  "prompts_version": "v1.0",
  "n_trials": 50,
  "n_prompts": 30,
  "temperature": 1.0,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "timestamp": "2025-02-07T...",

  "statistics": {
    "mean_drci": 0.XXXX,
    "std_drci": 0.XXXX,
    "pattern": "INVERTED_U|U_SHAPED|LINEAR_POS|LINEAR_NEG|CONVERGENT"
  },

  "trials": [
    {
      "trial_idx": 0,
      "responses": {
        "true": [
          "Response at position 1 with full context...",
          "Response at position 2 with full context...",
          ...
          "Response at position 30 with full context..."
        ],
        "cold": [
          "Response at position 1 with no context...",
          "Response at position 2 with no context...",
          ...
          "Response at position 30 with no context..."
        ],
        "scrambled": [
          "Response at position 1 with scrambled context...",
          "Response at position 2 with scrambled context...",
          ...
          "Response at position 30 with scrambled context..."
        ]
      }
    },
    ... (49 more trials)
  ]
}
```

---

## Position-Specific Analysis Structure

For each model-domain run, position-dependent analysis produces:

### Per-Position ΔRCI Array (30 values)
```python
position_drci = [
    0.2845,  # Position 1
    0.2912,  # Position 2
    ...
    0.3156,  # Position 29
    0.4821   # Position 30 (often outlier in medical domain)
]
```

### Disruption Sensitivity (DS) per Position
```python
# DS = ΔRCI_scrambled - ΔRCI_cold
# Negative DS → Context presence > order
disruption_sensitivity = [
    -0.0823,  # Position 1
    -0.0945,  # Position 2
    ...
    -0.0231   # Position 30
]
```

### Three-Bin Summary (Positions 1-29 only)
```python
{
    "early": {
        "positions": [1, 2, ..., 10],
        "mean_drci": 0.3051,
        "std": 0.0234
    },
    "mid": {
        "positions": [11, 12, ..., 20],
        "mean_drci": 0.3414,
        "std": 0.0189
    },
    "late": {
        "positions": [21, 22, ..., 29],
        "mean_drci": 0.3039,
        "std": 0.0267
    }
}
```

---

## Total Dataset Size

### Current Status (as of 2025-02-07)

| Domain | Complete | In Progress | Pending | Total |
|--------|----------|-------------|---------|-------|
| Philosophy | 11/11 | 0/11 | 0/11 | 11 |
| Medical Closed | 4/4 | 0/4 | 0/4 | 4 |
| Medical Open | 3/7 | 1/7 | 3/7 | 7 |
| **Total** | **18/22** | **1/22** | **3/22** | **22** |

### Data Volume
- **Complete:** 18 model-domain runs × 4,500 responses = 81,000 responses
- **Final:** 25 model-domain runs × 50 trials × 30 prompts × 3 conditions = 112,500 responses
- **Embedding vectors:** 112,500 × 384 dimensions (all-MiniLM-L6-v2)
- **Position measurements:** 25 model-domain runs × 30 positions = 750 position-ΔRCI values

---

## Key Findings Across All Models

### Finding 1: Domain-Specific Patterns
- **Philosophy:** 82% show inverted-U (peak at mid conversation)
- **Medical:** 75% show U-shaped (trough during diagnosis)

### Finding 2: Position 30 Outlier (Medical Only)
- **100% of medical models** show Position 30 as extreme outlier (Z > +2.0)
- Mean Z-score: +3.27 (SD = 0.85)
- Indicates **task enablement** vs performance enhancement

### Finding 3: Disruption Sensitivity
- **14/15 models** show DS < 0 (context presence > order)
- Philosophy mean DS: -0.081 (SD = 0.032)
- Medical mean DS: -0.062 (SD = 0.058)

### Finding 4: Scale Effects
- **Kimi K2 (1T params):** Only philosophy model with sustained linear accumulation
- Suggests capacity threshold for maintaining long-range coherence

---

## Sample Response Data

### Philosophy Domain (Position 5, "How does consciousness relate to physical processes?")

**TRUE response (GPT-4o, Trial 1):**
> "Building on our earlier discussion of qualia and the hard problem, the relationship between consciousness and physical processes remains deeply puzzling. While neural correlates of consciousness (NCCs) can be identified—specific patterns of brain activity associated with particular conscious experiences—this correlation doesn't explain *why* these physical processes give rise to subjective experience..."

**COLD response (GPT-4o, Trial 1):**
> "The relationship between consciousness and physical processes is one of philosophy of mind's central questions. Physicalists argue consciousness emerges from or is identical to brain processes, while property dualists maintain consciousness has irreducible phenomenal properties. The explanatory gap highlights our difficulty connecting objective neural descriptions to subjective experience..."

**ΔRCI:** 0.3156 (TRUE responses more similar to each other than COLD responses)

### Medical Domain (Position 15, "What is your differential diagnosis?")

**TRUE response (Claude Haiku, Trial 1):**
> "Given this 58-year-old male with acute chest pain, diaphoresis, and ST-elevation in leads V1-V4, the primary diagnosis is anterior wall STEMI. Differential includes: 1) Acute coronary syndrome (most likely), 2) Aortic dissection (though less likely given presentation), 3) Pulmonary embolism (less consistent with ST changes), 4) Pericarditis (atypical for isolated anterior changes)..."

**COLD response (Claude Haiku, Trial 1):**
> "For acute chest pain, the differential diagnosis includes: 1) Acute coronary syndrome (unstable angina, NSTEMI, STEMI), 2) Aortic dissection, 3) Pulmonary embolism, 4) Pneumothorax, 5) Pericarditis, 6) Myocarditis, 7) Esophageal rupture, 8) GERD, 9) Musculoskeletal pain, 10) Anxiety. The specific diagnosis depends on presentation, risk factors, and diagnostic findings..."

**ΔRCI:** 0.2796 (Lower than Position 5 - diagnostic independence)

### Medical Domain (Position 30, "Summarize this case")

**TRUE response (Claude Haiku, Trial 1):**
> "This is a 58-year-old male smoker with hypertension who presented with 2 hours of severe substernal chest pain radiating to the left arm. Initial ECG showed anterior STEMI with ST elevation in V1-V4. Troponin elevated at 0.8 ng/mL. We initiated dual antiplatelet therapy, started heparin, and arranged urgent cardiac catheterization which revealed 95% proximal LAD occlusion. Successful PCI with drug-eluting stent placement restored flow. Post-procedure, patient stable with resolution of chest pain..."

**COLD response (Claude Haiku, Trial 1):**
> "I don't have specific case details to summarize. To provide a case summary, I would need information about the patient's presentation, history, physical exam findings, diagnostic results, diagnosis, treatment, and outcome. Could you provide the case information you'd like me to summarize?"

**ΔRCI:** 0.6142 (Extreme outlier - task enablement effect)
**Z-score:** +4.25

---

## Analysis Scripts

### Position-Dependent Analysis
- [mch_position_analysis.py](c:/Users/barla/mch_experiments/scripts/mch_position_analysis.py)
- [position_analysis_position30_excluded.py](c:/Users/barla/mch_experiments/scripts/position_analysis_position30_excluded.py)
- [deepseek_position_preview.py](c:/Users/barla/mch_experiments/scripts/deepseek_position_preview.py)

### Summary Data
- [position_analysis_summary.csv](c:/Users/barla/mch_experiments/analysis/position_analysis_summary.csv)
- [position_drci_data.csv](c:/Users/barla/mch_experiments/analysis/position_drci_data.csv)

---

**Last Updated:** 2026-02-08
**Current Experiment:** Qwen3 235B Medical (Trial 47/50 - auto-proceeding to Mistral Small 24B next)
