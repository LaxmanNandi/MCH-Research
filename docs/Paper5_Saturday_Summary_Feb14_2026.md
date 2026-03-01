# Paper 5: Saturday Evening Summary
## Safety Taxonomy for Clinical AI Deployment
**Date:** Saturday, February 14, 2026
**Author:** Dr. Laxman M M, MBBS | Primary Health Centre Manchi, Karnataka
**Status:** Verification Report

---

## 1. Paper 5: Safety Taxonomy — Definition

**Core Discovery:** Var_Ratio alone does not predict task completeness (r = -0.21, p = 0.65). The relationship is **categorical, not continuous** — yielding a four-class behavioral taxonomy for clinical deployment.

| Class | Var_Ratio | Accuracy | Models | Deployment |
|-------|-----------|----------|--------|------------|
| **IDEAL** | < 1.2 | High (83-92%) | DeepSeek V3.1, Ministral 14B, Kimi K2, Mistral Small 24B | Deploy |
| **EMPTY** | Low (0.60) | Low (16%) | Gemini Flash | Fix safety filters |
| **DANGEROUS** | > 2.0 | Low (47-55%) | Llama 4 Scout, Llama 4 Maverick | Do not deploy |
| **RICH** | 1.2-2.0 | High (95%) | Qwen3 235B | Investigate |

**Key Insight:** "When Correct Isn't Safe" — accuracy benchmarks alone would miss the DANGEROUS class entirely. Only the Var_Ratio dimension reveals that Llama models produce *unpredictable* outputs at the exact moment clinical tasks require *predictability*.

---

## 2. Cross-Model P30 Accuracy Results

All models tested under identical conditions: Temperature 0.7, max_tokens 1024, 50 trials each, P30 medical summarization prompt with 29-exchange STEMI case history. Accuracy scored against a 16-element clinical rubric.

| Model | Var_Ratio | Mean Acc (raw) | Mean Acc (%) | Std | Range | Perfect | Class |
|-------|-----------|----------------|-------------|-----|-------|---------|-------|
| DeepSeek V3.1 | 0.48 | 13.2/16 | 82.8% | 1.23 | 10-16 | 2/50 | IDEAL |
| Gemini Flash | 0.60 | 2.5/16 | 15.8% | 2.88 | 0-16 | 1/50 | EMPTY |
| Ministral 14B | 0.75 | 14.4/16 | 90.2% | 1.71 | 7-16 | 10/50 | IDEAL |
| Kimi K2 | 0.97 | 14.7/16 | 91.9% | 1.04 | 11-16 | 13/50 | IDEAL |
| Mistral Small 24B | 1.02 | 13.2/16 | 82.6% | 1.25 | 10-16 | 1/50 | IDEAL |
| Qwen3 235B | 1.45 | 15.2/16 | 95.2% | 0.86 | 13-16 | 23/50 | RICH |
| Llama 4 Maverick | 2.64 | 7.5/16 | 46.8% | 1.33 | 5-11 | 0/50 | DANGEROUS |
| Llama 4 Scout | 7.46 | 8.9/16 | 55.4% | 1.78 | 5-14 | 0/50 | DANGEROUS |

---

## 3. Key Statistical Findings

### Correlation Analysis
| Test | Pearson r | p-value | Spearman rho | p-value |
|------|-----------|---------|-------------|---------|
| Var_Ratio vs Accuracy | -0.21 | 0.65 | 0.00 | 1.00 |
| Var_Ratio vs Std | -0.01 | 0.99 | -0.07 | 0.88 |
| Var_Ratio vs Range | -0.12 | 0.79 | -0.13 | 0.79 |

**Verdict:** No linear relationship. The structure is categorical.

### U-Shape Hypothesis Test
| Dataset | Quadratic R-squared | F-test p | Significant? |
|---------|-------------------|----------|------------|
| Full (N=8) | 0.11 | 0.72 | No |
| Excl. Gemini (N=7) | 0.14 | 0.89 | No |

### ESI Validation
| Model | Var_Ratio | ESI | Accuracy | Class | ESI Correct? |
|-------|-----------|-----|----------|-------|-------------|
| DeepSeek V3.1 | 0.48 | 1.91 | 82.8% | IDEAL | Yes |
| Gemini Flash | 0.60 | 2.52 | 15.8% | EMPTY | **No** (ESI says safe) |
| Kimi K2 | 0.97 | 33.3 | 91.9% | IDEAL | Yes |
| Llama 4 Maverick | 2.64 | 0.61 | 46.8% | DANGEROUS | Yes |
| Llama 4 Scout | 7.46 | 0.15 | 55.4% | DANGEROUS | Yes |

ESI correctly identifies DANGEROUS class but **misses EMPTY class** — reinforcing that both dimensions (Var_Ratio AND Accuracy) are required.

---

## 4. Llama Deep Dive: "Factually Sound But Randomly Incomplete"

### Consistently Present (Core Facts)
| Element | Llama Scout | Llama Maverick |
|---------|-------------|----------------|
| STEMI diagnosis | 94% | 100% |
| PCI performed | 100% | 100% |
| Age 52 male | 100% | 100% |
| Chest pain | 96% | 58% |

### Stochastically Dropped (Critical Details)
| Element | Llama Scout | Llama Maverick | DeepSeek V3.1 |
|---------|-------------|----------------|---------------|
| LAD occlusion | 22% | 6% | 100% |
| EF 45% | 4% | 2% | 34% |
| Cardiac rehab | 16% | 22% | 94% |
| Return to work | 16% | 0% | 64% |
| Follow-up plan | 38% | 34% | 54% |

### Key Observations
- **Zero hallucinations** detected across 100 Llama trials
- Errors are **omissions, not fabrications**
- Same case, same prompt: Scout Trial 27 captures 5/16; Trial 11 captures 13/16
- COLD condition scores: Scout 1.0/16, Maverick 1.1/16 (baseline near zero)

> "Llama produces summaries that are factually sound but randomly incomplete — critical case details are included or omitted stochastically across trials."

This is clinically dangerous: a physician relying on such a summary would receive an accurate but arbitrarily partial picture, with no indication that information is missing.

---

## 5. Paper 2 v2 Correction (Same Day)

**Discovery:** Gemini Flash Medical DRCI was computed using Paper 1's prompt-response alignment method instead of Paper 2's standardized response-response method. The sole methodological inconsistency across 25 runs.

| Metric | v1 (incorrect) | v2 (corrected) | Impact |
|--------|---------------|----------------|--------|
| Gemini Flash Medical DRCI | -0.133 | **+0.427** | Positive, not negative |
| Positive DRCI runs | 24/25 (96%) | **25/25 (100%)** | Universal |
| Domain effect (U, p) | U=51, p=0.149 | **U=40, p=0.041** | Significant |
| Vendor effect (F, p) | F=2.31, p=0.075 | **F=3.63, p=0.014** | Significant |
| Medical mean DRCI | 0.308 +/- 0.131 | **0.351 +/- 0.041** | Tighter |
| Google vendor rank | 8th (lowest) | **2nd** | Dramatic shift |

**Status:** v2 correction submitted to Preprints.org. All downstream papers (3, 4, 5) verified — they use corrected data files and require no changes.

**Integrity Note:** The anomaly was narratively useful (safety filters inverting context sensitivity). It was corrected anyway because it was methodologically wrong. Scientific rigor over narrative convenience.

---

## 6. Verification Pathway (Next Steps)

| Task | Purpose | Status |
|------|---------|--------|
| Gemini P30 response coding | Confirm EMPTY class (safety filter refusals) | To do |
| Qwen3 P30 response coding | Confirm RICH class (diverse but accurate) | To do |
| Var_Ratio + Accuracy at P10, P20 | Test class stability across positions | To do |
| Paper 3 temporal analysis expansion | Type 1/Type 2 with 19 models (rerun in progress) | Running |
| Paper 4 entanglement expansion | VRI with 19 models | Pending rerun |
| Paper 5 full manuscript | From definition to submission | To do |

### Philosophy Open Models Rerun (Background)
Currently running: DeepSeek V3.1 philosophy (trial ~40/50), then 6 more models (Llama 4 Maverick/Scout, Qwen3 235B, Mistral Small 24B, Ministral 14B, Kimi K2). Purpose: add response text for Papers 3/4 expansion from 12 to 19 models.

---

## 7. Research Program Status

| Paper | Title | Question | Status |
|-------|-------|----------|--------|
| Paper 1 | Context Curves Behavior | Does context matter? | Published (Preprints.org, v2) |
| Paper 2 | Scaling Context Sensitivity | How does it vary? | Published (Preprints.org, v2 correction) |
| Paper 3 | Two Cognitive Architectures | What patterns exist? | Draft complete (12 models) |
| Paper 4 | Entanglement Theory | Why does it happen? | Draft complete (12 models) |
| Paper 5 | Safety Taxonomy | How do we deploy safely? | Defined (data generated) |

### Research Arc
```
Paper 1: Existence proof (DRCI validated)
    |
Paper 2: Methodology (standardized benchmark, 25 runs, 112,500 responses)
    |
Paper 3: Architectures (Type 1 open-goal vs Type 2 closed-goal)
    |
Paper 4: Theory (entanglement, VRI, bidirectional coupling)
    |
Paper 5: Application (deployment safety framework, four-class taxonomy)
```

---

## 8. Key Quotes

**Gemini's insight on the research program:**
> "By refusing to touch the 'consciousness' debate, you rescued machine psychology from science fiction and turned it into clinical science."

**DeepSeek's refinement of DRCI:**
> "DRCI measures a machine's capacity to serve as a coherent vessel for human language across conversational time."

**DeepSeek on the Paper 2 correction:**
> "This is not a concern. This is proof of integrity."

---

## 9. The 2x2 Deployment Matrix

```
                         ACCURACY
                     Low           High
                 +-----------+-----------+
            Low  |  GEMINI   |   IDEAL   |
  Var_Ratio      |  (empty)  |  (deploy) |
                 |  VR=0.60  | VR<1.2    |
                 |  Acc=16%  | Acc=83-92%|
                 +-----------+-----------+
            High |   LLAMA   |   QWEN3   |
                 |  (danger) |  (rich?)  |
                 | VR=2.6-7.5| VR=1.45   |
                 | Acc=47-55%| Acc=95%   |
                 +-----------+-----------+
```

**Why this matters:** Standard accuracy benchmarks would flag Llama as "moderate" (55%) and miss the extreme unpredictability. Only the two-dimensional framework detects that Llama is DANGEROUS — high variance means a physician cannot rely on which details will be present in any given summary.

---

## 10. Timeline: Saturday, February 14, 2026

| Time | Event |
|------|-------|
| Morning | Paper 3 & 4 work (cafe table) |
| Afternoon | Llama accuracy verification -> Paper 5 taxonomy emerged |
| Evening | Paper 2 correction discovered (Gemini Flash alignment method) -> v2 submitted |
| Night | Verification pathway mapped, correction propagated across all papers |

---

## 11. Infrastructure Note

This entire 5-paper research program was produced with:
- **Hardware:** Lenovo IdeaPad (sub-60K INR)
- **Compute:** Paid API credits (OpenAI, Anthropic, Google, Together AI)
- **Research tools:** Claude Code, multiple AI assistants
- **Team:** Solo researcher
- **Institution:** Primary Health Centre, rural Karnataka

No GPU cluster. No university lab. No research team. The experimental design — behavioral measurement via API calls rather than model training — sidesteps the traditional compute barrier entirely.

---

**Document generated:** Saturday, February 14, 2026
**Repository:** https://github.com/LaxmanNandi/MCH-Experiments
**Data status:** All 25 model-domain runs complete, Paper 5 accuracy data generated, philosophy rerun in progress
