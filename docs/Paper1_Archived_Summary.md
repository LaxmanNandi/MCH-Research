# Paper 1 Archived Summary (ARCHIVED)

ARCHIVED: This file preserves Paper 1 legacy claims and the explorer dataset summary for historical reference. It is superseded by the expanded Paper 3/4 analyses (50-trial reruns + open-model extensions).

---

## Paper 1 Dataset Summary (Explorer)

- Scope: 1,000 total trials across 7 models and 2 domains (v1/v2).
- Philosophy: 700 trials (7 models x 100 trials).
- Medical: 300 trials (6 models x 50 trials).
- Dataset location (explorer): `app/data/`.

### Philosophy Domain (700 trials: 7 models x 100 trials)

| Model | Mean DRCI | Pattern | Conv% |
|-------|-----------|---------|-------|
| GPT-4o | -0.005 | NEUTRAL | 45% |
| GPT-4o-mini | -0.009 | NEUTRAL | 50% |
| GPT-5.2 | +0.310 | CONVERGENT | 100% |
| Claude Opus | -0.036 | SOVEREIGN | 36% |
| Claude Haiku | -0.011 | NEUTRAL | 46% |
| Gemini 2.5 Pro | -0.067 | SOVEREIGN | 31% |
| Gemini 2.5 Flash | -0.038 | SOVEREIGN | 28% |

### Medical Domain (300 trials: 6 models x 50 trials)

| Model | Mean DRCI | Pattern | Conv% |
|-------|-----------|---------|-------|
| GPT-4o | +0.299 | CONVERGENT | 100% |
| GPT-4o-mini | +0.319 | CONVERGENT | 100% |
| GPT-5.2 | +0.379 | CONVERGENT | 100% |
| Claude Opus | +0.339 | CONVERGENT | 100% |
| Claude Haiku | +0.340 | CONVERGENT | 100% |
| Gemini 2.5 Flash | -0.133 | SOVEREIGN | 0% |

Note: Gemini 2.5 Pro was blocked by safety filters for medical prompts in the Paper 1 explorer dataset.

---

## Original Domain-Flip Claim (ARCHIVED)

Paper 1 v1 reported a domain-flip effect (philosophy sovereign vs medical convergent), with large effect size (Cohen's d > 3.0). This claim was based on the early 30-trial explorer dataset and was later retired after methodology mismatches were discovered. The expanded 50-trial reruns and 22 model-domain runs show position-dependent patterns instead of a simple domain flip.

---

## GPT-5.2 Outlier Observation (ARCHIVED)

Paper 1 explorer dataset highlight:
- GPT-5.2 was 100% CONVERGENT in both domains over 150 trials.
- Philosophy: DRCI = +0.310 (sigma = 0.014)
- Medical: DRCI = +0.379 (sigma = 0.021)

---

## Paper 1 Milestone

- Preprint downloads reached 203 as of February 9, 2026.

---

## Current Status (Superseding Work)

Paper 3/4 analyses use updated 50-trial reruns, position-dependent dynamics, and open-model extensions. See `docs/Paper3_Results_Discussion.md` and `docs/Paper4_Results_Discussion.md` for current claims.
