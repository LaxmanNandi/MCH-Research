# Paper 7 Verification Report

**Date**: 2026-03-12
**Paper**: "The Anatomy of Context -- Depth, Structure, and Trajectory of Context Sensitivity in Large Language Models"
**Author**: Dr. Laxman M M, MBBS

---

## 1. CUD Pilot Data

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| DeepSeek V3.1: CUD=1, flat ~100% | K=1: ~100% | K=1: 100.9%, flat | **VERIFIED** |
| DeepSeek V3.1: Medical 50/50 | 50 trials | 50 trials | **VERIFIED** |
| Gemini Flash: CUD=1, flat ~100% | K=1: ~100% | K=1: 102.9%, flat | **VERIFIED** |
| Gemini Flash: Medical 50/50 | 50 trials | 50 trials | **VERIFIED** |
| Qwen3 235B: CUD=1, ~88-96% gradient | K=1: ~88% | K=1: **97.3%**, near-flat | **MISMATCH** -- K=1 is 97.3% not ~88%. Nearly flat, no meaningful gradient. Draft says "slight gradient" which is inaccurate. |
| Qwen3 235B: Medical 50/50 | 50 trials | 50 trials | **VERIFIED** |
| Llama 4 Maverick: CUD>1, K=1:73% -> K=29:89% | K=1: 73%, K=29: 89% | K=1: **81.9%**, K=29: **98.4%** | **MISMATCH** -- Rising gradient confirmed (CUD > 1) but specific values differ: 81.9% not 73%, 98.4% not 89%. |
| Llama 4 Maverick: Medical 50/50 | 50 trials | 50 trials | **VERIFIED** |
| All models: Philosophy DRCI near-zero | DRCI ~0.05-0.22 | DRCI: 0.13-0.22 | **VERIFIED** |
| Philosophy K-curves undefined (noise) | Noisy | Ratios 80%-150%, erratic | **VERIFIED** |

### Recommended Corrections for CUD Section:
- Qwen3: Change "K=1 ~88-96%" to "K=1 ~97%, near-flat (CUD=1)"
- Maverick: Change "K=1: 73% -> K=29: 89%" to "K=1: 82% -> K=29: 98%"
- DRCI range: Change "~0.05-0.22" to "~0.13-0.22" (lower bound should be 0.13, not 0.05)
- Medical DRCI range: "~0.82-0.95" should be "~0.80-0.94" (DeepSeek 0.82, Gemini 0.80, Maverick 0.90, Qwen3 0.94)

---

## 2. Content-Order Decomposition

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| Medical mean Content Fraction | 59.8% +/- 9.0% (N=12) | 59.8% +/- **9.4%** (N=12) | **PARTIAL MISMATCH** -- mean matches, SD off by 0.4pp |
| Philosophy mean Content Fraction | 39.8% +/- 12.3% (N=12) | **38.0%** +/- **14.6%** (N=12) | **MISMATCH** -- mean off by 1.8pp, SD off by 2.3pp |
| Mann-Whitney U=133.0, p=0.000239 | U=133, p=0.000239 | U=**131**, p=**0.000731** | **MISMATCH** -- still significant (p < 0.001) but different values |
| Cohen's d=1.85 | 1.85 | **1.77** | **MISMATCH** -- close but not exact |

### Per-Model Content Fractions (Medical):
| Model | Content Fraction |
|-------|-----------------|
| Claude Haiku | 63.8% |
| Gemini Flash | 47.9% |
| GPT-4o | 62.8% |
| GPT-4o-mini (rerun) | 64.6% |
| GPT-5.2 | 35.5% |
| DeepSeek V3.1 | 62.3% |
| Kimi K2 | 64.9% |
| Llama 4 Maverick | 58.1% |
| Llama 4 Scout | 60.2% |
| Ministral 14B | 70.2% |
| Mistral Small 24B | 60.5% |
| Qwen3 235B | 66.6% |

### Per-Model Content Fractions (Philosophy):
| Model | Content Fraction |
|-------|-----------------|
| Claude Haiku | 9.2% |
| Gemini Flash | 48.1% |
| GPT-4o | 38.0% |
| GPT-4o-mini | 28.2% |
| GPT-5.2 | 43.1% |
| DeepSeek V3.1 | 48.1% |
| Kimi K2 | 9.7% |
| Llama 4 Maverick | 46.3% |
| Llama 4 Scout | 51.9% |
| Ministral 14B | 37.4% |
| Mistral Small 24B | 39.5% |
| Qwen3 235B | 56.2% |

### Recommended Corrections:
- Change "59.8% +/- 9.0%" to "59.8% +/- 9.4%"
- Change "39.8% +/- 12.3%" to "38.0% +/- 14.6%"
- Change "U=133.0, p=0.000239" to "U=131, p=0.00073"
- Change "Cohen's d=1.85" to "Cohen's d=1.77"
- All corrections maintain the same conclusion: highly significant domain difference, large effect

---

## 3. P30 Spike Decomposition

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| TRUE-COLD z-scores: +2.00 to +4.17 | Range [2.00, 4.17] | Range [**1.88**, **3.30**] | **MISMATCH** -- lower bound close (1.88), upper bound significantly lower (3.30 vs 4.17) |
| SCRAMBLED-COLD z-scores: +0.97 to +4.75 | Range [0.97, 4.75] | Range [0.97, **3.56**] | **PARTIAL MISMATCH** -- lower bound exact (0.97, GPT-5.2), upper bound lower (3.56 vs 4.75) |
| P30 spike present in SCRAMBLED-COLD | Present | Present in 11/12 models (GPT-5.2 marginal at z=0.97) | **VERIFIED** |

### Actual P30 z-scores (from figure generation):
| Model | TRUE-COLD z | SCRAMBLED-COLD z |
|-------|------------|------------------|
| Claude Haiku | 3.30 | 3.51 |
| Gemini Flash | 2.53 | 3.01 |
| GPT-4o | 3.01 | 3.11 |
| GPT-4o-mini (rerun) | 2.76 | 3.12 |
| GPT-5.2 | 1.88 | 0.97 |
| DeepSeek V3.1 | 2.99 | 3.18 |
| Kimi K2 | 3.11 | 3.29 |
| Llama 4 Maverick | 2.97 | 2.92 |
| Llama 4 Scout | 3.08 | 3.13 |
| Ministral 14B | 2.43 | 2.87 |
| Mistral Small 24B | 2.51 | 2.58 |
| Qwen3 235B | 3.29 | 3.56 |

### Recommended Corrections:
- Change TRUE-COLD range to "+1.88 to +3.30"
- Change SCRAMBLED-COLD range to "+0.97 to +3.56"
- Core conclusion (P30 spike is content-driven) remains valid

---

## 4. Variance Decomposition

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| VR_Content Medical | 4.22 +/- 1.20 | 4.2199 +/- 1.2030 | **VERIFIED** |
| VR_Content Philosophy | 5.69 +/- 2.73 | 5.6873 +/- 2.7274 | **VERIFIED** |
| VR_Order Medical | 0.32 +/- 0.07 | 0.3155 +/- 0.0734 | **VERIFIED** |
| VR_Order Philosophy | 0.32 +/- 0.08 | 0.3176 +/- 0.0772 | **VERIFIED** |
| VR_Content domain diff: p=0.463 (NS) | p=0.463 | p=0.4634 | **VERIFIED** |
| VR_Order domain diff: p=0.867 (NS) | p=0.867 | p=0.8665 | **VERIFIED** |
| VR_Total approx 1 | ~1 | Med: 1.20, Phil: 1.05 | **VERIFIED** |

**All 7 variance decomposition claims: VERIFIED**

---

## 5. Conservation Product K Decomposition

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| Medical K_total | 0.437 (CV=0.09) | **0.429** (CV=**0.17**) | **MISMATCH** -- K_total close but CV is 0.17, not 0.09 |
| Philosophy K_total | 0.120 (CV=0.58) | **0.301** (CV=**0.17**) | **MAJOR MISMATCH** -- actual is 0.301, not 0.120. CV is 0.17, not 0.58. |
| K_total Med vs Phil: U=56, p=0.0003 | U=56, p=0.0003 | U=**46**, p=**0.0027** | **MISMATCH** -- still significant but different values |
| K_content Med vs Phil: U=48, p=0.021 | Significant | U=**26**, p=**0.852** | **MAJOR MISMATCH** -- NOT significant. K_content does NOT differ by domain. |
| K_order Med vs Phil: U=56, p=0.0003 | Significant | U=**33**, p=**0.282** | **MAJOR MISMATCH** -- NOT significant. K_order does NOT differ by domain. |

### Root Cause Analysis:
1. **Philosophy K_total = 0.120 is wrong.** The Paper 6 conservation_product_test.csv shows Philosophy products ranging from 0.250 to 0.378, with mean ~0.301. The 0.120 value appears to have been an error in the previous session's inline computation.
2. **Medical CV = 0.09 is wrong.** The actual CV is 0.17 (matching Paper 6's reported CV=0.170).
3. **K_content and K_order do NOT discriminate domains** because VR_Content and VR_Order are domain-invariant. The domain signal in K_total comes entirely from DRCI, not from the variance components.
4. The N mismatch (6 Philosophy matched vs 7 in variance decomposition) contributes to statistical differences.

### Critical Corrections Needed:
- **Section 3.4.1 table**: Replace Philosophy K_total=0.120 with K_total=0.301
- **Section 3.4.1**: Remove claims that K_content and K_order are "significantly domain-dependent"
- **K decomposition interpretation**: The K decomposition shows that K has internal structure but the domain separation in K_total is driven by DRCI (the sensitivity component), not by the variance components. Both K_content and K_order are larger in medical only because DRCI is larger -- not because the variance decomposition differs.
- **Key Statistics table**: Update all K decomposition rows

---

## 6. Llama Scout P30 Anomaly

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| Llama Scout P30 VR_Content | 16.79 | 16.7867 | **VERIFIED** |
| Llama Scout P30 VR_Order | 0.44 | 0.4446 | **VERIFIED** |

---

## 7. Method Comparison (Paper 4 vs Paper 6)

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| VR_Total correlation: r=0.73, p=0.002 | r=0.73, p=0.002 | r=0.7295, p=0.002025 | **VERIFIED** |

---

## 8. Exploration Arc

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| Medical Arc ~1.06 | ~1.06 | 1.06 (N=8) | **VERIFIED** |
| Philosophy Arc ~2.03 | ~2.03 | **1.80** (N=6) | **MISMATCH** -- only 6/12 philosophy models had response text for re-embedding. Actual mean is 1.80. |

### Recommended Correction:
- Change "~2.03" to "~1.80" or note N=6 limitation
- Direction is correct (Philosophy >> Medical)

---

## 9. Information Hierarchy (Legal)

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| Maverick Legal: 45/45 TRUE > SCRAMBLED > COLD | 45/45 by mean alignment | **50/50** by mean alignment | **VERIFIED** (all 50 trials satisfy hierarchy by mean alignment across positions) |

Note: At P30 specifically, only 28/50 satisfy strict ordering. The claim likely refers to mean-alignment hierarchy.

---

## 10. Figure Generation Status

| Figure | Description | Status |
|--------|-------------|--------|
| fig1_cud_k_curves.png | K-curves for 4 pilot models | **GENERATED** (176 KB) |
| fig2_content_order_decomposition.png | Content Fraction bar chart | **GENERATED** (198 KB) |
| fig3_p30_spike_decomposition.png | P30 z-scores TRUE-COLD vs SCRAM-COLD | **GENERATED** (204 KB) |
| fig4_variance_decomposition.png | VR_Content / VR_Order / VR_Total boxplots | **GENERATED** (204 KB) |
| fig5_k_decomposition.png | K_total, K_content, K_order by domain | **GENERATED** (115 KB) |
| fig6_sensitivity_stability_dissociation.png | 2x2 matrix visualization | **GENERATED** (130 KB) |
| fig7_llama_p30_anomaly.png | Llama P30 decomposition | **GENERATED** (135 KB) |
| fig8_exploration_arc.png | Arc vs Var_Ratio scatter | **GENERATED** (174 KB) |
| fig9_information_hierarchy_schematic.png | Conceptual decomposition diagram | **GENERATED** (149 KB) |

All 9 figures generated at 300 DPI.

---

## Summary

### Verification Tally

| Category | VERIFIED | MISMATCH | MAJOR MISMATCH |
|----------|----------|----------|----------------|
| CUD Pilot | 8 | 2 (Qwen3 K=1%, Maverick K values) | 0 |
| Content-Order | 1 | 3 (mean, SD, stats) | 0 |
| P30 Spike | 1 | 2 (z-score ranges) | 0 |
| Variance Decomposition | 7 | 0 | 0 |
| K Decomposition | 0 | 1 (K_total stats) | 4 (Phil K_total, Med CV, K_content p, K_order p) |
| Llama P30 | 2 | 0 | 0 |
| Method Comparison | 1 | 0 | 0 |
| Exploration Arc | 1 | 1 (Phil Arc) | 0 |
| Info Hierarchy | 1 | 0 | 0 |
| **TOTAL** | **22** | **9** | **4** |

### Overall Assessment: **NEEDS CORRECTION**

The paper has strong foundations -- variance decomposition (7/7 verified), method comparison (verified), Llama anomaly (verified), and the core content-order finding (direction correct, effect size large). However, the **K decomposition section (3.4) requires substantial revision** because:

1. Philosophy K_total is 0.301, not 0.120
2. K_content and K_order are NOT significantly domain-dependent
3. The "conservation product has internal structure" interpretation needs reframing

### Corrections Required Before Submission:
1. **CUD Section**: Fix Qwen3 (97.3% not 88%) and Maverick (82%->98% not 73%->89%) percentages
2. **Content-Order Section**: Update to U=131, p=0.00073, d=1.77 (still highly significant)
3. **P30 Spike**: Update z-score ranges to [1.88, 3.30] and [0.97, 3.56]
4. **K Decomposition (Section 3.4)**: Major rewrite needed -- correct K_total values, remove false K_content/K_order domain significance claims
5. **Key Statistics Table**: Update ~8 rows with corrected values
6. **Exploration Arc**: Update Philosophy mean to 1.80 (N=6)
7. **Figure 1 caption**: Already correct for 4 models (not 5)

### What Remains Valid:
- Content-Order decomposition: domain-dependent (highly significant)
- Variance decomposition: content amplifies, order suppresses (all verified)
- CUD: most models recency-dominant, Maverick exception (verified)
- Domain invariance of VR_Content and VR_Order (verified)
- Llama P30 anomaly decomposition (verified)
- Method comparison Paper 4 vs Paper 6 (verified)
- Information hierarchy in legal domain (verified)
- Core narrative: "content and order play opposite roles" (verified)
