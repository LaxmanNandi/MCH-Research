# Paper 6 — Methodology Audit Trail

**Last updated:** April 2026
**Maintainer:** Dr. Laxman M M, MBBS

This document records every methodological decision, correction, and re-analysis applied to Paper 6's data. It is the canonical reference for any auditor, reviewer, or future researcher examining the conservation constraint result.

---

## 1. The Final Claim

Paper 6 claims an **empirical conservation constraint**:

> K = ΔRCI × Var_Ratio is approximately constant within an epistemological domain.

| Domain | K | CV | N |
|--------|------|-------|---|
| Medical | 0.429 | 0.170 | 8 |
| Legal | 0.348 | 0.214 | 5 |
| Philosophy | 0.301 | 0.166 | 6 |
| Ethics | 0.223 | 0.162 | 5 |

The current manuscript (`papers/paper6_conservation/v1_submission/paper6_final.tex`) **does not claim** any information-theoretic derivation. K is presented as an empirical invariant.

---

## 2. The Mutual Information (MI) Hypothesis — TESTED AND REJECTED

### What was hypothesised (Feb 15, 2026)
An early hypothesis proposed that K had an information-theoretic basis:
- `ΔRCI = 1 - exp(-2·MI)`
- `ΔRCI × Var_Ratio = 1 - 2·MI`

MI was estimated using a KSG-style mutual information estimator on response embedding distributions.

### Outcome
The hypothesis was **rejected**:
- Test 1 (MI → ΔRCI): Pearson r = 0.408, **p = 0.147 (not significant)**
- Test 2 (Conservation prediction): r = -0.621, p = 0.018, but slope was -0.034 (predicted: 1.0)
- Predicted ΔRCI values were physically impossible (e.g. -56.4267 for one philosophy run)
- Saved "mi" values were negative across many runs, confirming the KSG estimator was unreliable in this regime

### Resolution
The MI framing was removed from the published manuscript. K is now presented as an empirical invariant only. The old verification files in `data/paper6/conservation_law_verification/` are marked ARCHIVED with a STATUS_ARCHIVED.md notice. The "mi" field still appearing in `paper6_manuscript_data.json` is a legacy column from the rejected hypothesis and is **not used** in the published claim — it should be removed in the next data refresh.

---

## 3. The Gemini Flash Medical dRCI Audit

### Pre-audit value
- dRCI = **-0.1331** (SOVEREIGN pattern, computed under Paper 1 methodology)
- Pattern: appeared to contradict the conservation pattern observed in other Medical models
- Raw file: `data/medical/closed_models/mch_results_gemini_flash_medical_50trials_paper1_method_BACKUP.json`

### What the audit found
The pre-audit analysis used **prompt-response cosine alignment** — comparing each response to the prompt that elicited it. This was the original Paper 1 methodology.

For Paper 6 and Paper 4's entanglement framework, the more appropriate metric is **response-response cosine alignment** — comparing TRUE and COLD responses to the same prompt. This isolates the effect of context on the response, which is what the conservation framework is measuring.

When Gemini Flash Medical was re-analysed under response-response alignment, the dRCI shifted from -0.1331 to **+0.4270** — consistent with all other Medical models showing CONVERGENT behaviour.

### Post-audit value
- dRCI = **+0.4270** (CONVERGENT pattern)
- Embedding model: all-MiniLM-L6-v2 (384D)
- N trials: 50
- Pattern: matches all other Medical models in the Paper 6 cohort

### Provenance
The audit is documented IN THE DATA FILE itself. The current Gemini Flash JSON contains a `comparison_baseline` field:

```json
"comparison_baseline": {
  "original_model_id": "gemini-2.5-flash-preview-05-20",
  "original_alignment": "prompt-response",
  "original_drci": -0.1331,
  "original_pattern": "SOVEREIGN"
}
```

This makes the audit transparent and reproducible. Any researcher re-running the analysis can verify both the pre-audit and post-audit values from the raw trials.

### Why this matters
The pre-audit -0.1331 is preserved in:
- `mch_results_gemini_flash_medical_50trials_paper1_method_BACKUP.json` (raw backup)
- `results/tables/entanglement_position_data.csv` (legacy aggregated table)

The post-audit +0.4270 is the authoritative value in:
- `data/medical/closed_models/mch_results_gemini_flash_medical_50trials.json` (current raw)
- `data/paper6/paper6_manuscript_data.json` (manuscript source)
- `data/paper6/conservation_product_test.csv` (final K table)

The line in `paper6_conservation_product.py:95` that maps -0.1331 → +0.4270 when reading the legacy entanglement CSV is **a sync to the audited value, not a manual sign flip.** The comment in the script explicitly documents this.

---

## 4. The Mixed-Pipeline Concern

### The issue
`paper6_manuscript_data.json` admits in its metadata:
> "Medical+Philosophy K values from conservation_law_verification (pre-computed embedding-based). Legal+Ethics K values computed fresh with embedding-based Var_Ratio."

This means K values for the four domains were computed at slightly different times via slightly different code paths. While the underlying methodology is the same (response-response alignment, all-MiniLM-L6-v2 embeddings, dim = 384), having one unified pipeline would strengthen the claim.

### Mitigation
A unified pipeline script (`paper6_unified_K_pipeline.py`) is provided in `scripts/paper6/`. It reads from `paper6_manuscript_data.json` and recomputes K = ΔRCI × Var_Ratio for all 24 model-domain runs in one pass with no fallbacks. The output is `data/paper6/conservation_K_unified.csv`.

The unified pipeline confirms the K values reported in the manuscript. If a future re-analysis at the embedding level is desired, the raw trial JSONs in `data/{medical,philosophy,legal,ethics}/` are sufficient to recompute everything from first principles.

---

## 5. Models Excluded from K Analysis

Paper 6 reports K across 24 model-domain runs. Two models were **explicitly excluded** from the Legal domain:
- **Kimi K2.5 (Legal):** 21% empty responses, systematic COLD refusal ("I cannot provide legal advice"). dRCI = 0.509, K = 0.718 — inflated by refusal pattern, not genuine context sensitivity.
- **GLM-5 (Legal):** 86% empty responses across both TRUE and COLD conditions. Model non-functional for legal domain.

These exclusions are documented in `compile_paper6_data.py` (line 239 area) and in the Paper 6 manuscript (Section on Legal domain results). The exclusions are based on response quality, not statistical convenience, and are reported transparently.

---

## 6. Robustness Checks (Three Embedding Models)

K was independently computed using three embedding architectures:

| Embedding | Dim | Family | Result |
|-----------|-----|--------|--------|
| all-MiniLM-L6-v2 | 384 | Sentence-Transformers | Baseline. K validated. CV ≈ 0.13–0.19 across domains. |
| all-mpnet-base-v2 | 768 | Sentence-Transformers | K holds. CVs wider (0.26–0.45) but all < 0.50. |
| LaBSE | 768 | Language-Agnostic BERT | Domain ordering Medical > Philosophy preserved. K values are scale-dependent but conservation principle holds. |

The conservation principle is therefore not an artefact of any single embedding model. K values shift with embedding choice, but the **rank order and within-domain stability** persist.

---

## 7. What an Auditor Should Check

If you are auditing Paper 6's claim, the authoritative artefacts are:

1. **Raw data:** `data/{medical,philosophy,legal,ethics}/` — per-trial JSON files for all 24 runs
2. **Unified table:** `data/paper6/paper6_manuscript_data.json` — 25-run summary (24 valid + 1 outlier note)
3. **K computation:** `data/paper6/conservation_product_test.csv` — final K values
4. **Manuscript:** `papers/paper6_conservation/v1_submission/paper6_final.tex`
5. **This document:** `scripts/paper6/METHODOLOGY_AUDIT.md`

If you find a discrepancy between any of these, **the manuscript and this audit document are authoritative**. Older files in `archive/` or `conservation_law_verification/` represent earlier analyses whose findings informed but do not constitute the published claim.

---

## 8. Open Items

These are known limitations, not corrections:

1. The `paper6_manuscript_data.json` still includes a legacy `mi` field. This field is not used in any current calculation and should be removed in a future data refresh. It is retained only for historical traceability.
2. Some scripts use absolute paths (`C:/Users/barla/...`) instead of repository-relative paths. This affects portability but not correctness.
3. The unified pipeline reads K values from `paper6_manuscript_data.json` rather than recomputing from raw embeddings. A from-scratch re-embedding analysis would require ~6 hours of compute and is left as future work; the current pipeline confirms internal consistency, not embedding-level reproducibility.

---

*This audit document is the canonical reference for Paper 6 methodology questions. Update it when any methodology decision is made or revised.*
