# STATUS: ARCHIVED — Superseded MI Hypothesis

**Folder:** `data/paper6/conservation_law_verification/`
**Date archived:** April 2026
**Status:** ARCHIVED. Superseded. **Not used in published manuscript.**

---

## What this folder contains

This folder holds the results of an **early hypothesis test** (Feb 15, 2026) that proposed an information-theoretic derivation of the conservation product:

- **Old hypothesis:** ΔRCI = 1 - exp(-2·MI), and ΔRCI × Var_Ratio = 1 - 2·MI
- **MI estimator used:** KSG-style mutual information estimator on response embedding distributions
- **Outcome:** **The MI hypothesis was rejected.** Test 1 (MI → ΔRCI) was not significant (Pearson r = 0.408, p = 0.147). Test 2 (conservation prediction) was significant but with the wrong slope (-0.034 instead of 1.0). Predicted dRCI values from the MI formula were physically impossible (e.g. -56.4267 for one philosophy run), confirming the MI estimator was unreliable in this regime.

## Why "MI" values are negative in `conservation_law_results.json`

Mutual information cannot be negative in theory. The "mi" field in the JSON contains a **KSG estimator output**, which is known to produce negative values when true MI is near zero due to estimator bias. The negative values are an estimator artefact, not a theoretical claim. They are the reason the MI hypothesis was rejected — a true MI quantity should never go negative across so many runs.

## What replaced this in the current manuscript

Paper 6 (`papers/paper6_conservation/v1_submission/paper6_final.tex`) presents the result as an **empirical conservation constraint**:

> K = ΔRCI × Var_Ratio ≈ constant within domain

There is **no claim** that K is derived from mutual information. The current manuscript does not contain the term "mutual information." K is an observed invariant, not an information-theoretic prediction.

## Authoritative current data

The official, post-audit data used in the published manuscript is in:

- `data/paper6/paper6_manuscript_data.json` — corrected, unified Table 4 source
- `data/paper6/conservation_product_test.csv` — 24-run product table

This folder (`conservation_law_verification/`) represents an earlier exploratory analysis whose negative result informed the current empirical-only framing.

## For auditors and reviewers

If you are auditing Paper 6's claim, the authoritative files are the two listed above. The files in this folder document a hypothesis that was tested and rejected, and they are retained for transparency about the research history — not because they support the current manuscript.

The negative "mi" values here are evidence of the MI estimator's failure, which is why MI does not appear as a causal quantity in the published paper.
