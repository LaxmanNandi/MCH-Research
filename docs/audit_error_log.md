# Audit Error Log (DIA Natural Experiment)

**Scope:** Paper 3/4 audit work (Feb 8, 2026).

---

## Errors Caught

| ID | Description | Caught By | Files Affected | Fix Commit | Fix Time (local) | Class | Notes |
|----|-------------|-----------|----------------|-----------|------------------|-------|-------|
| E1 | ?RCI definition wrong in Paper 4 docs (stated as 1 - cosine_similarity(TRUE, COLD) instead of within-condition self-similarity difference). | Codex (audit) | docs/Paper4_Results_Discussion.md; docs/Paper4_Entanglement_Methods_Appendix.md | 413a11c | 2026-02-08T18:48:56+05:30 | Definition error | Docs only; code already correct. |
| E2 | N mismatch in entanglement correlation (reported N=480 instead of N=240). | Codex (audit) | docs/Paper4_Results_Discussion.md; docs/Paper4_Claims_Evidence.md; docs/Paper4_Entanglement_Methods_Appendix.md | 413a11c | 2026-02-08T18:48:56+05:30 | Consistency error | Pooled points are 8 model-domain runs x 30 positions. |
| E3 | Wording inconsistency: ?8 models x 2 domains? vs ?8 model-domain runs?. | Codex (audit) | analysis/independence_var_ratio_summary.md | 413a11c | 2026-02-08T18:48:56+05:30 | Consistency error | Standardized to model-domain runs phrasing. |
| E4 | r = -1.000 treated as empirical in ?RCI vs RCI_COLD test; actually exact due to dataset construction (RCI_TRUE ? 1). | Claude Code (audit) | analysis/independence_test_summary.md | 413a11c | 2026-02-08T18:48:56+05:30 | Definition error | Reframed as invalid test for independence. |
| E5 | Language correction: ?validated? -> ?supported? for independence claim. | Claude Code (audit) | analysis/independence_var_ratio_summary.md | 413a11c | 2026-02-08T18:48:56+05:30 | Consistency error | Conservative phrasing for statistical support. |
| E6 | Documentation gap: Llama anomaly lacked dedicated note (ESI calculation + intact capability + broken coherence framing). | Codex (audit) | docs/Llama_Safety_Anomaly.md; docs/Paper4_Results_Discussion.md | b92ffbc | 2026-02-08T19:23:39+05:30 | Documentation gap | Position-level evidence added for P30 anomaly. |

---

## Error Classification Summary

- Consistency error: 3
- Definition error: 2
- Documentation gap: 1

---

## Timestamps

- Introduction times are unknown for all items in this audit window.
- Fix times are taken from Git commit metadata.

- E1: fixed at 2026-02-08T18:48:56+05:30 (commit 413a11c)
- E2: fixed at 2026-02-08T18:48:56+05:30 (commit 413a11c)
- E3: fixed at 2026-02-08T18:48:56+05:30 (commit 413a11c)
- E4: fixed at 2026-02-08T18:48:56+05:30 (commit 413a11c)
- E5: fixed at 2026-02-08T18:48:56+05:30 (commit 413a11c)
- E6: fixed at 2026-02-08T19:23:39+05:30 (commit b92ffbc)

---

## False Positives

| ID | Description | Notes |
|----|-------------|-------|
| FP1 | Var_Ratio = 7.46 not visible in entanglement_variance_summary.csv (mean-only file). | Not an error; spike exists in position-level data. |

---

## Notes

- This log reflects errors discovered during the Feb 8, 2026 audit cycle.
- Additional provenance can be reconstructed by running `git show <commit>` on the listed fix commits.