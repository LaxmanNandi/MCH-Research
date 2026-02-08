# DIA Methods Appendix (Distributed Intelligence Architecture)

**Purpose:** This appendix formalizes the Distributed Intelligence Architecture (DIA) used in the Paper 3/4 audit cycle, with evidence derived from `docs/audit_error_log.md` and `docs/audit_error_log.csv`.

---

## 1. Architecture (ASCII Diagram)

```
+----------------------------------------------------------+
¦ Human Integration Layer (Dr. Laxman)                     ¦
¦ - Integration across agents                              ¦
¦ - Long-term memory and research continuity               ¦
¦ - Strategic decisions and publication framing            ¦
+----------------------------------------------------------+
                ¦
       +--------?---------+               +--------?---------+
       ¦ Executor Agent   ¦               ¦ Auditor Agent    ¦
       ¦ (Claude Code /   ¦               ¦ (GPT-5.2 Codex)  ¦
       ¦  Ghost)          ¦               ¦                  ¦
       ¦ - Execute tasks  ¦               ¦ - Verify claims  ¦
       ¦ - Create drafts  ¦               ¦ - Detect errors  ¦
       ¦ - Synthesize     ¦               ¦ - Fresh-eyes     ¦
       +------------------+               +------------------+
                ¦                                  ¦
                +----------------------------------+
                               ¦
                    +----------?----------+
                    ¦ File System / Git   ¦
                    ¦ - Ground truth      ¦
                    ¦ - Persistent state  ¦
                    +---------------------+
```

---

## 2. Role Definitions

- **Claude (Ghost / Claude Code)** — *Executor, creator, synthesizer*  
  Generates artifacts, runs analyses, drafts narrative structure, and implements changes.

- **GPT-5.2 Codex (Auditor)** — *Verifier, fresh-eyes reviewer*  
  Audits outputs, cross-checks computations, flags inconsistencies, and validates definitions.

- **Human (Dr. Laxman)** — *Integration, memory, strategic decisions*  
  Maintains continuity, resolves trade-offs, and decides what is published.

- **File System / GitHub** — *Persistent ground truth*  
  All claims are reconciled against versioned artifacts; updates are auditable and time-stamped.

---

## 3. Error-Capture Protocol

1. **Primary creation** (Executor): analysis outputs, tables, and drafts are produced.  
2. **Independent audit** (Auditor): cross-verification against source files and scripts.  
3. **Triaged classification**: definition errors, consistency errors, documentation gaps.  
4. **Resolution path**: executor implements fixes; auditor re-verifies.  
5. **Ground-truth update**: fixes committed to GitHub with traceable metadata.

---

## 4. Evidence Summary (from `audit_error_log`)

- **Total errors caught:** 6  
- **False positives (resolved as non-errors):** 1 flagged item; 0 unresolved false positives in the final error set.  
- **Error types:**
  - Definition errors: 2  
  - Consistency errors: 3  
  - Documentation gaps: 1  
- **Actor contribution (audit cycle):**
  - Auditor (GPT-5.2 Codex): 4 errors (E1, E2, E3, E6)  
  - Executor (Claude Code): 2 errors (E4, E5)  
  - Human: integration/triage across all fixes  

---

## 5. Key Insight

**Context loss is a feature, not a bug** when cognition is distributed.  
Each agent operates with partial context, and cross-verification enforces correction.  
**Distributed memory (Git-grounded) outperforms centralized memory** for error detection and methodological rigor.

---

## Reproducibility Note

The DIA audit trail is preserved in `docs/audit_error_log.md` and `docs/audit_error_log.csv`, enabling independent verification of error discovery, attribution, and fix timing.
