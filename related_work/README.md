# Related Work — Contemporary Research Landscape

This folder documents the contemporary research landscape relevant to the
MCH Research Program (Papers 1–9). It is a living scholarly record of
parallel and adjacent work in the field, maintained alongside the
MCH research as the field evolves.

## What this is — and is not

This folder is a **snapshot of the contemporary research landscape**
maintained by the programme lead alongside the MCH research. It is
intended as scholarly scaffolding — a living, dated record of how the
MCH program reads and locates itself within the broader field.

It is **not** a peer-reviewed literature review, a published survey,
or an authoritative bibliographic source. Entries are summaries
written from a single reader's perspective for working purposes.
Readers should consult the cited primary sources directly for any
formal use.

## Purpose

1. **Document the convergence.** The questions explored in the MCH program —
   context sensitivity, encoding fidelity, multi-turn variance, multilingual
   clinical AI, conservation behaviour in LLMs — are being investigated
   simultaneously by groups across the world. This folder records that
   convergence honestly.

2. **Preserve provenance of awareness.** Each entry is dated. This timestamps
   when the MCH program became aware of each parallel work and what
   relationships exist between findings.

3. **Support future synthesis work.** When integrative papers connecting
   multiple subfields (encoding → interpretability → multi-turn → clinical
   safety) become possible, the source material is already organised here.

4. **Reduce session-restart cost for AI collaboration.** Future Claude
   sessions reading this folder can quickly understand the contemporary
   landscape rather than re-searching every time.

## Scope

In scope:
- Empirical and theoretical work on multilingual LLM behaviour
- Multi-turn LLM evaluation methodology
- Interpretability and fidelity-related metrics
- Clinical AI safety in deployment contexts
- Indian-language AI infrastructure
- Misalignment, monitorability, and structural failure modes

Out of scope:
- Editorial correspondence with journals (kept private)
- Speculation about specific researchers or institutions
- Anything that could read as competitive or adversarial

## Organisation

```
related_work/
├── README.md                              (this file)
├── 01_multilingual_clinical_ai.md         multilingual benchmarks + clinical safety
├── 02_multi_turn_evaluation.md            context sensitivity + multi-turn metrics
├── 03_interpretability_fidelity.md        interpretability + fidelity measurement
├── 04_misalignment_safety.md              misalignment + AI safety + monitorability
├── 05_indian_language_infrastructure.md   indigenous Indic LLMs and datasets
└── timeline.md                            chronological view of field convergence
```

## Entry format

Each entry uses this template:

```markdown
## Paper Title (Authors, Year)

**Venue:** [arXiv ID / Journal / Conference]
**Date:** YYYY-MM-DD (or approximate)
**Link:** [DOI or URL]

**Summary:**
2-3 sentence description of what the paper measured, argued, or proposed.

**Intersection with MCH research:**
Specific intersection with the MCH program. This field describes
overlap or convergence, not precedence or dependence. Where two
works approach the same problem from different angles, that is
stated as parallel work — not as one extending the other unless
the chronological and citation record supports that claim.

**Citation status:**
- Cites MCH: [Yes / No / Indirectly / N/A]
- Cited by MCH: [Yes — in Paper X / Not yet]

**Notes:** (optional)
```

## A note on framing

These entries describe the contemporary research landscape as it
intersects with the MCH program. The MCH program is one contributor
to a multi-group convergence on encoding fidelity, multi-turn
evaluation, multilingual clinical AI, and fidelity-as-safety in LLMs.
Several of the works listed here predate or are contemporaneous with
specific MCH papers; the intersection fields aim to describe overlap
rather than to claim any positioning of MCH relative to the field.

## Maintenance

- Add new entries when new relevant papers are encountered.
- Annotate citation status if/when MCH papers begin to be cited externally.
- This folder is public and reflects scholarly engagement with the field.
- All entries should be factual and respectful. No speculation about
  motives, no commentary on individuals, no internal correspondence.

## MCH Research Program reference

For context on the nine MCH papers this folder relates to, see the
[main repository README](../README.md) and [papers/README.md](../papers/README.md).

## Revision history

Lightweight changelog for this folder. Entries are added when material
changes are made (new papers, citation-status updates, structural
edits, corrections).

- **2026-05-21** — Initial commit. ~35 papers across 5 thematic files
  plus timeline. Entries verified for accuracy against primary sources
  before commit (see verification report archived in session notes).

---

**Last updated:** May 21, 2026
**Maintainer:** Dr. Laxman M M, MBBS, DNB General Medicine Resident,
KC General Hospital, Bangalore.
