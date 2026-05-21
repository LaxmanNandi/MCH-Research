# 01. Multilingual Clinical AI

Work on clinical deployment of LLMs across non-English languages,
particularly Indian languages, with attention to safety, fidelity,
and real-world translation quality.

---

## IndiHealthBench: Evaluating LLMs for Clinical Translation Across Indian Linguistic Diversity (2026)

**Venue:** Springer Lecture Notes in Computer Science — PReMI 2025 proceedings
**Date:** 2026
**Link:** https://link.springer.com/chapter/10.1007/978-3-032-18480-1_59

**Summary:**
Multilingual benchmark comprising 11,000 parallel sentence pairs spanning
13 Indian languages, designed to evaluate general-purpose and medically
fine-tuned LLMs on clinical terminology and semantic fidelity. Frames
healthcare communication in linguistically diverse environments as a
quality-and-equity-of-care issue.

**Intersection with MCH research:**
Most direct overlap with Paper 8 (Encoding Fidelity and Coherent
Misalignment). IndiHealthBench provides a structured benchmark to test
the failure modes Paper 8 documents empirically. Where Paper 8 measures
EFI as a single-axis fidelity metric, IndiHealthBench provides a
multi-axis benchmark for downstream clinical accuracy.

**Citation status:**
- Cites MCH: Not yet (timing makes this unlikely)
- Cited by MCH: Not yet

---

## Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Romanized Scripts in a Real World Setting

**Venue:** arXiv 2512.10780
**Date:** Late 2025 / 2026
**Link:** https://arxiv.org/pdf/2512.10780

**Summary:**
Empirical study of LLM clinical triage across Indian languages in their
native scripts versus Romanized transliterations, in a real-world
deployment context. Reports that script differences can cause
"nearly 2 million excess errors" in LLM-based health systems, and
proposes Uncertainty-based Selective Routing as mitigation.

**Intersection with MCH research:**
Provides large-scale empirical sizing of the clinical impact that
Paper 8's EFI measurement predicts. Where Paper 8 measures encoding
loss in controlled conditions, Script Gap measures error consequences
in real deployment. Complementary at different scales.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet — strong candidate for future citation in
  any follow-up Paper 10 or perspective piece

---

## MILA: Multilingual Indic Language Archive

**Venue:** OpenReview (under review)
**Date:** 2025–2026
**Link:** https://openreview.net/forum?id=WPw6ERKUZL

**Summary:**
The largest expert-curated Indic corpus to date — 7.5 trillion tokens
across 16 scheduled Indic languages and English. Constructed via a
multi-stage pipeline integrating large-scale web acquisition,
script-sensitive OCR for under-digitized Indic writing systems, and
LLM-assisted post-correction for translation fidelity.

**Intersection with MCH research:**
Infrastructure work that addresses the upstream data scarcity behind
Paper 8 / Paper 9 findings on EFI degradation. The fidelity problems
MCH measures may be partially addressable through better pretraining
corpora; MILA provides one such corpus.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet

---

## Indic-MMLU (benchmark within MILA submission) / IndicMMLU-Pro

**Venue:** Indic-MMLU appears as a benchmark embedded in the MILA
submission (above). For a standalone published version, see
**IndicMMLU-Pro (arXiv 2501.15747, Singh et al., Jan 2025)**.
**Date:** 2025
**Link:** https://arxiv.org/abs/2501.15747

**Summary:**
A standardised, semantically faithful benchmark for multilingual
LLM evaluation in Indic languages. Frames the question of whether
LLMs genuinely reason in Indic languages or rely on hidden
translation heuristics — native cognition vs. translation shortcut.

**Intersection with MCH research:**
The "reason in Indic or translate?" question is closely related to
what Paper 8 calls Coherent Misalignment: outputs that look fluent
but are semantically degraded because the model is operating on
degraded encoded input. Benchmark-side complement to MCH's
metric-side (EFI) measurement.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet — candidate for citation

---

## Bridging Health Literacy Gaps in Indian Languages: Multilingual LLMs for Clinical Text Simplification (Pavithra)

**Venue:** ACL SciProdLLM workshop, 2025 (Mumbai, Dec 2025)
**Date:** December 2025
**Link:** https://aclanthology.org/2025.sciprodllm-1.1.pdf

**Summary:**
Investigates LLM performance on simplification of medical reports
across multilingual settings, with attention to faithfulness of the
simplification to the source clinical content.

**Intersection with MCH research:**
Adjacent to Paper 8's clinical-context EFI measurements. Examines
faithfulness at the simplification rather than translation layer.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Not yet

---

## Simplification and Translation of Medical Reports Using Large Language Models — A Protocol for the Indian Context

**Venue:** Preprints.org
**Date:** August 2025
**Link:** https://www.preprints.org/manuscript/202508.0955

**Summary:**
Protocol paper describing a methodology for using LLMs to simplify and
translate medical reports specifically for the Indian linguistic
context.

**Intersection with MCH research:**
Same Preprints.org platform as several MCH papers. Adjacent clinical
focus. Worth comparing methodology when a future Paper 10 addresses
clinical deployment.

**Citation status:**
- Cites MCH: Not yet (timing)
- Cited by MCH: Not yet

---

## Towards Safe and Trustworthy Healthcare AI: Risk Assessment of Medical Dialogue Using LLMs

**Venue:** Human-Centric Intelligent Systems, Springer
**Date:** 2025
**Link:** https://link.springer.com/article/10.1007/s44230-025-00131-4

**Summary:**
Quantitative framework for evaluating the safety and trustworthiness
of LLMs in multilingual medical dialogues. Examines 13 LLMs spanning
general-purpose, open-source, and biomedical variants. Uses the German
subtask of the NTCIR-18 MedNLP-CHAT dataset.

**Intersection with MCH research:**
Parallel safety framework for medical dialogue with multilingual focus.
Paper 5's IDEAL/EMPTY/DIVERGENT/RICH taxonomy and this paper's risk
assessment framework address overlapping concerns from different
angles.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Not yet

---

## The State of Multilingual LLM Safety Research

**Venue:** arXiv 2505.24119
**Date:** May 2025
**Link:** https://arxiv.org/html/2505.24119v1

**Summary:**
Systematic review of nearly 300 publications (2020–2024) on multilingual
LLM safety. Documents the English-centric nature of the field and the
"language gap" in safety research. Calls for explicit focus on
mitigating the gap.

**Intersection with MCH research:**
Survey-level confirmation of the gap that Paper 8 measures empirically
for Indian languages. Useful framing context for Paper 8's argument
that EFI degradation has been under-quantified for the languages of
the global majority.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — strong candidate for citation in any
  future related paper

---

## The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts

**Venue:** arXiv 2401.13136
**Date:** January 2024
**Link:** https://arxiv.org/pdf/2401.13136

**Summary:**
Investigates how LLMs respond to safety-relevant prompts in
lower-resource versus higher-resource languages. Reports that LLMs
generate unsafe or irrelevant content more frequently in low-resource
languages, and respond to malicious prompts more often in those
languages.

**Intersection with MCH research:**
Predates MCH program. Provides safety-side empirical context for
Paper 8's findings on EFI degradation. The asymmetric safety failures
in low-resource languages are consistent with the encoding fidelity
gap Paper 8 measures.

**Citation status:**
- Cites MCH: No (predates)
- Cited by MCH: Could be added to Paper 8 v2 / future related work

---

## Assessing Translation Capabilities of Large Language Models Involving English and Indian Languages

**Venue:** arXiv 2311.09216
**Date:** November 2023
**Link:** https://arxiv.org/pdf/2311.09216

**Summary:**
Empirical assessment of LLM translation capabilities between English
and several Indian languages.

**Intersection with MCH research:**
Predates Paper 8. Translation-quality framing of issues that Paper 8
reframes as encoding fidelity at the input layer.

**Citation status:**
- Cites MCH: No (predates)
- Cited by MCH: Possible future citation
