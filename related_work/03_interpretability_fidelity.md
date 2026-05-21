# 03. Interpretability and Fidelity Measurement

Work that uses fidelity-based metrics or reconstruction techniques to
probe LLM internals. Most directly related to MCH Paper 8 (EFI) and
Paper 9 (Measurement Matters).

---

## Natural Language Autoencoders (Anthropic)

**Venue:** Anthropic Research Blog + Transformer Circuits technical report
**Date:** May 7, 2026
**Link:** https://www.anthropic.com/research/natural-language-autoencoders ;
https://transformer-circuits.pub/2026/nla/

**Summary:**
Method translating internal LLM activations into readable natural-language
descriptions and reconstructing activations from those descriptions.
Quality is validated through reconstruction fidelity. Activation
verbalizer (AV) and activation reconstructor (AR) jointly trained via
RL. Anthropic reports that NLAs helped trace why an early Claude Opus
4.6 sometimes answered English prompts in other languages — back to
specific training-data artifacts.

**Intersection with MCH research:**
Structurally analogous to Paper 8's EFI at a different layer of the
stack. Where EFI measures fidelity between languages at the input
embedding boundary, NLAs measure fidelity between activations and
natural-language descriptions at the residual-stream level. Anthropic's
multilingual case (English query → other-language response) is exactly
the Coherent Misalignment phenomenon Paper 8 names at the behavioural
level. NLA provides the interpretability tool that could trace the
internal cause of MCH-observed external failures.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — strong candidate for Paper 10 / future
  perspective piece

---

## Indic-TunedLens: Interpreting Multilingual Models in Indian Languages (Panchal, Varshney, Mamta, Ekbal)

**Venue:** arXiv 2602.15038
**Date:** January 29, 2026 (v1); v2 February 18, 2026
**Link:** https://arxiv.org/pdf/2602.15038

**Summary:**
Interpretability framework adapted for Indian languages. Key empirical
finding: **LLMs represent non-Roman script languages in Romanized form
in intermediate layers**, a phenomenon the authors call **Latent
Romanization**.

**Intersection with MCH research:**
Independent and roughly contemporaneous work investigating Indian-language
representation in multilingual LLMs. Indic-TunedLens (Jan 2026) and
MCH Paper 8 (April 2026) approach the same problem from different
layers: Indic-TunedLens probes intermediate-layer representations and
documents Latent Romanization; Paper 8 measures input-layer encoding
fidelity (EFI) and behavioural output variance. The two findings, if
both robust, are consistent — a model that internally transliterates
non-Roman script to Roman in middle layers would plausibly exhibit
the input-layer encoding degradation Paper 8 measures externally.
Neither work depends on the other; they are convergent.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet. Strong candidate for citation in any
  future work that bridges input-layer and middle-layer perspectives
  on Indic-language LLM behaviour.

---

## Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours (Melo, Abreu, Pasareanu)

**Venue:** arXiv 2510.01288
**Date:** October 1, 2025
**Link:** https://arxiv.org/pdf/2510.01288

**Summary:**
Probing method that perturbs positional encodings to amplify
behavioural deviations, making them detectable without task-specific
supervision. Borrows from neuroscience the analogy of microsaccades
revealing visual processing through small involuntary perturbations.

**Intersection with MCH research:**
Methodological parallel to MCH's SCRAMBLED condition. Paper 1's
three-condition protocol uses scrambled ordering to perturb context
and reveal sensitivity. Microsaccade probing uses positional-encoding
perturbations to reveal latent behaviour. Different perturbation
sites, same probing logic.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet

---

## Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages

**Venue:** arXiv 2507.11230
**Date:** July 2025
**Link:** https://arxiv.org/abs/2507.11230

**Summary:**
Demonstrates that sparse autoencoders trained on multilingual model
activations can isolate language-specific concept representations.
Provides interpretability infrastructure for understanding how
multilingual models internally represent language identity and
language-specific concepts.

**Intersection with MCH research:**
Interpretability-side complement to Paper 8 and 9. SAEs could
potentially be used to identify which features distinguish a
high-EFI from a low-EFI encoding, providing internal grounding
for the external fidelity measurements MCH reports.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — candidate for citation

---

## Tracing the Thoughts of a Large Language Model (Anthropic)

**Venue:** Anthropic Research Blog
**Date:** 2025
**Link:** https://www.anthropic.com/research/tracing-thoughts-language-model

**Summary:**
Anthropic interpretability work tracing how LLMs internally process
information. Foundational work for the subsequent NLA approach.

**Intersection with MCH research:**
Provides interpretability infrastructure that MCH-style behavioural
measurements can be aligned with. The "thoughts" Anthropic traces
internally would, in MCH framing, produce the ΔRCI patterns observed
externally.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — candidate for citation in synthesis pieces

---

## Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer-wise Pooled Representations

**Venue:** EMNLP 2025
**Date:** 2025
**Link:** https://aclanthology.org/2025.emnlp-main.145.pdf

**Summary:**
Proposes an intrinsic alignment metric (AQI) balancing fine-grained
alignment fidelity with macro-level latent organisation. Argues that
"refusal" behaviour is a coarse alignment signal and that finer
metrics are needed.

**Intersection with MCH research:**
Fidelity-based metric in the same family as MCH's EFI and Paper 4's
VRI. Different target (alignment quality vs encoding/variance) but
shared methodological move: use a single quantitative index to
summarise a behavioural property that was previously qualitative.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — candidate for citation in any
  metric-comparison piece

---

## When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation (Jiang, Chang, McAuley, Xu)

**Venue:** EACL 2026 (short paper)
**Date:** 2026
**Link:** https://aclanthology.org/2026.eacl-short.37.pdf

**Summary:**
Measures temporal misalignment in LLMs as benchmarks age, using
factuality evaluation (model alignment with benchmark gold answers).

**Intersection with MCH research:**
Adjacent measurement-of-LLM-behaviour work in the broader convergence
MCH is part of. Different target (temporal benchmark drift) but
shares the move of operationalising an abstract property as a
quantitative metric.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently directly relevant
