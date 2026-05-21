# Timeline — Field Convergence (2024 – 2026)

Chronological view of how the research areas surrounding the MCH
program emerged and converged. Useful for situating MCH papers
within the broader field timeline.

---

## 2024

- **January 2024** — "The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts" (arXiv 2401.13136). Early documentation of multilingual safety gap.
- **July 2024** — "Building Pre-train LLM Dataset for the Indic Languages: A Case Study on Hindi" (arXiv 2407.09855). Infrastructure-side work.
- **September 2024** — "Safety Challenges of AI in Medicine in the Era of LLMs" (arXiv 2409.18968). Clinical safety mapping.
- **2024 (ongoing)** — BharatGen announced. Sarvam AI founded. AI4Bharat releases IndicLLMSuite.

## Early 2025

- **February 2025** — Krutrim LLM released (arXiv 2502.09642).
- **March 2025** — "Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey" (arXiv 2503.22458).
- **April 2025** — "Beyond Single-Turn: A Survey on Multi-Turn Interactions" (arXiv 2504.04717).
- **May 9, 2025** — "LLMs Get Lost In Multi-Turn Conversation" (Laban et al., arXiv 2505.06120). **Foundational empirical finding** that multi-turn degrades by 39% on average across 6 generation tasks. Paper 4 will later provide mechanistic explanation.
- **May 29, 2025** — "Evaluating the Sensitivity of LLMs to Prior Context" (Hankache et al., arXiv 2506.00069). Up to 73% performance drops in multi-turn.
- **May 30, 2025** — "The State of Multilingual LLM Safety Research" (arXiv 2505.24119). Survey documenting the language gap.
- **July 2025** — Chain of Thought Monitorability statement (arXiv 2507.11473). Joint statement from OpenAI, DeepMind, Anthropic, Meta researchers.
- **July 2025** — Sparse Autoencoders for language-specific concepts (arXiv 2507.11230).
- **August 2025** — "Learning an Efficient Multi-Turn Dialogue Evaluator" (arXiv 2508.00454).
- **August 2025** — "Simplification and Translation of Medical Reports for Indian Context" (Preprints.org).

## Late 2025 — MCH Program preparatory phase

- **March 2025** — Dr Laxman M M's early submission attempts (ECOF framework, declined).
- **July 2025** — Mirror-Consciousness Hypothesis submission (declined).
- **September 2025** — Same paper to JAIC (declined with notable reviewer feedback).
- **September 2025 – January 2026** — Silent regrouping. Shift from theoretical frameworks to empirical measurement.

## January – February 2026 — MCH Program begins

- **January 26, 2026** — **MCH Paper 1** published (Preprints.org). ΔRCI introduced.
  *Context curves behavior*.
- **February 12, 2026** — **MCH Paper 2** published. 14 models, 25 model-domain runs.
- **February 16, 2026** — **MCH Paper 3** published. Temporal dynamics.
- **February 22, 2026** — **MCH Paper 4** published. VRI mechanism, r=0.76. Provides mechanistic explanation for Laban et al. findings.
- **February 28, 2026** — **MCH Paper 5** published. Safety taxonomy IDEAL/EMPTY/DIVERGENT/RICH.

## March – April 2026 — MCH continues, field accelerates

- **March 2026** — Berkeley commentary on emergent misalignment.
- **March 16, 2026** — **MCH Paper 7** published. Content-order decomposition.
- **April 2, 2026** — **MCH Paper 8** published. EFI metric. Coherent Misalignment named.
- **April 7, 2026** — **MCH Paper 9** published (Zenodo). Measurement Matters.
- **April 9, 2026** — **MCH Paper 6 (Capstone)** submitted to Preprints.org. Conservation constraint K across four domains.
- **April 2026** — Anthropic Opus 4.7 released with tokenizer update (1.0–1.35× token growth). (User-observed; not corroborated by a public Anthropic announcement at time of writing.)
- **May 7, 2026** — **Anthropic Natural Language Autoencoders** released. Activation-level fidelity reconstruction. Anthropic acknowledges multilingual response failures in Opus 4.6.

## Late 2025 / 2026 — Parallel work in same territory

- **arXiv 2510.07777** — "Drift No More? Context Equilibria in Multi-Turn LLM Interactions". Equilibrium framing of multi-turn drift.
- **arXiv 2510.01288** — "Microsaccade-Inspired Probing" (Melo, Abreu, Pasareanu; Oct 2025).
- **arXiv 2512.10780** — "Script Gap" paper on LLM triage in Indian languages, native vs Romanized scripts. Reports nearly 2 million excess errors.
- **arXiv 2512.23701** — "Eliciting Behaviors in Multi-Turn Conversations". New evaluation metrics.
- **arXiv 2601.06047** — "Misalignment as structural fidelity in LLMs". Conceptual neighbour to Coherent Misalignment.
- **arXiv 2601.23045** — "The Hot Mess of AI: How Does Misalignment Scale" (v2).
- **arXiv 2602.15038** — **"Indic-TunedLens"**. Documents **Latent Romanization** — middle-layer mechanism that may explain Paper 8's EFI findings.
- **arXiv 2602.16935** — "DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift".
- **arXiv 2604.04325** — "Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction".
- **EACL 2026** — "When Benchmarks Age: Temporal Misalignment Through Fidelity".
- **Springer 2026** — **IndiHealthBench**. 11,000 parallel sentence pairs across 13 Indian languages for clinical translation.
- **Nature 2026** — Training LLMs on narrow tasks leads to broad misalignment.

## May 2026 (current)

- MCH Paper 8 with editor at Scientific Reports (Nature portfolio).
- Field continues to converge on encoding fidelity, multi-turn evaluation,
  Indian-language clinical AI, and fidelity-as-safety frames.
- This `related_work/` folder created to track the convergence.

---

## Observations on the convergence pattern

These observations describe the shape of the field as of May 2026.
They are descriptive of multi-group convergence rather than positioning
claims about any single contributor.

1. **Multi-turn evaluation** emerged as a distinct subfield through 2025,
   with Laban et al. (May 2025) providing foundational empirical scope
   (200,000+ conversations, 39% average degradation), Hankache et al.
   (May 2025) extending to context-sensitivity measurement on GPQA,
   and various survey work (arXiv 2503.22458, 2504.04717) consolidating
   methodology. MCH Papers 1 and 4 sit within this subfield, contributing
   the ΔRCI metric and the VRI/entanglement framework respectively.

2. **Indian-language clinical AI and interpretability** consolidated
   through late 2025 / early 2026 across multiple independent groups:
   IndiHealthBench (PReMI 2025), MILA (OpenReview), Indic-TunedLens
   (arXiv 2602.15038, Jan 2026), Script Gap (arXiv 2512.10780), Sarvam,
   BharatGen, AI4Bharat infrastructure work, and MCH Papers 8 and 9.
   The work spans input-layer encoding measurement, intermediate-layer
   probing, large-scale deployment evaluation, and dataset/model
   construction.

3. **Fidelity-as-frame for AI safety** crystallised through 2026 across
   layers: chain-of-thought monitorability at the reasoning layer
   (Korbak et al., July 2025); natural-language autoencoders at the
   activation layer (Anthropic, May 2026); intermediate-layer
   interpretability (Indic-TunedLens, Jan 2026); input-layer encoding
   fidelity (MCH Paper 8, April 2026); benchmark factuality drift
   (EACL 2026). Different layers, shared methodological move:
   operationalise an abstract property as a quantitative metric.

4. **Misalignment as structural property** gained ground with the
   Nature 2026 narrow-task work on broad misalignment from narrow
   training, the "Hot Mess of AI" scaling analysis, the philosophical
   essay framing misalignment as structural fidelity, and MCH Paper 8's
   Coherent Misalignment naming. Convergent move: treating misalignment
   as structural rather than instance-specific.

The field state in May 2026 is one of active multi-group convergence
on overlapping problem-territory. A synthesis paper that ties these
threads together has not yet appeared.
