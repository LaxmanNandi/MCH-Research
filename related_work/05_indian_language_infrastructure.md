# 05. Indian-Language AI Infrastructure

Indigenous Indic LLMs, datasets, and infrastructure being built for
Indian languages. These provide the substrate against which MCH's
Paper 8 and Paper 9 findings on encoding fidelity can be addressed.

---

## BharatGen

**Venue:** Government of India / academic consortium
**Date:** Announced 2024–2025, ongoing development
**Link:** https://www.drishtiias.com/daily-updates/daily-news-analysis/bharatgen-india-s-first-ai-multimodal-llm

**Summary:**
India's first indigenously developed, government-funded Multimodal
Large Language Model targeting 22 Indian languages. Objectives include
promoting ethical, inclusive, multilingual AI and providing
region-specific solutions in healthcare, agriculture, education, and
governance.

**Intersection with MCH research:**
Infrastructure work attempting to address the encoding-fidelity gap
that Paper 8 measures empirically. If BharatGen achieves better
EFI than MiniLM/MPNet for Indian languages, it would represent a
deployment-side response to the problem MCH characterises. Future
MCH work could use Paper 8's EFI methodology to benchmark BharatGen.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — likely citation in Paper 10 or any
  Indian-AI deployment piece

---

## Sarvam Models (Sarvam AI)

**Venue:** Sarvam AI (commercial / open-source releases)
**Date:** 2024–2026, ongoing
**Link:** https://www.sarvam.ai/models

**Summary:**
Indigenous Indian AI startup. Current flagship models are **Sarvam 30B**
and **Sarvam 105B**; **Sarvam-M** is listed as deprecated; **Sarvam-1**
was an earlier release (Oct 2024) since superseded. Saaras V3 voice
model supports 22 Indian languages. Trained on domestic AI
infrastructure. Active in voice AI for India.

**Intersection with MCH research:**
Direct candidate for benchmarking Paper 8's EFI methodology against
indigenous Indian models. The MCH measurements were performed on
MiniLM/MPNet (Western embeddings) and DeepSeek/Mistral (non-Indic
LLMs). Sarvam models trained primarily on Indian-language corpora
should show different EFI profiles, providing a key comparison.
Strong candidate for future Paper 10 benchmarking.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — strong candidate for citation

---

## AI4Bharat IndicLLMSuite

**Venue:** GitHub / IIT Madras open-source
**Date:** 2024–ongoing
**Link:** https://github.com/AI4Bharat/IndicLLMSuite

**Summary:**
Open-source blueprint and resources for creating pretraining and
fine-tuning datasets for Indic languages. Provides infrastructure
for building Indic LLMs.

**Intersection with MCH research:**
Infrastructure work that addresses the upstream data scarcity behind
Paper 8 / Paper 9 findings. AI4Bharat's IndicCorp, IndicNLP suite,
and related resources are the kind of datasets needed to improve
encoding fidelity at the model-training level.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Possible future citation when discussing remedies
  for EFI degradation

---

## Krutrim LLM: Multilingual Foundational Model for over a Billion People

**Venue:** arXiv 2502.09642
**Date:** February 2025
**Link:** https://arxiv.org/html/2502.09642

**Summary:**
Multilingual foundational LLM developed for Indian languages,
explicitly positioned for Indian deployment context.

**Intersection with MCH research:**
Another candidate Indian-built LLM for Paper 8 EFI benchmarking.
Together with Sarvam and BharatGen, forms a triad of indigenous
Indic LLMs that would be natural comparison points.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Possible future citation

---

## NVIDIA Enterprise Work on Indian-Language LLMs (Vishal Dhupar)

**Venue:** NVIDIA blog / industry
**Date:** October 23, 2024
**Link:** https://blogs.nvidia.com/blog/llms-indian-languages/

**Summary:**
Industry/infrastructure-side support for enterprises building
LLMs for Indian languages. Provides compute and model-engineering
support.

**Intersection with MCH research:**
Industry context. Not currently a direct citation candidate but
useful background for understanding the deployment landscape.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently relevant

---

## Building Voice AI Agents for India with Sarvam and Vision Agents (GetStream)

**Venue:** GetStream blog / Sarvam integration
**Date:** April 15, 2026
**Link:** https://getstream.io/blog/sarvam-integration/

**Summary:**
Integration of Sarvam voice AI models into agentic frameworks for
Indian-language voice applications.

**Intersection with MCH research:**
Relevant to MCH's future voice-agent direction (discussed in
session notes but not yet a paper). Voice transcription failures
in clinical Kannada would represent a natural extension of Paper 8
to the voice modality.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Possible future citation if MCH moves into voice

---

## Building Pre-train LLM Dataset for the Indic Languages: A Case Study on Hindi

**Venue:** arXiv 2407.09855
**Date:** July 2024
**Link:** https://arxiv.org/html/2407.09855v1

**Summary:**
Methodology paper on constructing pretraining datasets for Hindi,
applicable to other Indic languages. Pre-MCH timing.

**Intersection with MCH research:**
Predates MCH. Infrastructure-side context. The data construction
methods described here would, if applied at scale, affect the
encoding fidelity Paper 8 measures.

**Citation status:**
- Cites MCH: No (predates)
- Cited by MCH: Possible future citation
