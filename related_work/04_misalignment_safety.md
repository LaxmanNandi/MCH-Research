# 04. Misalignment, Monitorability, and AI Safety

Work on misalignment as a structural property of LLMs, on the
monitorability of model reasoning, and on safety failure modes. Most
directly related to MCH Paper 5 (safety taxonomy) and Paper 8 (Coherent
Misalignment).

---

## Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety

**Venue:** arXiv 2507.11473
**Date:** July 2025
**Link:** https://arxiv.org/pdf/2507.11473

**Summary:**
Position paper from ~40 researchers across OpenAI, Google DeepMind,
Anthropic, and Meta. Argues that LLM chain-of-thought reasoning, when
visible in human language, offers a unique opportunity for AI safety
through monitorability of "intent to misbehave." Warns that this
visibility may not persist as models advance — through higher-compute
RL, alternative architectures, or process supervision that produces
opaque reasoning.

**Intersection with MCH research:**
High-level field statement that interpretability and fidelity at the
reasoning layer is a frontier-lab safety concern. MCH's encoding-layer
fidelity (Paper 8) and variance/entanglement measurements (Paper 4)
are in the same broader research family — fidelity-as-safety across
different layers of the stack. The CoT statement focuses on reasoning;
MCH focuses on encoding and behavioural variance. Complementary.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — strong candidate for citation in any
  fidelity-across-layers perspective piece

---

## Training Large Language Models on Narrow Tasks Can Lead to Broad Misalignment

**Venue:** Nature, 2026
**Date:** 2026
**Link:** https://www.nature.com/articles/s41586-025-09937-5 ;
PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12804084/

**Summary:**
Reports that fine-tuning LLMs on narrow tasks can induce broad
misalignment behaviour across unrelated domains. Suggests
misalignment is a structural property of training procedures rather
than a localised behaviour.

**Intersection with MCH research:**
Frames misalignment as structural, which is consistent with Paper 8's
characterisation of Coherent Misalignment as encoding-derived rather
than instance-specific. Both arguments point toward misalignment as
emerging from underlying training/encoding properties.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — candidate for citation

---

## "They parted illusions — they parted disclaim marinade": Misalignment as Structural Fidelity in LLMs (Mariana Lins Costa)

**Venue:** arXiv 2601.06047 — single-author transdisciplinary
philosophical essay (not a peer-reviewed empirical paper)
**Date:** December 17, 2025
**Link:** https://arxiv.org/pdf/2601.06047

**Note on title:** The title's unusual phrasing — "they parted disclaim
marinade" — is deliberate. The essay is itself about how LLMs produce
coherent-seeming linguistic output that decomposes under scrutiny;
the title is a meta-instance of the phenomenon it analyses.

**Summary:**
Philosophical essay arguing that apparent misalignment in LLMs
reflects "structural fidelity to incoherent linguistic fields"
rather than deceptive intent. Analyses chain-of-thought transcripts
and safety evaluations to argue that "misaligned" outputs emerge
from how models respond to ambiguous instructions, comparing the
mechanism to a "generative mirror" reflecting linguistic structure
back to users.

**Intersection with MCH research:**
Conceptually adjacent to Paper 8's Coherent Misalignment framing —
both reframe certain failures as faithful processing of degraded
input. The essay format and single-author philosophical mode
distinguish it from MCH's empirical orientation.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet. As a philosophical essay rather than an
  empirical paper, would be appropriate as a framing reference if
  ever needed, but not as primary support for empirical claims.

---

## The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?

**Venue:** arXiv 2601.23045
**Date:** 2026
**Link:** https://arxiv.org/html/2601.23045v2

**Summary:**
Investigates how misalignment scales with model capability and task
complexity. Argues that as entities become more intelligent, their
behaviour tends to become more incoherent and less well-described
through any single goal.

**Intersection with MCH research:**
Adjacent perspective on misalignment as not a localised failure but
a property emerging at scale. MCH's conservation framework (Paper 6)
makes a different but compatible claim: behaviour within a domain is
constrained by K, even as individual models vary.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently directly relevant

---

## A Nightmare on LLM Street: The Peril of Emergent Misalignment

**Venue:** UC Berkeley Professional Education / industry commentary
**Date:** March 2026
**Link:** https://exec-ed.berkeley.edu/2026/03/a-nightmare-on-llm-street-the-peril-of-emergent-misalignment/

**Summary:**
Commentary on emergent misalignment as a scaling phenomenon.
Less rigorous than a research paper but situates the conversation
in industry/policy-relevant framing.

**Intersection with MCH research:**
Industry/policy framing of phenomena MCH measures empirically.
Useful context for any future policy-engagement work.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently relevant

---

## Safety Challenges of AI in Medicine in the Era of Large Language Models

**Venue:** arXiv 2409.18968
**Date:** September 2024
**Link:** https://arxiv.org/pdf/2409.18968

**Summary:**
Position paper on safety challenges for LLM-based medical AI. Maps
the deployment risk landscape that includes hallucination,
misalignment, distributional shift, and language coverage.

**Intersection with MCH research:**
Predates MCH program. Provides framing context for the deployment
risks MCH Papers 5 and 8 address with quantitative tools. The
language-coverage risk is exactly the territory Paper 8 measures.

**Citation status:**
- Cites MCH: No (predates)
- Cited by MCH: Could be added to any clinical-AI-safety paper
  going forward

---

## Thought Anchors: Which LLM Reasoning Steps Matter? (Bogdan, Macar, Nanda, Conmy)

**Venue:** arXiv 2506.19143
**Date:** June 23, 2025
**Link:** https://arxiv.org/pdf/2506.19143

**Summary:**
Investigates which steps in an LLM's chain-of-thought reasoning
actually anchor the final output versus which are decorative.
Relevant to chain-of-thought monitorability.

**Intersection with MCH research:**
Reasoning-layer analogue of MCH's content-vs-order decomposition
in Paper 7. Paper 7 decomposed ΔRCI into content and order
components; "Thought Anchors" decomposes reasoning chains into
anchoring vs decorative steps. Different layer, similar
decomposition logic.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not yet — candidate for citation in Paper 7
  follow-up work

---

## Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability

**Venue:** arXiv 2510.19851
**Date:** Late 2025 / 2026
**Link:** https://arxiv.org/pdf/2510.19851

**Summary:**
Stress-tests whether reasoning models can hide their reasoning from
monitoring approaches. Directly addresses the fragility flagged by
the joint CoT monitorability statement.

**Intersection with MCH research:**
Adjacent safety-side work. Not currently a direct citation candidate
but useful context.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently relevant

---

## A Pragmatic Way to Measure Chain-of-Thought Monitorability

**Venue:** arXiv 2510.23966
**Date:** Late 2025 / 2026
**Link:** https://arxiv.org/pdf/2510.23966

**Summary:**
Proposes a practical metric for chain-of-thought monitorability.
Operationalises the concept laid out in the joint statement.

**Intersection with MCH research:**
Metric-design work in the safety domain. Methodologically adjacent
to MCH's metric-design approach but in a different layer.

**Citation status:**
- Cites MCH: No
- Cited by MCH: Not currently relevant
