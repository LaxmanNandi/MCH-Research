# 02. Multi-Turn Evaluation and Context Sensitivity

Work on how LLM behaviour changes across multiple turns of conversation,
including context-sensitivity metrics, drift detection, and degradation
patterns. Most directly related to MCH Papers 1, 3, 4, 7.

---

## LLMs Get Lost In Multi-Turn Conversation (Laban et al.)

**Venue:** arXiv 2505.06120 / OpenReview
**Date:** May 2025
**Link:** https://arxiv.org/abs/2505.06120 ; https://openreview.net/forum?id=VKGTGGcwl6

**Summary:**
Large-scale study of LLM performance degradation across multi-turn
conversation. Analysis of 200,000+ simulated conversations decomposes
performance loss into a minor "aptitude" component and a significant
"unreliability" component. All top open- and closed-weight LLMs tested
exhibit significantly lower performance in multi-turn than single-turn
conditions, with an average drop of 39% across six generation tasks.

**Intersection with MCH research:**
Independent large-scale empirical observation of multi-turn degradation
in production LLMs. Paper 4 (published February 22, 2026 on Preprints.org)
cites this work and frames its ΔRCI~VRI correlation (r=0.76,
p=2.37×10⁻⁶⁸, N=360) as a candidate structural account of the
"unreliability" component Laban et al. report. Whether the entanglement
framework fully explains the Laban et al. observations is a hypothesis
that requires further work.

**Citation status:**
- Cites MCH: No (predates Paper 4)
- Cited by MCH: Yes — Paper 4

---

## Evaluating the Sensitivity of LLMs to Prior Context (Hankache et al.)

**Venue:** arXiv 2506.00069
**Date:** May 29, 2025
**Link:** https://arxiv.org/html/2506.00069v1

**Summary:**
Introduces new benchmarks derived from GPQA Diamond to evaluate LLM
sensitivity to prior context across multi-turn settings. Reports
performance drops of up to 73% for some models, 32% for GPT-4o.
Strategic placement of task description within the context can
mitigate drops by as much as a factor of 3.5.

**Intersection with MCH research:**
Independent parallel work to MCH Papers 1, 3, 4. Same metric family
(context sensitivity in multi-turn) on a different benchmark domain
(general scientific QA rather than philosophy/medical). The ΔRCI
formulation from Paper 1 is conceptually adjacent to their sensitivity
measure. Useful comparison point when Paper 4 finds its next venue.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet — strong candidate for citation

---

## Drift No More? Context Equilibria in Multi-Turn LLM Interactions

**Venue:** arXiv 2510.07777
**Date:** Late 2025 / early 2026
**Link:** https://arxiv.org/pdf/2510.07777

**Summary:**
Studies context drift as slow erosion of user intent across turns
(e.g. a summariser gradually losing requested tone). Analyses
equilibrium conditions where drift stabilises versus continues.

**Intersection with MCH research:**
Directly engages with the phenomenon Paper 4 frames as
entanglement-driven variance amplification. The "equilibrium"
framing is conceptually similar to Paper 6's conservation constraint
(K = ΔRCI × Var_Ratio approximately conserved within a domain).
Worth examining whether their equilibria correspond to MCH's K values.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet — candidate for citation

---

## Eliciting Behaviors in Multi-Turn Conversations

**Venue:** arXiv 2512.23701
**Date:** Late 2025 / 2026
**Link:** https://arxiv.org/pdf/2512.23701

**Summary:**
Proposes new evaluation metrics for multi-turn conversations using
static test cases produced by LLMs with human-in-the-loop validation.

**Intersection with MCH research:**
Methodological neighbour to the three-condition (TRUE/COLD/SCRAMBLED)
protocol introduced in Paper 1. Different methodology, same family
of question (how do you measure multi-turn behaviour reliably).

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet

---

## DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift in LLMs

**Venue:** arXiv 2602.16935
**Date:** 2026
**Link:** https://arxiv.org/html/2602.16935v1

**Summary:**
Adversarial framing of multi-turn drift. Detects when users or agents
manipulate context across turns to exploit reasoning capabilities or
context-window limits. Stateful, real-time detection approach.

**Intersection with MCH research:**
Different angle on multi-turn behaviour than MCH (safety/adversarial
rather than behavioural-science). But the underlying phenomenon —
context manipulation across turns producing behavioural shifts — is
adjacent to the variance and entanglement effects MCH measures.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet

---

## Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models

**Venue:** arXiv 2504.04717
**Date:** April 2025
**Link:** https://arxiv.org/pdf/2504.04717

**Summary:**
Survey of multi-turn interaction research. Addresses cross-turn
coherence, state tracking, adaptation to evolving user intent, and
dynamic/context-dependent challenges.

**Intersection with MCH research:**
Survey-level mapping of the field MCH Papers 1, 3, 4, 7 contribute
to. Useful for situating MCH work in the broader literature when
writing future papers.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Possible future citation in review-style writing

---

## Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey

**Venue:** arXiv 2503.22458
**Date:** March 2025
**Link:** https://arxiv.org/pdf/2503.22458

**Summary:**
Survey identifying key components of LLM-based agents for multi-turn
conversations and their evaluation dimensions: task completion,
response quality, user experience, memory and context retention,
planning and tool integration.

**Intersection with MCH research:**
Maps the evaluation-methodology landscape that MCH metrics
(ΔRCI, VRI, K) contribute to. The "context retention" dimension is
particularly close to MCH's ΔRCI framing.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Possible future citation

---

## Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges

**Venue:** arXiv 2508.00454
**Date:** August 2025
**Link:** https://arxiv.org/abs/2508.00454

**Summary:**
Method for learning a multi-turn dialogue evaluator from multiple
LLM judges, addressing inconsistency among individual judge models.

**Intersection with MCH research:**
Methodological work on the evaluator side rather than the evaluand
side. Complements MCH's metric-based evaluation with a learned
evaluator approach.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Not currently relevant

---

## Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ

**Venue:** arXiv 2601.16508
**Date:** Late 2025 / 2026
**Link:** https://arxiv.org/pdf/2601.16508

**Summary:**
Empirical test of whether conversation length itself degrades model
performance, using BoolQ as the testbed. Investigates length as an
independent variable.

**Intersection with MCH research:**
Length is one of several factors MCH controls for through the
30-position protocol. The question they isolate is one factor in
the broader context-position landscape MCH Paper 3 maps.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Not yet

---

## Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction

**Venue:** arXiv 2604.04325
**Date:** 2026
**Link:** https://arxiv.org/pdf/2604.04325

**Summary:**
Benchmark for multi-turn medical diagnosis tasks, with attention to
"hold" (sustaining clinical context), "lure" (resisting misleading
prompts), and "self-correction" capabilities.

**Intersection with MCH research:**
Closest combined intersection of MCH territories: multi-turn (Papers
1, 4) AND medical/clinical (Papers 5, 8). The "hold" capability they
measure relates to ΔRCI dynamics across the clinical-question
trajectory. Strong candidate for citation in any clinical-AI follow-up.

**Citation status:**
- Cites MCH: Not verified
- Cited by MCH: Not yet — strong candidate

---

## MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs (Sirdeshmukh, Deshpande et al., Scale AI)

**Venue:** ACL Findings 2025 (also arXiv 2501.17399, Jan 2025)
**Date:** 2025
**Link:** https://aclanthology.org/2025.findings-acl.958.pdf

**Summary:**
Realistic multi-turn conversation benchmark for evaluation. Constructs
test cases reflecting deployed-system conversation patterns rather
than artificial probing.

**Intersection with MCH research:**
Methodological neighbour to MCH's three-condition protocol. Different
realism axis — they prioritise ecological validity; MCH prioritises
controlled-variable rigour.

**Citation status:**
- Cites MCH: Not yet
- Cited by MCH: Not yet
