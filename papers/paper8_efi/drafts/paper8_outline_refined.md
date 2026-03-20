# Paper 8: Encoding Fidelity and Coherent Misalignment
## Refined Outline — All Citations Verified (March 19, 2026)

**Title:** "Encoding Fidelity and Coherent Misalignment: Why Shannon's Channel Model Breaks for Non-English Clinical AI"

**Author:** Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

---

## Abstract

Shannon's Mathematical Theory of Communication (1948) assumes encoding fidelity — that the encoder preserves the statistical structure of the source. Large Language Models violate this assumption systematically for non-English languages, producing outputs that are internally consistent but semantically degraded. We call this failure mode *Coherent Misalignment* and introduce the Encoding Fidelity Index (EFI) to measure it. Using the MCH Research Program's conservation framework (ΔRCI × Var_Ratio ≈ K), we show that conservation holds even when encoding fidelity collapses — K ⊥ Truth. Across 4 languages (English, Kannada, Tamil, Hindi), 2 embedding models (384D, 768D), and 2 LLMs (DeepSeek V3.1, Mistral Small 24B), we find: (1) EFI degrades ~90% for all non-English Indian languages (p = 1.3 × 10⁻¹³), independent of language family; (2) output variance amplifies 2–3× through the encoding bottleneck (Tamil VR = 2.85×, p = 0.004); (3) complex medical sentences show paradoxical EFI increase from English loanword anchoring; (4) the conservation law K(domain) holds normally even when outputs are semantically wrong. These findings demonstrate that self-consistency metrics are blind to encoding degradation, with direct implications for clinical AI deployment serving 1.5 billion+ non-English speakers.

---

## 1. Introduction

### 1.1 The Clinical Reality
- Practicing physician at PHC Manchi uses LLMs daily in Kannada for clinical decision support.
- Observation: Responses are grammatically correct but semantically unreliable in non-English languages.
- This paper identifies the mechanism and measures the degradation.

### 1.2 The MCH Conservation Framework (Brief)
- Papers 1–7 established ΔRCI, Var_Ratio, VRI, and the conservation law K = ΔRCI × Var_Ratio ≈ constant across domains [self-citations 1–7].
- These metrics measure *output behavior* — context sensitivity and variance structure.
- Missing piece: What happens at the *input encoding* layer?

### 1.3 Shannon's Unexamined Assumption
- Shannon (1948) assumed the encoder preserves source statistics [1].
- The information bottleneck framework (Tishby & Zaslavsky, 2015) extended this to deep learning but retained the fidelity assumption [2].
- LLM tokenizers demonstrably violate this: token fertility varies 2–15× across languages (Petrov et al., 2023) [3], with downstream accuracy drops of ~10 pp (Alonso et al., 2024) [4].
- **Gap:** No prior work measures encoding fidelity directly or connects it to output variance structure.

### 1.4 Contributions
1. Introduce EFI as a practical encoding fidelity proxy.
2. Demonstrate ~90% EFI degradation for 3 Indian languages across 2 embedding models.
3. Show degradation is language-family independent (Dravidian ≈ Indo-Aryan).
4. Document 2–3× variance amplification for non-English clinical inputs across 2 LLMs.
5. Identify English loanword anchoring as a paradoxical partial mitigation.
6. Establish K ⊥ Truth — conservation is orthogonal to semantic correctness.
7. Define Coherent Misalignment as a distinct failure mode.

---

## 2. Related Work

### 2.1 Information Theory Foundations
- Shannon (1948): source coding, channel capacity, implicit encoding fidelity assumption [1].
- Tishby & Zaslavsky (2015): information bottleneck principle for deep learning [2].
- Shwartz-Ziv & Tishby (2017): information plane dynamics in DNNs [5].
- Delétang et al. (2024): language modeling as compression — ICLR 2024 [6].

### 2.2 Tokenizer Inequality
- Petrov et al. (2023): tokenizers introduce unfairness between languages — NeurIPS 2023 [3].
- Rust et al. (2021): monolingual tokenizer performance shapes downstream accuracy — ACL 2021 [7].
- Lundin et al. (2025): "The Token Tax" — 2× tokens → 4× compute cost for African languages [8].
- Ali et al. (2024): tokenizer choice shapes LLM training outcomes — NAACL 2024 Findings [9].

### 2.3 Cross-Lingual Representation
- Hammerl et al. (2024): survey of cross-lingual alignment methods — ACL 2024 Findings [10].
- Peng & Søgaard (2024): concept space alignment in multilingual LLMs — EMNLP 2024 [11].
- Pallucchini et al. (2025): survey of cross-lingual alignment for contextualized representations — ACM Computing Surveys [12].

### 2.4 Multilingual Medical AI
- Alonso et al. (2024): MedExpQA — accuracy drops ~10 pp for non-English medical QA [4].
- Qiu et al. (2024): multilingual medical LLM — Nature Communications [13].
- Asgari et al. (2025): clinical safety and hallucination framework — npj Digital Medicine [14].
- WHO (2024): ethics and governance of AI for health, large multi-modal models guidance [15].

### 2.5 Multilingual Benchmarks
- Xuan et al. (2025): MMLU-ProX — up to 24.3% gap across 29 languages — EMNLP 2025 [16].
- Adelani et al. (2025): IrokoBench — African language evaluation, Outstanding Paper NAACL 2025 [17].

### 2.6 Alignment and Failure Modes
- Hubinger et al. (2019): deceptive alignment — goal misrepresentation in learned optimizers [18].
- Ardoin et al. (2025): latent directions of confabulation — EMNLP 2025 [19].
- **Distinction:** Coherent Misalignment (this paper) is neither deceptive nor confabulatory — it is encoding-level degradation producing internally consistent but semantically wrong outputs.

### 2.7 Dravidian Language NLP
- Chakravarthi et al. (2022): DravidianCodeMix — sentiment/offensive language in Tamil, Kannada, Malayalam [20].

---

## 3. Theoretical Framework

### 3.1 Shannon's Channel Model and Its Assumption
- Standard diagram: Source → Encoder → Channel → Decoder → Destination.
- Shannon proved optimal coding exists *given* the encoder preserves source statistics.
- For LLMs, the "encoder" includes tokenizer + embedding layer + early attention. This composite encoder does not preserve source statistics equally across languages.

### 3.2 Encoding Fidelity Index (EFI)
- **Theoretical definition:** EFI = I(X; X̂) / H(X), where X is source message, X̂ is model's internal representation.
- **Practical proxy:** EFI_proxy = cos_sim(embed(non-English), embed(English_equivalent)).
- EFI = 1.0 for English (by construction — training distribution).
- Limitation: proxy measures embedding similarity, not true mutual information. Validated by correlation with downstream task degradation.

### 3.3 Coherent Misalignment
- When EFI << 1.0, the model operates on a degraded representation of the input.
- Outputs are: grammatically correct, internally consistent, confidently delivered — and semantically wrong.
- Distinct from: hallucination (confabulation of facts), deceptive alignment (goal misrepresentation), sycophancy (user-pleasing bias).
- The system cannot detect its own misalignment because all internal consistency checks pass.

### 3.4 K ⊥ Truth
- From Papers 2–6: K(domain) = ΔRCI × Var_Ratio ≈ constant.
- K measures the variance *structure* of responses, not their semantic content.
- Therefore K is orthogonal to truth — conservation holds whether outputs are correct or not.
- Implication: no self-consistency metric can detect Coherent Misalignment. External ground truth is required.

### 3.5 Variance Amplification Through the Encoding Bottleneck
- Low EFI → noisier internal representation → higher output variance.
- Predicted: VR_nonEnglish ∝ 1/EFI (inverse relationship).
- Measured: EFI ≈ 0.08, VR ≈ 2.5× — consistent with amplification through degraded encoding.

---

## 4. Methods

### 4.1 Clinical Sentence Battery
- 3 complexity levels × 5 sentences × 4 languages (English, Kannada, Tamil, Hindi).
- Simple: single symptoms ("The patient has fever" / "ರೋಗಿಗೆ ಜ್ವರ ಇದೆ").
- Medium: multi-symptom descriptions.
- Complex: diagnostic reasoning with English medical terms (ST elevation, D-dimer, etc.).
- Translations verified by native-speaking clinician (author).

### 4.2 EFI Measurement (Experiments 1–3)
- Embedding models: all-MiniLM-L6-v2 (384D), all-mpnet-base-v2 (768D).
- EFI_proxy = cosine similarity of non-English embedding to English equivalent.
- Statistical tests: Mann-Whitney U for pairwise comparisons, Welch's t-test for English vs non-English.

### 4.3 LLM Replication (Experiments 4–5)
- Models: DeepSeek V3.1 (671B MoE, via Together.ai), Mistral Small 24B (via Together.ai).
- Task 1: Language identification (deterministic, temperature=0.0).
- Task 2: Translation fidelity — translate to English, measure cosine similarity to reference.
- Task 3: Clinical advice — same scenario in 4 languages, 5 trials at temperature=0.7.
- Variance Ratio = Var(non-English responses) / Var(English responses).

---

## 5. Results

### 5.1 EFI Degradation is Massive and Universal (Experiments 1–2)
- EFI_Kannada = 0.099 ± 0.122, EFI_Tamil = 0.069 ± 0.088, EFI_Hindi = 0.076 ± 0.071.
- All vs English: p = 1.3 × 10⁻¹³ (Kannada), p = 5.2 × 10⁻¹⁷ (Hindi).
- Dravidian vs Indo-Aryan: p = 0.763 — degradation is not language-family-specific.
- **Finding:** ~90% EFI loss for all Indian languages tested.

### 5.2 Embedding Robustness (Experiment 3)
- MPNet (768D): EFI_Kannada = 0.125, EFI_Tamil = 0.106 — still 87–89% degraded.
- Cross-model correlation: r = 0.88 (Kannada), r = 0.72 (Tamil).
- **Finding:** Degradation is embedding-model-independent.

### 5.3 English Loanword Anchoring
- Simple sentences: EFI = 0.09. Medium: EFI = 0.03. Complex: EFI = 0.17.
- Complex sentences contain English medical terms (ST elevation, D-dimer) that anchor embeddings toward English space.
- **Finding:** Code-switching partially rescues encoding fidelity — a paradoxical complexity effect.

### 5.4 LLM Translation and Clinical Advice (Experiments 4–5)
- Language ID: models correctly identify the language (93–100%).
- Translation fidelity: Kannada 0.807–0.816, Tamil 0.655–0.895, Hindi 0.922–0.928.
- **Catastrophic failure:** Mistral translates Tamil "pregnant woman with hypertension and swelling" as "woman with bellyache crying and screaming" (sim = 0.13).
- **Finding:** Models know the language but still mistranslate — encoding degradation persists through generation.

### 5.5 Variance Amplification (Experiment 5 — Key Result)
- Variance Ratio across models:

| Language | VR (DeepSeek) | VR (Mistral) | p-value (vs English) |
|----------|---------------|--------------|----------------------|
| English  | 1.00          | 1.00         | —                    |
| Hindi    | 2.10×         | 2.23×        | 0.011                |
| Kannada  | 2.47×         | 2.07×        | 0.008                |
| Tamil    | 2.85×         | 2.67×        | 0.004                |

- **Finding:** 2–3× variance amplification for non-English inputs, consistent across model scales.

### 5.6 Summary: The Encoding-to-Variance Pipeline

| Layer | Metric | English | Non-English | Degradation |
|-------|--------|---------|-------------|-------------|
| Embedding | EFI (MiniLM) | 1.000 | 0.081 ± 0.097 | ~92% |
| Embedding | EFI (MPNet) | 1.000 | 0.119 ± 0.110 | ~88% |
| Generation | Translation fidelity | 1.000 | 0.655–0.928 | 7–35% |
| Generation | Variance Ratio | 1.00× | 2.07–2.85× | 107–185% ↑ |

---

## 6. Discussion

### 6.1 Shannon's Assumption Does Not Hold
- The 90% EFI degradation demonstrates systematic encoding infidelity for non-English inputs.
- This is not a tokenizer-specific bug — it persists across 2 embedding models and 2 LLMs of different scales.
- The information bottleneck (Tishby & Zaslavsky, 2015) predicts compression of irrelevant information; here, the *relevant* information is being compressed because the model cannot distinguish it.

### 6.2 Coherent Misalignment is a Distinct Failure Mode
- Not hallucination: the model is not inventing facts — it is misrepresenting the input.
- Not deceptive alignment: the model is not pursuing misaligned goals — it is operating on degraded encoding.
- Not sycophancy: the model is not telling the user what they want to hear — it genuinely cannot tell the difference.
- Closest analogy: a physician working from a badly translated patient history. The reasoning may be sound; the premise is wrong.

### 6.3 K ⊥ Truth: Conservation Cannot Detect This
- Papers 2–6 showed K ≈ constant across 14 models, 3 domains.
- K measures variance structure, not semantic content.
- A system with EFI = 0.08 and VR = 2.85× can still show normal K — the conservation law holds even when outputs are wrong.
- **Implication:** Self-consistency is necessary but not sufficient for safety. External truth anchoring is required.

### 6.4 The English Loanword Paradox
- Complex medical sentences (with English terms) show *higher* EFI than simple sentences.
- English medical terminology acts as an anchor, pulling the embedding toward better-represented space.
- Clinical implication: code-mixed clinical notes may paradoxically yield more reliable AI responses than pure non-English input.

### 6.5 Clinical Implications
- A doctor querying an LLM in Kannada receives advice that is grammatically correct, confidently delivered, and up to 2.85× more variable than the same query in English.
- The Mistral catastrophic failure (sim = 0.13) shows worst-case scenarios are clinically dangerous.
- WHO (2024) guidance on AI for health does not address encoding fidelity as a failure mode [15].
- 1.5 billion+ people use languages where EFI << 1.0 — this is a global health equity issue.

### 6.6 Limitations
1. EFI proxy measures embedding similarity, not true mutual information I(X; X̂).
2. Small sentence set (5 per complexity level) — larger corpus needed for generalization.
3. Three Indian languages — extension to African, Southeast Asian, and other language families required.
4. Clinical harm not directly measured — requires prospective clinical validation study.
5. Variance amplification mechanism is empirically supported but not formally proven causal.

---

## 7. Conclusion

Shannon's channel model assumes encoding fidelity. LLMs violate this assumption systematically for non-English languages, producing Coherent Misalignment — outputs that are internally consistent but semantically degraded. Using EFI, we measured ~90% encoding degradation for Kannada, Tamil, and Hindi, with 2–3× downstream variance amplification in clinical advice. The conservation law K(domain) holds normally throughout — K ⊥ Truth — meaning no self-consistency metric can detect this failure. Clinical AI deployment in linguistically diverse populations requires external truth anchoring, not just internal consistency checks. The encoding fidelity gap is a new failure mode that extends Shannon's original model.

---

## 8. Future Work
1. Measure EFI across African and Southeast Asian languages using the same protocol.
2. Develop EFI-aware tokenizers that equalize encoding fidelity across languages.
3. Prospective clinical study: deploy LLMs in PHC setting and measure real-world error rates by language.
4. Formal derivation of EFI–VR relationship from information-theoretic first principles.
5. Integrate EFI into LLM safety evaluation frameworks alongside ΔRCI, VRI, and K.

---

## Verified Reference List

### Information Theory Foundations
[1] Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423; 27(4), 623–656. DOI: 10.1002/j.1538-7305.1948.tb01338.x

[2] Tishby, N. & Zaslavsky, N. (2015). Deep Learning and the Information Bottleneck Principle. *IEEE Information Theory Workshop (ITW)*, 1–5. DOI: 10.1109/ITW.2015.7133169. arXiv: 1503.02406

[5] Shwartz-Ziv, R. & Tishby, N. (2017). Opening the Black Box of Deep Neural Networks via Information. arXiv: 1703.00810

[6] Delétang, G., Ruoss, A., Duquenne, P.-A., et al. (2024). Language Modeling Is Compression. *ICLR 2024*. arXiv: 2309.10668

### Tokenizer Inequality
[3] Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. *NeurIPS 2023*. arXiv: 2305.15425

[7] Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *ACL-IJCNLP 2021*, 3118–3135. arXiv: 2012.15613

[8] Lundin, J. M., Zhang, A., Karim, N., et al. (2025). The Token Tax: Systematic Bias in Multilingual Tokenization. arXiv: 2509.05486

[9] Ali, M., Fromm, M., Thellmann, K., et al. (2024). Tokenizer Choice For LLM Training: Negligible or Crucial? *Findings of NAACL 2024*, 3907–3924. arXiv: 2310.08754

### Cross-Lingual Representation
[10] Hammerl, K., Libovický, J., & Fraser, A. (2024). Understanding Cross-Lingual Alignment — A Survey. *Findings of ACL 2024*, 10922–10943. arXiv: 2404.06228

[11] Peng, Q. & Søgaard, A. (2024). Concept Space Alignment in Multilingual LLMs. *EMNLP 2024*, 5511–5526. DOI: 10.18653/v1/2024.emnlp-main.315

[12] Pallucchini, F., Malandri, L., Mercorio, F., & Mezzanzanica, M. (2025). Lost in Alignment: A Survey on Cross-Lingual Alignment Methods for Contextualized Representation. *ACM Computing Surveys*, 58(5). DOI: 10.1145/3764112

### Multilingual Medical AI
[4] Alonso, I., Oronoz, M., & Agerri, R. (2024). MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering. *Artificial Intelligence in Medicine*, 155, 102938. arXiv: 2404.05590

[13] Qiu, P., Wu, C., Zhang, X., et al. (2024). Towards Building Multilingual Language Model for Medicine. *Nature Communications*, 15, 8384. DOI: 10.1038/s41467-024-52417-z

[14] Asgari, E., Montana-Brown, N., Dubois, M., et al. (2025). A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation. *npj Digital Medicine*, 8(1), 274. DOI: 10.1038/s41746-025-01670-7

[15] World Health Organization. (2024). *Ethics and Governance of Artificial Intelligence for Health: Guidance on Large Multi-Modal Models.* Geneva: WHO. ISBN: 978-92-4-008475-9

### Multilingual Benchmarks
[16] Xuan, W., Yang, R., Qi, H., et al. (2025). MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation. *EMNLP 2025*. arXiv: 2503.10497

[17] Adelani, D. I., Ojo, J., Azime, I. A., et al. (2025). IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models. *NAACL 2025*, 2732–2757. Outstanding Paper Award. arXiv: 2406.03368

### Alignment and Failure Modes
[18] Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. (2019). Risks from Learned Optimization in Advanced Machine Learning Systems. arXiv: 1906.01820

[19] Ardoin, T., Cai, Y., & Wunder, G. (2025). Where Confabulation Lives: Latent Feature Discovery in LLMs. *EMNLP 2025*. ACL Anthology: 2025.emnlp-main.1515

### Dravidian Language NLP
[20] Chakravarthi, B. R., Priyadharshini, R., Muralidaran, V., et al. (2022). DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text. *Language Resources and Evaluation*. DOI: 10.1007/s10579-022-09583-7. arXiv: 2106.09460

### MCH Program Self-Citations
[21] Laxman, M. M. (2026a). Context curves behavior: Measuring AI relational dynamics with ΔRCI. *Preprints.org*, 202601.1881. (Paper 1)

[22] Laxman, M. M. (2026b). Scaling context sensitivity: Standardized benchmark across 25 LLM-domain configurations. *Preprints.org*, 202602.1114. (Paper 2)

[23] Laxman, M. M. (2026c). Domain-specific temporal dynamics of context sensitivity in large language models. *Preprints.org*, 202602.1674. (Paper 3)

[24] Laxman, M. M. (2026d). Engagement as entanglement: Variance signatures of bidirectional context coupling. *Preprints.org*, DOI: 10.20944/preprints202603.0055.v1. (Paper 4)

[25] Laxman, M. M. (2026e). Stochastic incompleteness: A predictability taxonomy for clinical AI deployment. *Preprints.org*, DOI: 10.20944/preprints202602.2034.v1. (Paper 5)

[26] Laxman, M. M. (2026f). Conservation constraint across three domains. *In preparation.* (Paper 6)

[27] Laxman, M. M. (2026g). The structure and trajectory of context sensitivity: Content-order decomposition and variance dissociation. *Preprints.org*, DOI: 10.20944/preprints202603.1116.v1. (Paper 7)

---

## Novelty Assessment

| Contribution | Status | Nearest Prior Work |
|---|---|---|
| EFI as encoding fidelity proxy | **Novel metric** | Token fertility (Petrov 2023) measures tokens, not fidelity |
| ~90% degradation for Indian languages | **Novel empirical finding** | MedExpQA (Alonso 2024) shows ~10 pp accuracy gap, not 90% encoding gap |
| Language-family independence | **Novel finding** | Prior work assumes Dravidian-specific challenges |
| English loanword anchoring | **Novel mechanism** | Code-switching studied but not as EFI anchor |
| 2–3× variance amplification by language | **Novel empirical finding** | No prior VR-by-language measurement exists |
| K ⊥ Truth | **Novel theoretical contribution** | Conservation law itself is novel (Paper 6); orthogonality to truth not previously stated |
| Coherent Misalignment (as defined) | **Novel concept** | Distinct from deceptive alignment (Hubinger 2019) and confabulation (Ardoin 2025) |
| Shannon model extension | **Novel theoretical proposal** | Information bottleneck (Tishby 2015) assumes fidelity; we show it fails |

---

## Journal Targets (Ranked)

| Priority | Journal | Rationale |
|----------|---------|-----------|
| 1 | **npj Digital Medicine** | Clinical AI safety, open access, multilingual health equity. Asgari et al. (2025) published here — same domain. |
| 2 | **Nature Communications** | Qiu et al. (2024) multilingual medical LLM published here. Broader reach. |
| 3 | **TMLR** | Rigorous ML venue, open review. Good for theory + empirics combination. |
| 4 | **Lancet Digital Health** | Clinical audience, global health. High bar but perfect fit. |
| 5 | **ACL Rolling Review → ClinicalNLP** | NLP audience, multilingual clinical focus. Workshop track if journal rejected. |

---

## Changes from DeepSeek Draft

1. **Removed** "Aiyappa et al." sycophancy citation — author name was wrong (actual: Chuck Arvin). Sycophancy section removed; not central to our argument.
2. **Added** Tishby information bottleneck [2, 5] — foundational for connecting Shannon to deep learning.
3. **Added** Rust et al. (2021) tokenizer fertility paper — landmark work we should cite.
4. **Added** Peng & Søgaard (2024), Hammerl et al. (2024), Pallucchini et al. (2025) — cross-lingual alignment survey papers.
5. **Added** Qiu et al. (2024) Nature Communications — multilingual medical LLM, strong comparison point.
6. **Added** WHO (2024) LMM guidance — more relevant than the 2025 news item DeepSeek cited.
7. **Trimmed** Section 2 from 10 subsections to 7 — removed standalone sycophancy and information-theory-for-LLMs sections (merged relevant papers into other sections).
8. **Trimmed** Future Work from verbose to 5 concrete items.
9. **Restructured** Results into 6 subsections with summary pipeline table.
10. **Added** Mistral catastrophic failure as highlighted finding in Section 5.4 (not buried in results table).
