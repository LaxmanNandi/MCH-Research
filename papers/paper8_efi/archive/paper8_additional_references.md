# Paper 8 — Additional References for Resubmission
## 16 new external references to address Preprints.org self-citation feedback

---

## 1. Tokenizer Inequality & Fragmentation

### [D] Liang et al. (2025)
- **Citation:** Liang, Y., et al. (2025). Tokenization disparities as infrastructure bias: How subword systems create inequities in LLM access and efficiency. arXiv: 2510.12389.
- **Why:** Quantifies extreme tokenization burden. Names Kannada specifically — lowest efficiency (0.55 chars/token). Frames tokenization as infrastructure bias.
- **Section:** 2.2 Tokeniser Inequality, Discussion

### [E] Arunachalam et al. (2025)
- **Citation:** Arunachalam, S., et al. (2025). Multilingual tokenization through the lens of Indian languages: Challenges and insights. arXiv: 2506.17789.
- **Why:** Directly studies tokenization for Kannada and Tamil. English-centric tokenizer configurations are a fundamental design flaw for Indic languages.
- **Section:** 2.2 Tokeniser Inequality, 2.6 Dravidian Language NLP

---

## 2. Cross-Lingual Representation

### [B] Conneau et al. (2020)
- **Citation:** Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale. Proceedings of ACL 2020, 8440-8451. arXiv: 1911.02116.
- **Why:** Introduces XLM-R. "Curse of multilinguality." Foundational, 2000+ citations.
- **Section:** 2.3 Cross-Lingual Representation

### [C] Pires, Schlinger & Garrette (2019)
- **Citation:** Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is multilingual BERT? Proceedings of ACL 2019, 4996-5001. arXiv: 1906.01502.
- **Why:** mBERT cross-lingual probing. Degradation for typologically distant languages.
- **Section:** 2.3 Cross-Lingual Representation

### [F] Joshi et al. (2020)
- **Citation:** Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. Proceedings of ACL 2020, 6282-6293. arXiv: 2004.09095.
- **Why:** 6-class taxonomy for language resources. Kannada is low-resource. Macro framing for WHY EFI degrades.
- **Section:** Introduction, 2.3 Cross-Lingual Representation

### [G] Ahuja et al. (2023)
- **Citation:** Ahuja, K., Diddee, H., Hada, R., et al. (2023). MEGA: Multilingual evaluation of generative AI. Proceedings of EMNLP 2023. arXiv: 2303.12528.
- **Why:** First comprehensive multilingual LLM benchmark across 70 languages. Systematic degradation for low-resource languages.
- **Section:** 2.5 Multilingual Benchmarks

### [H] Chang et al. (2024)
- **Citation:** Chang, T.A., et al. (2024). When is multilinguality a curse? Language modeling for 250 high- and low-resource languages. Proceedings of EMNLP 2024. arXiv: 2311.09205.
- **Why:** Curse of multilinguality quantified. Explains WHY encoding fidelity degrades.
- **Section:** 2.3 Cross-Lingual Representation, Discussion

---

## 3. Multilingual Clinical/Medical AI

### [I] Jin et al. (2024) — KEY ADDITION
- **Citation:** Jin, Y., Chandra, M., Verma, G., Hu, Y., De Choudhury, M., & Kumar, S. (2024). Better to ask in English: Cross-lingual evaluation of large language models for healthcare queries. Proceedings of the ACM Web Conference 2024. arXiv: 2310.13132.
- **Why:** TITLE SAYS IT ALL. Independent confirmation of Paper 8's thesis. LLMs give higher quality healthcare in English than non-English.
- **Section:** 2.4 Multilingual Medical AI, Introduction

### [J] Singhal et al. (2023)
- **Citation:** Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. Nature, 620(7973), 172-180. DOI: 10.1038/s41586-023-06291-2.
- **Why:** Med-PaLM in Nature. Clinical knowledge exists but benchmarked only in English. Knowledge there — non-English encoding can't access it.
- **Section:** 1.1 Clinical Reality, 2.4 Multilingual Medical AI

### [K] Dangi et al. (2025)
- **Citation:** Dangi, A., et al. (2025). Transforming healthcare in low-resource settings with artificial intelligence. Public Health Nursing. DOI: 10.1111/phn.13500.
- **Why:** AI in low-resource settings (like PHCs). Your clinical deployment context.
- **Section:** 1.1 Clinical Reality, Discussion

---

## 4. Dravidian Language NLP

### [L] Kakwani et al. (2020)
- **Citation:** Kakwani, D., Kunchukuttan, A., Golla, S., et al. (2020). IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages. Findings of EMNLP 2020, 4948-4961.
- **Why:** Main NLP resource for Kannada, Tamil, Hindi. Despite these efforts, encoding fidelity still degraded.
- **Section:** 2.6 Dravidian Language NLP

### [M] Khanuja et al. (2020)
- **Citation:** Khanuja, S., Dandapat, S., Srinivasan, A., Sitaram, S., & Choudhury, M. (2020). GLUECoS: An evaluation benchmark for code-switched NLP. Proceedings of ACL 2020. arXiv: 2004.12376.
- **Why:** Code-switching benchmark. Explains your 84% English response finding.
- **Section:** 2.6 Dravidian Language NLP, Discussion

---

## 5. AI Safety, Alignment & Failure Modes

### [N] Huang et al. (2024)
- **Citation:** Huang, L., Yu, W., Ma, W., et al. (2024). A survey on hallucination in large language models. ACM Transactions on Information Systems, 43(2). arXiv: 2311.05232.
- **Why:** Comprehensive hallucination taxonomy. Coherent Misalignment is DISTINCT from both types. Positions your novel failure mode.
- **Section:** 3.3 Coherent Misalignment, 2.5 Alignment and Failure Modes

### [O] Sharma et al. (2024)
- **Citation:** Sharma, M., Tong, M., Korbak, T., et al. (2024). Towards understanding sycophancy in language models. Proceedings of ICLR 2024. arXiv: 2310.13548.
- **Why:** Formal sycophancy reference. Contrasts with Coherent Misalignment.
- **Section:** 3.3 Coherent Misalignment

### [P] Shanahan (2024)
- **Citation:** Shanahan, M. (2024). Talking about large language models. Communications of the ACM, 67(2), 68-79. arXiv: 2212.03551.
- **Why:** "Model doesn't know less, it receives less." Philosophical grounding for encoding degradation.
- **Section:** 1.1 Clinical Reality, Discussion

---

## 6. Variance and Consistency

### [Q] Atil et al. (2024)
- **Citation:** Atil, B., Aykent, S., Chittams, A., et al. (2024). Non-determinism of "deterministic" LLM settings. arXiv: 2408.04667.
- **Why:** 15% accuracy variation across runs. Encoding degradation amplifies fundamental LLM variance.
- **Section:** 3.5 Variance Amplification, Experiment 5

---

## Impact on Paper

| | Before | After |
|---|---|---|
| Total references | 28 | 44 |
| Self-citations | 7 (25%) | 7 (16%) |
| External references | 21 | 37 |
| Status | Declined | Ready to resubmit |
