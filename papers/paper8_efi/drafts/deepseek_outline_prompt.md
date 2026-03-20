# Paper 8 Outline Request — For DeepSeek

You are a research assistant helping draft an academic paper outline. The paper is part of a seven-paper research program studying LLM behavior. Papers 1-7 are published or drafted. This is Paper 8.

## Author
Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

## Paper Title (working)
"Encoding Fidelity and Coherent Misalignment: Why Shannon's Channel Model Breaks for Non-English Clinical AI"

## Core Thesis
Shannon's Mathematical Theory of Communication (1948) assumes encoding fidelity — that the encoder preserves the statistical structure of the source. Large Language Models violate this assumption systematically for non-English languages, producing outputs that are internally consistent but semantically degraded. We call this failure mode "Coherent Misalignment" and introduce the Encoding Fidelity Index (EFI) to measure it. We show that the conservation law from our prior work (ΔRCI × Var_Ratio ≈ K) holds even when encoding fidelity collapses — meaning the system's self-consistency metrics cannot detect its own degradation. We call this property K ⊥ Truth.

## Empirical Results (all from our experiments — cite as "this study")

### Experiment 1: Embedding-Level EFI (proxy)
- Model: all-MiniLM-L6-v2 (384D)
- Design: 3 complexity levels × 5 clinical sentences × 3 languages (English, Kannada, Tamil)
- EFI measured as cosine similarity between non-English embedding and English equivalent
- Results:
  - EFI_Kannada = 0.099 ± 0.122
  - EFI_Tamil = 0.069 ± 0.088
  - EFI_English = 1.000 (reference)
  - ~90% degradation for both Dravidian languages (p = 1.3 × 10⁻¹³)
  - Kannada vs Tamil: NOT significantly different (U=128, p=0.534) — equally degraded
  - Complexity effect: Simple EFI=0.09, Medium=0.03, Complex=0.17
  - Complex sentences have HIGHER EFI due to English loanword anchoring (ST elevation, D-dimer, etc.)
- Confusion ratio: Kannada-Tamil similarity 3.5x higher than Kannada-English

### Experiment 2: Hindi Comparison (Language Family Test)
- Added Hindi (Indo-Aryan) to test whether degradation is Dravidian-specific
- Results:
  - EFI_Hindi = 0.076 ± 0.071
  - Dravidian (Kn+Ta) vs Indo-Aryan (Hi): U=212, p=0.763 (NOT significant)
  - Hindi shows the SAME ~92% degradation as Dravidian languages
  - Conclusion: Degradation is universal for Indian languages, not Dravidian-specific
  - Hindi vs English: p = 5.2 × 10⁻¹⁷

### Experiment 3: Embedding Robustness (MPNet 768D)
- Same battery run with all-mpnet-base-v2 (768D) alongside all-MiniLM-L6-v2 (384D)
- Results:
  - MPNet EFI_Kannada = 0.125, EFI_Tamil = 0.106 (vs MiniLM 0.099, 0.069)
  - Degradation: ~87-89% under MPNet (vs ~90% under MiniLM)
  - Cross-model correlation: r=0.88 (Kannada, p=1.9×10⁻⁵), r=0.72 (Tamil, p=2.7×10⁻³)
  - Conclusion: Finding is embedding-model-independent

### Experiment 4: LLM Language Identification + Translation Fidelity
- Models: DeepSeek V3.1 (large, 671B MoE), Mistral Small 24B (small)
- Task 1 — Language ID: DeepSeek 93%, Mistral 100%. Models CAN identify the language.
- Task 2 — Translation fidelity (cosine sim of translation to reference English):
  - DeepSeek: Hindi=0.928, Tamil=0.895, Kannada=0.807
  - Mistral: Hindi=0.922, Kannada=0.816, Tamil=0.655
  - Mistral catastrophic failure: Tamil "Pregnant woman with high blood pressure and swelling" → "A woman with a bellyache is crying loudly and screaming" (sim=0.13)
- Task 3 — Clinical advice consistency: Response embeddings diverge by input language for same clinical scenario

### Experiment 5: Multi-Trial Clinical Variance (KEY RESULT)
- 5 trials × 5 scenarios × 4 languages × 2 models, temperature=0.7
- Variance Ratio (response variance in language X / response variance in English):
  - DeepSeek: Tamil VR=2.85x, Kannada VR=2.47x, Hindi VR=2.10x
  - Mistral: Tamil VR=2.67x, Kannada VR=2.07x, Hindi VR=2.23x
  - Tamil variance significantly > English: DeepSeek p=0.004, Mistral p=0.029
- Response fidelity (centroid similarity to English centroid):
  - DeepSeek: Hindi=0.890, Kannada=0.811, Tamil=0.764
  - Mistral: Hindi=0.910, Kannada=0.861, Tamil=0.837
- Conclusion: Language encoding degrades response CONSISTENCY, not just accuracy. 2-3x variance amplification through encoding bottleneck.

### Prior Results From Papers 1-7 (cite appropriately)
- Conservation law: ΔRCI × Var_Ratio ≈ K(domain). Medical K=0.429, Philosophy K=0.301, Legal K=0.362. (Paper 6)
- K holds across 14 model architectures from 8 vendors (Paper 6)
- Entanglement (ΔRCI~VRI correlation) depends on discovered vs constructed truth (Paper 6 legal extension)
- Var_Ratio measures output variance amplification from context (Papers 2-6)
- The conservation law's internal consistency cannot detect encoding degradation — K ⊥ Truth

## Theoretical Framework

### 1. Shannon's Assumption
Shannon (1948) defined channel capacity assuming the encoder faithfully maps source symbols to channel symbols. Every application of information theory to NLP inherits this assumption. But LLM tokenizers, embedding layers, and attention mechanisms do NOT preserve encoding fidelity equally across languages.

### 2. Encoding Fidelity Index (EFI)
- Theoretical: EFI = I(X; X̂) / H(X), where X is the source message and X̂ is the model's internal representation
- Practical proxy: EFI_proxy = cos_sim(embedding(non-English), embedding(English_equivalent))
- EFI = 1.0 for English (by construction — models trained primarily on English)
- EFI << 1.0 for low-resource languages

### 3. Coherent Misalignment
When EFI < 1.0, the model produces outputs that are:
- Grammatically correct
- Internally consistent
- Semantically wrong or degraded
This is distinct from Hubinger et al. (2019) "deceptive alignment" — not about goals, but about encoding. The system isn't being deceptive; it's operating on degraded input representation.

### 4. K ⊥ Truth (Conservation Orthogonal to Truth)
The conservation law ΔRCI × Var_Ratio ≈ K holds regardless of encoding fidelity. The system's variance structure is self-consistent even when the content is wrong. This means:
- You cannot detect Coherent Misalignment from within the system
- External ground truth comparison is required
- All self-consistency-based safety metrics are blind to this failure mode

### 5. Variance Amplification Mechanism
Low EFI → noisier internal representation → higher output variance for same input → clinical advice becomes less reliable. The 2-3x Var_Ratio we measured is the downstream consequence of the 90% EFI loss measured at the embedding level.

## What I Need From You

Please provide:

1. **A detailed paper outline** with section headers, subsection structure, and 1-2 sentence descriptions of what goes in each section

2. **A comprehensive reference list** with real, verifiable citations for:
   - Shannon (1948) original paper
   - Tokenizer bias / multilingual NLP inequality papers
   - Embedding space analysis for low-resource languages
   - Clinical AI safety in multilingual settings
   - LLM evaluation across languages (benchmarks showing performance gaps)
   - Coherent/deceptive alignment literature (Hubinger et al. 2019, etc.)
   - Confabulation/hallucination literature
   - Information theory applied to LLMs (Delétang et al. 2024 "Language Modeling Is Compression")
   - Sycophancy and output bias literature
   - WHO/health equity and digital divide literature
   - Any paper measuring encoding fidelity or equivalent concept
   - Dravidian language NLP papers
   - Multilingual embedding evaluation papers

3. **Identification of which claims are novel** vs which have prior literature support

4. **Suggested journal targets** for this paper (AI safety, clinical informatics, NLP, information theory)

## Important Constraints
- Use observational language ("shapes" not "causes")
- Acknowledge proxy EFI limitation honestly (we measure embedding similarity, not true mutual information)
- Do NOT overclaim — we show the mechanism and measure the degradation, we don't prove patient harm (that requires clinical study)
- Keep the clinical framing central — this matters because real doctors use LLMs in these languages
- The author is a practicing PHC doctor — the clinical perspective is first-person, not theoretical
