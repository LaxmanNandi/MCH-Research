# Paper 9: Measurement Matters — Embedding Model Choice Determines Encoding Fidelity Assessment

**Title (working):** Measurement Matters: How Embedding Model Selection Shapes Encoding Fidelity Assessment in Multilingual Clinical AI

**Author:** Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

**Status:** Data complete. Manuscript pending.

---

## Core Question

Does the choice of sentence embedding model determine the measured severity of encoding fidelity degradation for non-English clinical AI?

## Key Findings

### 1. EFI is Dramatically Embedding-Dependent
Same 15 clinical sentences. Same languages. Different measurement tool:

| Embedding Model | Category | Kannada EFI | Tamil EFI | Hindi EFI |
|----------------|----------|-------------|-----------|-----------|
| all-MiniLM-L6-v2 (384D) | Baseline | 0.081 | 0.073 | 0.041 |
| all-mpnet-base-v2 (768D) | Baseline | 0.151 | 0.126 | 0.093 |
| paraphrase-multilingual-MiniLM (384D) | Multilingual | 0.268 | 0.311 | 0.751 |
| paraphrase-multilingual-mpnet (768D) | Multilingual | 0.616 | 0.677 | 0.789 |
| LaBSE (768D) | Multilingual | **0.853** | **0.857** | **0.861** |
| MuRIL (768D) | Indic MLM | ~~0.999~~ | ~~0.999~~ | ~~0.999~~ |

**10x improvement from MiniLM to LaBSE for Kannada.** Same text. Different lens.

### 2. Indic MLMs Are Degenerate — Cannot Measure EFI
MuRIL (Google) and IndicBERTv2 (AI4Bharat) produce near-identical embeddings for ALL inputs:
- Same meaning EN-KN: 0.999
- Unrelated EN-EN: 0.999
- **Cannot distinguish sentences. Not encoding semantics.**

Any claim of "Indic models fix encoding" using raw MLM embeddings is methodologically invalid.

### 3. Variance Amplification is Real but Magnitude is Measurement-Dependent
Same LLM responses. Different embedding model measuring variance:

| Embedding | DeepSeek Kannada VR | Mistral Kannada VR |
|-----------|--------------------|--------------------|
| MiniLM | 3.34× | 11.52× |
| MPNet | **27.32×** | **28.06×** |
| LaBSE | **2.36×** | **4.08×** |

Amplification is real (VR > 1.0 even with LaBSE) but ranges from 2× to 27× depending on measurement tool.

### 4. LaBSE Eliminates the Dravidian-Specific EFI Gap
With LaBSE: Kannada 0.853, Tamil 0.857, Hindi 0.861 — **all equal.** The Dravidian-specific gap Paper 8 found disappears with the right embedding model.

## Relationship to Paper 8

- Paper 8 found the encoding fidelity gap (EFI = 0.07-0.10)
- Paper 9 shows the gap is **real but embedding-dependent** (0.08 to 0.85)
- Paper 8's MiniLM measurement is valid but conservative
- Paper 9 identifies LaBSE as the recommended measurement tool
- Paper 9 validates Paper 8's methodology by exposing MuRIL's degeneracy

## Relationship to Paper 6

- Paper 6 K values computed with MiniLM (384D) and validated with MPNet (768D)
- Paper 9 shows embedding choice affects ALL cross-lingual measurements
- Conservation law K holds across embedding models (Paper 6 robustness check confirmed)
- The measurement sensitivity discovered here applies to the entire MCH programme

## Validation Experiments (Apr 2026)

### Experiment A — MuRIL Degeneracy Sanity Check
- Tested CLS, mean pooling, and mean_normalized pooling
- Cosine similarity ~0.999 for ALL inputs including random strings
- Random numpy vectors: 0.002 (correct baseline) — confirms degeneracy is model-specific not numerical
- **Verdict: DEGENERATE — confirmed, not a pipeline bug**

### Experiment B — LaBSE European Language Control
| Language | MiniLM EFI | MPNet EFI | LaBSE EFI |
|----------|-----------|-----------|-----------|
| Kannada  | 0.081     | 0.151     | 0.853     |
| Tamil    | 0.073     | 0.126     | 0.857     |
| Hindi    | 0.041     | 0.093     | 0.861     |
| French   | 0.459     | 0.560     | 0.892     |
| Spanish  | 0.423     | 0.493     | 0.910     |
| German   | 0.304     | 0.398     | 0.875     |

- Indic-European gap: MiniLM=0.32, LaBSE=0.035
- **LaBSE is nearly script-invariant. Residual gap persists but 90% reduced.**

### Experiment C — Variance Ratio Bootstrap CIs (10,000 iterations)
- Kannada VR significantly >1.0 across ALL five embedding models
- LaBSE DeepSeek Kannada: VR=2.36 [1.02, 5.42] — barely significant
- LaBSE Mistral Kannada: VR=4.08 [1.77, 9.38] — strongly significant
- **Variance amplification is LLM-intrinsic, not measurement artifact**
- **EFI and variance are INDEPENDENT phenomena**

## Key Conclusion
LaBSE fixes EFI (0.08→0.85). LaBSE does NOT fix variance (still 2-4×). Different causes. Different solutions. Encoding fidelity is the tokenizer's problem. Variance amplification is the LLM's problem.

## Data

| File | Description |
|------|-------------|
| data/paper9/paper9_efi_v2_with_degeneracy.json | EFI across 7 models with degeneracy validation |
| data/paper9/paper9_exp_a_muril_degeneracy.json | Exp A: MuRIL degeneracy full results |
| data/paper9/paper9_exp_b_european_control.json | Exp B: European control across 3 embedding models |
| data/paper9/paper9_exp_c_variance_bootstrap.json | Exp C: Bootstrap CIs for variance ratios |
| data/paper9/paper9_variance_reembedding.json | Variance ratios re-computed across 5 models |

## Scripts
- `scripts/experiments/paper9_indic_efi.py` — v1 (initial run, invalidated by degeneracy)
- `scripts/experiments/paper9_indic_efi_v2.py` — v2 (with degeneracy validation)
- `scripts/experiments/paper9_validation.py` — v3 (Experiments A, B, C)

---

**Status:** Data complete. Three validation experiments passed. Manuscript writing pending after Paper 6 capstone.
