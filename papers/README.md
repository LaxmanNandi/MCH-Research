# Papers Directory: Research Lineage

This directory organizes research outputs by **paper lineage** rather than file type.

## Structure

```
papers/
├── paper1_legacy/           Paper 1 ✅ Published (Preprints.org)
├── paper2_standardized/     Paper 2 ✅ Published v2 (Preprints.org)
├── paper3_cross_domain/     Paper 3 ✅ Published (Preprints.org)
├── paper4_entanglement/     Paper 4 ✅ Published (Preprints.org) + JMLR under review
├── paper5_safety/           Paper 5 ✅ Published (Preprints.org)
├── paper6_conservation/     Paper 6 📄 Submitted (Preprints.org ID: 207501)
├── paper7_submission/       Paper 7 ✅ Published (Preprints.org)
├── paper8_efi/              Paper 8 ✅ Published (Preprints.org) + Scientific Reports (peer review)
└── paper9_measurement/      Paper 9 ✅ Published (Zenodo)
```

## Paper Lineage

```
Paper 1 (Legacy) → Paper 2 (Standardized) → Papers 3 & 4 (Extensions) → Paper 5 (Safety) → Paper 7 (Decomposition) → Paper 6 (Conservation Law — Capstone)
Paper 8 (Encoding Fidelity) → Paper 9 (Measurement Matters)
```

### Paper 1: Context Curves Behavior
- **Status**: Published (Preprints.org, February 2, 2026, corrected version)
- **DOI**: 10.20944/preprints202601.1881.v2
- **Role**: Foundation — Introduced ΔRCI metric and epistemological relativity concept
- **Models**: 7 closed models (GPT-4o/mini/5.2, Claude Opus/Haiku, Gemini Flash/Pro)
- **Domains**: 2 (Philosophy: 700 trials + Medical: 300 trials = 1,000 total)
- **Location**: `paper1_legacy/`

### Paper 2: Scaling Context Sensitivity
- **Status**: Published (Preprints.org, February 12, 2026; v2 correction)
- **DOI**: 10.20944/preprints202602.1114.v2
- **Role**: Core study — Unified methodology, cross-domain validation
- **Models**: 14 unique models, 25 model-domain runs
- **Location**: `paper2_standardized/`

### Paper 3: Domain-Specific Temporal Dynamics
- **Status**: Published (Preprints.org, February 16, 2026)
- **DOI**: 10.20944/preprints202602.1674.v1
- **Role**: Extension — Position-level analysis, U-shape vs inverted-U in 3-bin aggregation
- **Location**: `paper3_cross_domain/`

### Paper 4: Engagement as Entanglement
- **Status**: Published (Preprints.org, February 22, 2026) + JMLR under review
- **DOI**: 10.20944/preprints202603.0055.v1
- **Role**: Mechanism — ΔRCI~VRI correlation r=0.76, ESI metric
- **Location**: `paper4_entanglement/`

### Paper 5: Stochastic Incompleteness
- **Status**: Published (Preprints.org, February 28, 2026)
- **DOI**: 10.20944/preprints202602.2034.v1
- **Role**: Application — Four-class deployment taxonomy (IDEAL/EMPTY/DIVERGENT/RICH)
- **Location**: `paper5_safety/`

### Paper 6: Conservation Constraint (Capstone)
- **Status**: Resubmitted (Preprints.org, April 13, 2026 — ID: 208191, Pending Decision)
- **Title**: A Conservation Constraint in LLM Context Processing Across Four Epistemological Domains
- **Role**: Capstone — ΔRCI × Var_Ratio ≈ K(domain) across 4 domains, 24 runs, 14 architectures
- **K values**: Medical 0.429, Legal 0.348, Philosophy 0.301, Ethics 0.223
- **Location**: `paper6_conservation/` (v1_submission/ has submitted tex)

### Paper 7: Content-Order Decomposition
- **Status**: Published (Preprints.org, March 16, 2026)
- **DOI**: 10.20944/preprints202603.1116.v1
- **Role**: Structural decomposition — Content fraction, exploration arc, CUD pilot
- **Location**: `paper7_submission/`

### Paper 8: Encoding Fidelity & Coherent Misalignment
- **Status**: Published (Preprints.org, April 2, 2026) + Scientific Reports (peer review)
- **DOI**: 10.20944/preprints202604.0061.v1
- **Role**: Cross-lingual — EFI metric, Coherent Misalignment, Dravidian-specific variance
- **Location**: `paper8_efi/`

### Paper 9: Measurement Matters
- **Status**: Published (Zenodo, April 7, 2026)
- **DOI**: 10.5281/zenodo.19466613
- **Role**: Measurement validation — EFI is embedding-dependent, MuRIL degeneracy, LaBSE validated
- **Location**: `paper9_measurement/`

## Data Location

**All experimental data is stored in `/data/` directory** (single source of truth).

```
/data/
├── medical/
│   ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 ×2, Mistral ×2, Qwen)
│   └── closed_models/    6 models (GPT ×3, Claude ×2, Gemini)
├── philosophy/
│   ├── open_models/      7 models (DeepSeek, Kimi, Llama 4 ×2, Mistral ×2, Qwen)
│   └── closed_models/    5 models (GPT ×3, Claude, Gemini)
├── legal/open_models/    5 valid + 2 excluded (Kimi K2.5, GLM-5)
├── ethics/open_models/   5 models (DeepSeek, Llama 70B, Maverick, Mistral, Qwen)
├── paper5/               Accuracy verification and Llama deep-dive data
├── paper6/               Conservation law verification, manuscript data JSON, robustness
├── paper7/               Cold prior voice analysis
├── paper9/               Validation experiments (A through G)
└── eeg_pilot/            EEG pilot (Sleep-EDF, 20 subjects)
```

## Quick Navigation

| Document | Location |
|----------|----------|
| Paper 6 submitted tex | `paper6_conservation/v1_submission/paper6_final.tex` |
| Paper 6 supplementary | `paper6_conservation/paper6_supplementary.tex` |
| Paper 6 manuscript data | `/data/paper6/paper6_manuscript_data.json` |
| Paper 8 manuscript | `paper8_efi/paper8.tex` |
| Paper 9 manuscript | `paper9_measurement/paper9.tex` |
| Conservation data | `/data/paper6/conservation_product_test.csv` |
| Submission status | `SUBMISSION_STATUS.md` |

---

**Last Updated**: April 15, 2026
**Publications**: 8 published, 1 resubmitted (Paper 6, ID: 208191), Paper 4 under JMLR review, Paper 8 under Scientific Reports peer review
