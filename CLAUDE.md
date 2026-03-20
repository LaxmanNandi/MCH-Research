# MCH Research Program — Project Context

## Repository
- GitHub: LaxmanNandi/MCH-Research
- Local: C:\Users\barla\mch_experiments
- Author: Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

## What This Is
Cross-domain experimental study measuring how domain structure shapes context
sensitivity in 14 LLMs across 112,500+ responses. Eight-paper research program:

- **Paper 1** (Published): Legacy foundation, 7 models, introduced ΔRCI metric
- **Paper 2** (Published, Preprints.org v1: 198770, v2: 198986): "Scaling Context Sensitivity: A Standardized Benchmark of ΔRCI Across 25 Model-Domain Runs" — 14 models, 25 runs, 50 trials each
- **Paper 3** (Published, Preprints.org ID: 199272): "Domain-Specific Temporal Dynamics of Context Sensitivity in Large Language Models" — 3-bin aggregation (Early/Mid/Late)
- **Paper 4** (Published, Preprints.org ID: 199894, DOI: 10.20944/preprints202603.0055.v1; JMLR submission prepared): "Engagement as Entanglement: Variance Signatures of Bidirectional Context Coupling in Large Language Models" — VRI (Variance Reduction Index)
- **Paper 5** (Published, Preprints.org DOI: 10.20944/preprints202602.2034.v1): "Stochastic Incompleteness: A Predictability Taxonomy for Clinical AI Deployment" — four-class framework (IDEAL/EMPTY/DIVERGENT/RICH)
- **Paper 6** (Draft): Conservation constraint — ΔRCI × Var_Ratio ≈ K(domain)
- **Paper 7** (Published, DOI: 10.20944/preprints202603.1116.v1): "The Structure and Trajectory of Context Sensitivity in LLMs: Content-Order Decomposition and Variance Dissociation" — decomposes ΔRCI into content/order components, exploration arc, CUD pilot in supplementary
- **Paper 8** (Submitted, Preprints ID: 204266): "Encoding Fidelity and Coherent Misalignment: Why Shannon's Channel Model Breaks for Non-English Clinical AI" — EFI metric, ~90% semantic loss for Indian languages, European control (d=1.33), Dravidian-specific variance amplification, K⊥Truth

## Key Metrics
- **ΔRCI** = mean(RCI_TRUE) - mean(RCI_COLD) — context sensitivity measure
- **Var_Ratio** = Var_TRUE / Var_COLD — variance of per-trial RCI across 50 trials at each position
- **VRI** = 1 - Var_Ratio — Variance Reduction Index (formerly MI_Proxy, renamed Feb 2026)
- **RCI** computed via cosine similarity of response embeddings (all-MiniLM-L6-v2, 384D)
- Three conditions: TRUE (coherent 29-message history), COLD (no context), SCRAMBLED (randomized)
- **Content Fraction** = mean(SCRAM−COLD) / mean(TRUE−COLD) × 100% — % of ΔRCI from content alone (Paper 7)
- **Exploration Arc** = Var(TRUE embeddings at P_last) / Var(TRUE embeddings at P1) — convergent (<3.0) vs divergent (>5.0) (Paper 7)
- **EFI** = cosine_similarity(non-English embedding, English equivalent) — encoding fidelity proxy (Paper 8)
- **Coherent Misalignment** = outputs that are fluent, confident, and semantically wrong due to encoding degradation (Paper 8)
- **K ⊥ Truth** = conservation law holds even when semantic content is wrong — inferred from structural definition, not directly measured in non-English conditions (Paper 8)

## Key Findings
- Philosophy (open-goal): mid-conversation peak + late decline (inverted-U in 3-bin aggregation only)
- Medical (closed-goal): diagnostic independence trough + integration rise (U-shape in 3-bin only)
- Raw 30-position curves are oscillatory — do NOT claim smooth U-shape/inverted-U at position level
- Vendor signatures significant even excluding Gemini Flash outlier (F(7,16)=3.55, p=0.017)
- ΔRCI ~ VRI correlation: r=0.76, p=8.2e-69, N=360 (12 models × 30 positions)
- Llama safety anomaly at medical P30: Var_Ratio up to 7.46
- Information hierarchy: 25/25 configs show ΔRCI_COLD > ΔRCI_SCRAMBLED (v2; was 24/25 in v1)
- Domain effect: Mann-Whitney p=0.041 (v2; was p=0.149 in v1), vendor effect p=0.014 (v2; was p=0.075 in v1)
- **Conservation constraint (Paper 6):** ΔRCI × Var_Ratio ≈ K(domain). Medical K=0.429 (CV=0.170), Philosophy K=0.301 (CV=0.166). Mann-Whitney U=46, p=0.003, Cohen's d=2.06
- Four-class predictability taxonomy (Paper 5): IDEAL, EMPTY, DIVERGENT, RICH

## Paper 7 — Content-Order Decomposition (submitted Mar 12, 2026)

### What It Does
Decomposes ΔRCI into **content** and **order** components using SCRAMBLED condition. Also introduces:
- **Content Fraction (CF)**: % of ΔRCI attributable to content alone = mean(SCRAM−COLD) / mean(TRUE−COLD) × 100% at last position
- **Exploration Arc**: Arc = Var(TRUE embeddings at P_last) / Var(TRUE embeddings at P1) — measures how response diversity changes across conversation
- **K decomposition**: Factoring conservation product K = ΔRCI × Var_Ratio into content/order components
- **CUD (Context Utilization Depth)**: Moved to Supplementary S1 — pilot-only, too noisy for main paper

### Model Set
Uses Paper 6's conservation-validated subset: **N=8 Medical + N=6 Philosophy** (14 model-domain runs total)
- Medical (8): Gemini Flash, DeepSeek V3.1, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B
- Philosophy (6): Claude Haiku, DeepSeek V3.1, Gemini Flash, GPT-4o, GPT-4o-mini, Llama 4 Maverick

### Key Results
- Content accounts for ~45-55% of ΔRCI in Medical, ~35-55% in Philosophy
- Exploration Arc: Medical 1.72±0.68 (convergent), Philosophy 15.23±16.64 (divergent) — zero domain overlap
- Arc thresholds: convergent < 3.0, divergent > 5.0
- P30 spike: all N=8 Medical models show z > +2.43 at position 30 (content + order both contribute)
- Llama safety anomaly at P30: driven entirely by order component, not content
- Sensitivity-stability dissociation: high content fraction ≠ high Var_Ratio

### Figures (9 total)
1. Content-Order Decomposition bars (N=14)
2. P30 Spike Decomposition (N=8 Medical, z-scores)
3. Variance Decomposition (VR_Content vs VR_Order)
4. K Decomposition
5. Sensitivity-Stability scatter
6. Llama P30 Anomaly
7. Exploration Arc (log scale)
8. Information Hierarchy schematic
(+ fig1_cud_k_curves.png — legacy, used in supplementary)

### Submission Files
- /papers/paper7_submission/paper7.tex, paper7.pdf
- /papers/paper7_submission/figures/ (9 PNGs)
- /papers/paper7_submission/archive/ (old generate_figures.py, verification scripts, prior manuscript versions)
- Desktop zip: Paper7_Final_Submission.zip

## Paper 6 — Legal Domain Experiment (updated Mar 15, 2026)

### Design
- Third domain to test conservation constraint generalization
- Pre-registered prediction: K(Legal) ≈ 0.41 (between Medical 0.429 and Philosophy 0.301)
- Script: /scripts/experiments/run_legal_experiments.py
- Data: /data/legal/open_models/
- 4 valid + 1 outlier + 1 excluded models × 50 trials × 30 positions (Legal P30)

### Trial Status
| Model | ΔRCI | Var_Ratio | K | Trials | Status |
|-------|------|-----------|------|--------|--------|
| DeepSeek V3.1 | 0.276 | 1.147 | 0.317 | 50/50 | COMPLETE |
| Llama 4 Maverick | 0.209 | 1.263 | 0.264 | 50/50 | COMPLETE |
| Qwen3 235B | 0.265 | 1.568 | 0.415 | 50/50 | COMPLETE |
| Mistral Small | 0.252 | 1.191 | 0.300 | 50/50 | COMPLETE |
| Kimi K2.5 | 0.509 | 1.413 | 0.718 | 50/50 | EXCLUDED — COLD refusal + 21% empty responses |
| GLM-5 | — | — | — | 13/50 | EXCLUDED — 86% empty responses, non-functional |
| Llama 3.3 70B | 0.206 | — | — | 40/50 | IN PROGRESS — dRCI stable, needs re-embedding for VR |
| Ministral 14B | — | — | — | 0/50 | UNAVAILABLE (Together AI serverless) |
| Llama 4 Scout | — | — | — | 0/50 | UNAVAILABLE (Together AI serverless) |
| Kimi K2 | — | — | — | 0/50 | UNAVAILABLE (Together AI serverless) |

### Legal Domain Results (N=4 valid)
- **K(Legal) = 0.324 (range 0.264–0.415)** — between Philosophy (0.301) and Medical (0.429)
- **Information hierarchy**: TRUE > SCRAMBLED > COLD confirmed in all 4 models
- **No P30 spike**: max z=0.66 (vs Medical all z>2.43)
- **No entanglement**: ΔRCI~VRI r=-0.033, p=0.722 (vs Medical+Phil r=0.76, p=2.37e-68)
- **Mixed temporal dynamics**: 2 U-shape, 1 declining, 1 rising (no consensus pattern)
- **Discovered vs constructed truth**: Entanglement depends on whether domain truth is discovered (medical) or constructed (legal). Conservation holds in both.
- **Kimi K2.5 outlier**: K=0.718, dRCI=0.509 inflated by systematic COLD refusal ("I cannot provide legal advice") + 21% empty responses across positions. Parallels Llama safety anomaly in medical domain.
- **GLM-5 excluded**: 86% empty responses in both TRUE and COLD conditions — model non-functional for legal domain.

### Together.ai Cost
- DeepSeek V3.1: $36.36, Maverick: $2.78+, Mistral Small: ~$0.50, Qwen3: ~$5, Kimi K2.5: ~$3, GLM-5: ~$1
- Total so far: ~$48

## Data Structure
- /data/medical/closed_models/, open_models/, gemini_flash/
- /data/philosophy/closed_models/, open_models/
- /results/tables/ — pre-computed CSVs (entanglement, independence, position data)
- /papers/paper2_standardized/ — Paper 2 tex + figures
- /papers/paper3_cross_domain/ — Paper 3 results
- /papers/paper4_entanglement/ — Paper 4 results + figures
- /papers/paper5_safety/ — Paper 5 safety taxonomy
- /papers/paper6_conservation/ — Paper 6 conservation constraint draft + figures
- /data/paper5/ — P30 accuracy verification data
- /data/paper6/ — Conservation product CSV + MI verification
- /data/legal/open_models/ — Legal domain experiment results (Paper 6 extension)
- /scripts/experiments/paper7_pilot/ — Paper 7 CUD pilot script + results/raw/
- /papers/paper7_submission/ — Paper 7 final submission (tex, pdf, figures/, archive/)
- /papers/paper8_efi/ — Paper 8 EFI (tex, scripts/, results/, figures/, drafts/)
- /papers/paper7_submission/archive/ — Paper 7 concept docs, CUD pilot analysis, prior manuscript versions

## Important Rules
1. NEVER delete data files, experimental results, or git history
2. Always verify statistical claims against raw data before accepting them
3. U-shape/inverted-U language requires "3-bin aggregation" qualifier
4. Use "VRI" not "MI_Proxy" — renamed across entire repo Feb 2026
5. Use "shapes" not "causes" — observational language, no causal claims
6. RCI_COLD = responses with no conversational history, not cross-condition similarity
7. Paper 2 says 14 models, 112,500 responses — keep all docs consistent with this
8. When pushing to git, always verify changes don't break Paper 3, 4, 5, or 6 claims
9. Preprint submission folder: C:\Users\barla\Desktop\Paper2_Preprint_Submission\
10. RCI = "Relational Coherence Index" (NOT "Response Coherence Index")
11. Paper 7 uses conservation-validated subset (N=8 Med + N=6 Phil from Paper 6) — do NOT mix with Paper 2's N=14 or Paper 3-4's N=12

## Workflow
User edits manuscripts externally → shares via Downloads folder → assistant verifies
statistics against repo data, pushes to git, and checks cross-paper consistency.
