# MCH Research Program — Project Context

## Repository
- GitHub: LaxmanNandi/MCH-Research
- Local: C:\Users\barla\mch_experiments
- Author: Dr. Laxman M M, MBBS, Primary Health Centre Manchi, Karnataka, India

## What This Is
Cross-domain experimental study measuring how domain structure shapes context
sensitivity in 14 LLMs across 112,500 responses. Six-paper research program:

- **Paper 1** (Published): Legacy foundation, 7 models, introduced ΔRCI metric
- **Paper 2** (Published, Preprints.org ID: 198770): "Scaling Context Sensitivity: A Standardized Benchmark of ΔRCI Across 25 Model-Domain Runs" — 14 models, 25 runs, 50 trials each
- **Paper 3** (Draft): Temporal dynamics — 3-bin aggregation (Early/Mid/Late)
- **Paper 4** (Draft): Entanglement mechanism — VRI (Variance Reduction Index)
- **Paper 5** (Draft): Safety taxonomy — four-class deployment framework (IDEAL/EMPTY/DANGEROUS/RICH)
- **Paper 6** (Draft): Conservation law — ΔRCI × Var_Ratio ≈ K(domain)

## Key Metrics
- **ΔRCI** = mean(RCI_TRUE) - mean(RCI_COLD) — context sensitivity measure
- **Var_Ratio** = Var_TRUE / Var_COLD — variance of per-trial RCI across 50 trials at each position
- **VRI** = 1 - Var_Ratio — Variance Reduction Index (formerly MI_Proxy, renamed Feb 2026)
- **RCI** computed via cosine similarity of response embeddings (all-MiniLM-L6-v2, 384D)
- Three conditions: TRUE (coherent 29-message history), COLD (no context), SCRAMBLED (randomized)

## Key Findings
- Philosophy (open-goal): mid-conversation peak + late decline (inverted-U in 3-bin aggregation only)
- Medical (closed-goal): diagnostic independence trough + integration rise (U-shape in 3-bin only)
- Raw 30-position curves are oscillatory — do NOT claim smooth U-shape/inverted-U at position level
- Vendor signatures significant even excluding Gemini Flash outlier (F(7,16)=3.55, p=0.017)
- ΔRCI ~ VRI correlation: r=0.76, p=1.5e-62, N=330
- Llama safety anomaly at medical P30: Var_Ratio up to 7.46
- Information hierarchy: 24/25 configs show ΔRCI_COLD > ΔRCI_SCRAMBLED
- Mann-Whitney U=51, p=0.149, rank-biserial r=0.35 for domain ΔRCI comparison (Paper 2)
- **Conservation law (Paper 6):** ΔRCI × Var_Ratio ≈ K(domain). Medical K=0.429 (CV=0.170), Philosophy K=0.301 (CV=0.166). Mann-Whitney U=46, p=0.003, Cohen's d=2.00
- Four-class safety taxonomy (Paper 5): IDEAL, EMPTY, DANGEROUS, RICH

## Data Structure
- /data/medical/closed_models/, open_models/, gemini_flash/
- /data/philosophy/closed_models/, open_models/
- /results/tables/ — pre-computed CSVs (entanglement, independence, position data)
- /papers/paper2_standardized/ — Paper 2 tex + figures
- /papers/paper3_cross_domain/ — Paper 3 results
- /papers/paper4_entanglement/ — Paper 4 results + figures
- /papers/paper5_safety/ — Paper 5 safety taxonomy
- /papers/paper6_conservation/ — Paper 6 conservation law draft + figures
- /data/paper5/ — P30 accuracy verification data
- /data/paper6/ — Conservation product CSV + MI verification

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

## Workflow
User edits manuscripts externally → shares via Downloads folder → assistant verifies
statistics against repo data, pushes to git, and checks cross-paper consistency.
