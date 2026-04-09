# COLD Prior Voice Analysis

**Status**: Exploratory analysis complete — Paper 6 supplementary candidate
**Date**: March 6, 2026

## Overview
Analysis of COLD condition (no-context) responses across 14 model-domain runs to reveal intrinsic model "prior voice" signatures. Proposed by DeepSeek V3.1 during collaborative analysis.

## Dataset
- 14 model-domain runs (8 Medical, 6 Philosophy)
- 50 trials × 30 positions × COLD condition = 1,500 responses per run
- Embedded with all-MiniLM-L6-v2 (384D)

## Key Findings

### 1. Var_COLD Ranking
- Highest: Gemini Flash Philosophy (0.000628), Kimi K2 Medical (0.000623)
- Lowest: Qwen3 235B Medical (0.000302), Llama Scout Medical (0.000306)
- Gemini Flash consistently noisiest; Llama/Qwen consistently tightest

### 2. Domain Effect on Variance
- Medical mean: 0.000434, Philosophy mean: 0.000466
- Mann-Whitney U=19.0, **p=0.573 (NOT significant)**
- Prior voice variability is model-intrinsic, not domain-shaped

### 3. Prior Voice Centroids
- **Cross-vendor Medical convergence**: cos > 0.97 (Llama, Mistral, Qwen say similar things)
- **Cross-domain same model**: cos ~0.18 (nearly orthogonal — domain reshapes the voice entirely)
- **GPT-4o ≈ GPT-4o Mini**: cos=0.992 (nearly identical default voice)

### 4. Compressed Spring Effect
- Llama models: lowest Var_COLD (tightest prior) → highest Var_Ratio with context (1.61)
- Kimi K2: highest Var_COLD → lowest Var_Ratio (1.006)
- Negative correlation (ρ=-0.381) but not significant at N=8 (p=0.352)

### 5. Model Size Correlation
- Spearman ρ=0.504, p=0.166 — trending but not significant

### 6. Family Clustering
| Family | Mean Var_COLD | Character |
|--------|--------------|-----------|
| Gemini | 0.000611 | Noisiest |
| Kimi | 0.000623 | Noisiest |
| DeepSeek | 0.000471 | Mid |
| Claude | 0.000511 | Mid |
| Mistral | 0.000439 | Mid |
| GPT | 0.000361 | Tight |
| Llama | 0.000356 | Tight |
| Qwen | 0.000302 | Tightest |

## Interpretation for Paper 6
- Domain sets centroid direction (WHAT models say) — explains K(Medical) > K(Philosophy)
- Architecture sets variance level (HOW MUCH models vary) — explains position on hyperbola
- Conservation law K(domain) operates above the prior level
- COLD centroid convergence grounds domain-dependent context processing at the baseline

## Files
- `cold_prior_voice_results.json` — full results with per-position variance

## Figures
Located in `docs/figures/paper7/`:
- `var_cold_ranking.png` — bar chart ranking all models
- `var_cold_domain_boxplot.png` — Medical vs Philosophy comparison
- `var_cold_by_vendor.png` — vendor comparison boxplot
- `cold_tsne_clustering.png` — t-SNE by domain and vendor
- `cold_tsne_by_model.png` — t-SNE by individual model
- `prior_voice_heatmap.png` — centroid cosine similarity matrix

## Script
`scripts/analysis/cold_prior_voice_analysis.py`
