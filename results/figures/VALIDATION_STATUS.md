# Figure Data Validation Status

## ✓ ALL FIGURES USE 50-TRIAL DATA

### Validated Figures

| Figure | Trial Count | Models | Date | Status |
|--------|-------------|--------|------|--------|
| entanglement_validation.png | 50 | 11 | Feb 11 2026 | ✓ CURRENT (Featured) |
| fig1_position_drci_publication.png | 50 | 22 | Feb 8 2026 | ✓ CURRENT |
| fig3_zscores_pub.png | 50 | 22 | Feb 8 2026 | ✓ CURRENT |
| fig4_entanglement_multipanel_final_v2.png | 50 | 11 | Feb 9 2026 | ✓ CURRENT |
| fig5_independence_rci_var.png | 50 | 22 | Feb 9 2026 | ✓ CURRENT |
| fig6_type2_scaling_final.png | 50 | 22 | Feb 8 2026 | ✓ CURRENT |
| fig7_llama_safety_anomaly_final.png | 50 | 2 | Feb 9 2026 | ✓ CURRENT |
| trial_level_drci_convergence.png | 50 | 14 | Feb 9 2026 | ✓ CURRENT |
| paper4/figure6_gaussian_verification.png | 50 | 2 | Feb 9 2026 | ✓ CURRENT |
| paper4/figure8_trial_convergence.png | 50 | 14 | Feb 10 2026 | ✓ CURRENT |
| paper4/figure9_model_comparison.png | 50 | 14 | Feb 10 2026 | ✓ CURRENT |

### Data Sources
- **Medical**: 7 models × 50 trials × 30 positions = 10,500 responses per model
- **Philosophy**: 11 models × 50 trials × 30 positions = 16,500 responses per model
- **Total responses**: ~99,000+

### Embedding Model
All figures use consistent embedding:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Temperature**: 0.7 (all runs)

### Key Updates
- **Feb 11, 2026**: Entanglement analysis updated from 8 to 11 models
  - Previous: r=0.74, p=3.0e-42, N=240 (8 models)
  - Current: r=0.76, p=1.5e-62, N=330 (11 models)
  
## Legacy Figures (Archived)
All pre-50-trial figures moved to `.dev/archive/old_figures/`

## Verification
To verify any figure's data source, check the corresponding script in `scripts/analysis/`:
- `validate_entanglement.py` - Entanglement figures
- `generate_paper3_figures.py` - Temporal dynamics figures
- `compute_trial_drci.py` - Trial-level analysis

---
**Last Validated**: February 11, 2026
**Validator**: Automated consistency check
