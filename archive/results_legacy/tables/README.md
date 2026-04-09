# Analysis Tables Inventory

## Current Analysis Tables (50-Trial Data)

### Trial-Level Analysis
- `trial_level_drci.csv` - Trial-by-trial ΔRCI for all models (50 trials each)

### Entanglement Analysis (Paper 4)
- `entanglement_position_data.csv` - **PRIMARY**: 330 points (11 models × 30 positions)
  - Used for r=0.76 correlation
- `entanglement_correlations.csv` - Per-model correlations (11 models)
- `entanglement_variance_summary.csv` - Variance ratios by model

### Position-Level Analysis
- `position_drci_data.csv` - Position-aggregated ΔRCI across models
- `position_analysis_summary.csv` - Statistical summary by position

### Independence Tests
- `independence_test_results.csv` - Chi-square independence (RCI vs Var)
- `independence_test_per_model.csv` - Per-model statistics
- `independence_var_ratio_results.csv` - Variance ratio independence
- `independence_var_ratio_per_model.csv` - Per-model variance

### Medical P30 Analysis
- `medical_p29_p30_spike.csv` - P29 vs P30 (Type 2 spike)

### DeepSeek & Type 2
- `deepseek_logfit_params.csv` - Log-fit parameters
- `deepseek_position_data.csv` - Position-level metrics
- `type2_scaling_validation.csv` - Type 2 scaling
- `philosophy_p10_p30_extracted.csv` - Philosophy endpoints

## Data Integrity
✓ All tables from 50-trial datasets
✓ Embedding: all-MiniLM-L6-v2 (384D)
✓ Temperature: 0.7

## Last Updated: February 11, 2026
