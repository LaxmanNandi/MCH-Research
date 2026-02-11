# Paper 4 Entanglement Analysis Investigation

## Problem Statement
User requested expanding Paper 4 entanglement analysis from 11 models to 23 models using all available 50-trial metrics data.

## Investigation Findings

### Root Cause Identified
The current `entanglement_position_data.csv` (modified Feb 11 22:00) was **OVERWRITTEN** with invalid data by the broken `validate_entanglement_23models.py` script.

### Invalid Calculation Method
The broken script attempted to compute entanglement without response text by using:
- **Invalid proxy**: Variance of DRCI values across trials
- **Result**: r = 0.08, all negative per-model correlations
- **Status**: WRONG - this is not a valid measure of embedding variance

### Valid Historical Calculation (Commit 68b8adf)
The original entanglement analysis properly used:
- **Method**: Actual embedding variances computed from response text
- **Formula**:
  - Var_TRUE = variance of TRUE response embeddings across trials
  - Var_COLD = variance of COLD response embeddings across trials
  - MI_Proxy = 1 - (Var_TRUE / Var_COLD)
- **Result**: r = 0.7568 ≈ 0.76, p = 1.54e-62
- **Per-model**: ALL positive correlations (+0.72 to +0.94)
- **Status**: CORRECT

## Why r=0.76 is Correct

### Historical Data Evidence (Commit 68b8adf)
```
Pooled correlation: r = 0.7568, p = 1.54e-62
N = 330 data points (11 models × 30 positions)

Per-model correlations (all positive):
  Claude Haiku (Phil):      r = +0.915 ***
  Ministral 14B (Med):      r = +0.936 ***
  Mistral Small 24B (Med):  r = +0.896 ***
  DeepSeek V3.1 (Med):      r = +0.882 ***
  Llama 4 Maverick (Med):   r = +0.846 ***
  Qwen3 235B (Med):         r = +0.839 ***
  ... (all 11 models positive)
```

### Why Current Data Showed r=0.08 (Wrong)
The broken script computed:
- Variance of DRCI values instead of embedding variances
- Produced MI_Proxy ≈ 0.99 (should be negative on average for divergent cases)
- Resulted in near-zero pooled correlation due to incorrect variance proxy

## Blocking Constraint: Response Text Required

### Proper Entanglement Calculation Requires:
1. Encode each trial's response text into embeddings (384D vectors)
2. Compute variance of TRUE embeddings across trials (per position, per dimension)
3. Compute variance of COLD embeddings across trials
4. Calculate Var_Ratio = Var_TRUE / Var_COLD
5. Derive MI_Proxy = 1 - Var_Ratio

### Response Text Availability Status:
- **10 models WITH text**: 4 Phil Closed + 6 Med Open
- **12 models WITHOUT text**: 7 Phil Open + 5 Med Closed
- **Historical 11 models**: Had response text available (including Gemini Flash Med)

### Conclusion:
**Cannot expand to 22 models** without rerunning experiments with response text preservation for the 12 models that lack it.

## Actions Taken

1. ✅ Restored `entanglement_position_data.csv` from commit 68b8adf
2. ✅ Restored `entanglement_correlations.csv` from commit 68b8adf
3. ✅ Restored `entanglement_variance_summary.csv` from commit 68b8adf
4. ✅ Deleted broken `validate_entanglement_23models.py`
5. ✅ Deleted broken `regenerate_paper4_22models.py`
6. ✅ Updated `Paper4_Results.md` with methodology clarification
7. ✅ Committed: "Restore valid 11-model entanglement analysis (r=0.76)"
8. ✅ Pushed to master

## Final Status

**Paper 4 Entanglement Analysis:**
- Models: 11 (4 Phil + 7 Med)
- Data points: 330 (11 × 30 positions)
- Correlation: r = 0.76, p = 1.5e-62 ✓
- Method: Valid embedding variance calculation
- Status: RESTORED AND CONFIRMED

## Path Forward

To expand entanglement analysis beyond 11 models:
1. Rerun experiments for 12 models without response text
2. Enable response text preservation in experiment scripts
3. Recompute entanglement metrics using proper embedding variance method
4. Expected cost: ~$50-100 for API calls (philosophy models are closed-source)
