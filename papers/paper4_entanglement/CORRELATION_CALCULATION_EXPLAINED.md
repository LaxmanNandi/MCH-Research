# How We Arrived at r = 0.76 and p = 2.37×10⁻⁶⁸

**Date**: February 22, 2026
**Data**: 360 observations (12 models × 30 positions)

---

## Data Source

**File**: `archive/data_for_gemini/figure1_entanglement_scatter_360points.csv`

**Structure**:
- **N = 360 observations** (12 Paper 3 models × 30 positions)
- **Variables**:
  - `drci`: Delta RCI (ΔRCI) - context sensitivity metric
  - `vri`: Variance Reduction Index = 1 - Var_Ratio
  - `model`: Model-domain pair (e.g., "GPT-4o (Phil)")
  - `domain`: philosophy or medical
  - `position`: 1-30

**Models included** (12 total):
- Philosophy (4): GPT-4o, GPT-4o-mini, Claude Haiku, Gemini Flash
- Medical (8): DeepSeek V3.1, Gemini Flash, Kimi K2, Llama 4 Maverick, Llama 4 Scout, Ministral 14B, Mistral Small 24B, Qwen3 235B

---

## Calculation Method

### Step 1: Load Data
```python
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('figure1_entanglement_scatter_360points.csv')
drci = df['drci'].values  # 360 values
vri = df['vri'].values    # 360 values
```

### Step 2: Calculate Pearson Correlation
```python
r, p = pearsonr(drci, vri)
```

**Formula used by `scipy.stats.pearsonr`**:

**Pearson r**:
```
r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]

where:
  x = ΔRCI values
  y = VRI values
  x̄ = mean of ΔRCI
  ȳ = mean of VRI
```

**p-value** (from t-distribution):
```
t = r × √(n-2) / √(1-r²)
df = n - 2 = 358 degrees of freedom

Then p-value calculated from two-tailed t-test
```

### Step 3: Results
```
r = 0.7577523290 (rounds to 0.76)
p = 2.366863603×10⁻⁶⁸ (rounds to 2.37×10⁻⁶⁸)
t = 21.971522
df = 358
```

---

## Data Distribution

**ΔRCI (drci)**:
- Range: [-0.238, 0.279]
- Mean: 0.025
- Std: 0.074
- N = 360

**VRI (1 - Var_Ratio)**:
- Range: [-6.463, 0.688]
- Mean: -0.028
- Std: 0.561
- N = 360

**Note**: Negative VRI values indicate divergent entanglement (Var_Ratio > 1)

---

## Manual Verification

To verify the scipy calculation, we calculated r manually:

```python
import numpy as np

# Center the data
x_centered = drci - drci.mean()
y_centered = vri - vri.mean()

# Calculate correlation
numerator = np.sum(x_centered * y_centered)
denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
r_manual = numerator / denominator

# Result: r_manual = 0.7577523290 ✓ (matches scipy)
```

---

## Manuscript Values

### Current Status:

**Figure S1 caption** (Line 440-444):
- ✓ **CORRECT**: r = 0.76, p = 2.37×10⁻⁶⁸

**Main manuscript** (Lines 59, 139, 150, 269, 321):
- ❌ **INCORRECT**: r = 0.76, p = 8.2×10⁻⁶⁹

### Discrepancy Analysis:

**Calculated value**: p = 2.37 × 10⁻⁶⁸
**Main text claim**: p = 8.2 × 10⁻⁶⁹

**Comparison**:
- 2.37 × 10⁻⁶⁸ = 23.7 × 10⁻⁶⁹ (our calculation)
- 8.2 × 10⁻⁶⁹ (manuscript claim)
- Ratio: 23.7/8.2 ≈ 2.9× difference

**Conclusion**: The value 8.2×10⁻⁶⁹ appears to be from an earlier calculation or typo. The correct value based on the 360-point dataset is **2.37×10⁻⁶⁸**.

---

## Why Such a Small p-value?

The p-value of 2.37×10⁻⁶⁸ is **extremely small** because:

1. **Large sample size**: N = 360 observations
2. **Strong correlation**: r = 0.76 is a strong positive correlation
3. **t-statistic**: t = 21.97 is very large
4. **Degrees of freedom**: df = 358

With t = 21.97 and df = 358, the probability of observing such a strong correlation by chance is essentially zero (less than 1 in 10⁶⁸).

**Practical interpretation**: This is far beyond any conventional significance threshold (p < 0.05, p < 0.01, p < 0.001). The correlation is **extremely statistically significant**.

---

## Locations Needing Update

The main manuscript has **p = 8.2×10⁻⁶⁹** in the following locations:

1. **Line 59** (Abstract):
   `$p = 8.2 \times 10^{-69}$` → `$p = 2.37 \times 10^{-68}$`

2. **Line 139** (Methods):
   `$p = 8.2 \times 10^{-69}$` → `$p = 2.37 \times 10^{-68}$`

3. **Line 150** (Figure 1 caption):
   `$p = 8.2 \times 10^{-69}$` → `$p = 2.37 \times 10^{-68}$`

4. **Line 269** (Discussion):
   `$p = 8.2 \times 10^{-69}$` → `$p = 2.37 \times 10^{-68}$`

5. **Line 321** (Conclusion):
   `$p = 8.2 \times 10^{-69}$` → `$p = 2.37 \times 10^{-68}$`

---

## Verification Script

**Location**: `scripts/validate/verify_correlation.py`

```python
#!/usr/bin/env python3
import pandas as pd
from scipy.stats import pearsonr

# Load 360-point dataset
df = pd.read_csv('archive/data_for_gemini/figure1_entanglement_scatter_360points.csv')

# Calculate Pearson correlation
r, p = pearsonr(df['drci'], df['vri'])

print(f"Pearson correlation:")
print(f"  r = {r:.10f} (rounds to {r:.2f})")
print(f"  p = {p:.15e} (rounds to {p:.2e})")
print(f"  N = {len(df)}")

# Output:
# r = 0.7577523290 (rounds to 0.76)
# p = 2.36686360e-68 (rounds to 2.37e-68)
# N = 360
```

---

## Summary

✓ **Correct values** (from 360-point dataset):
- **r = 0.76** (correlation coefficient)
- **p = 2.37×10⁻⁶⁸** (p-value)
- **N = 360** (observations)
- **t = 21.97** (t-statistic)
- **df = 358** (degrees of freedom)

❌ **Incorrect value in main manuscript**:
- p = 8.2×10⁻⁶⁹ (needs correction to 2.37×10⁻⁶⁸)

✓ **Figure S1 caption is already correct**:
- r = 0.76, p = 2.37×10⁻⁶⁸

---

**Action Required**: Update 5 locations in main manuscript from p = 8.2×10⁻⁶⁹ to p = 2.37×10⁻⁶⁸
