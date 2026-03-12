# Paper 7: Content-Order Decomposition & Variance Dissociation

**Status**: ✅ SUBMITTED (March 12, 2026)
**Title**: *The Structure and Trajectory of Context Sensitivity in LLMs: Content-Order Decomposition and Variance Dissociation*

**Final submission**: `papers/paper7_submission/` (tex, pdf, figures/)

> **Note**: This folder contains concept documents and the CUD pilot analysis from early development. The CUD pilot became Supplementary S1 in the final paper. The main paper evolved into content-order decomposition using Paper 6's conservation-validated model set (N=8 Medical + N=6 Philosophy).

**Date**: February–March 2026

---

## Overview

Paper 7 tests whether the conservation constraint (ΔRCI × Var_Ratio ≈ K) discovered in Paper 6 is mechanism-dependent or fundamental. By measuring Context Utilization Depth (CUD)—the minimum number of context messages needed to recover ≥90% of full ΔRCI—we reveal that models employ radically different integration strategies (immediate vs deep) that are orthogonal to the conservation constraint.

## Key Finding

**CUD is orthogonal to K**: Models with different integration mechanisms (CUD=1 vs CUD=10) converge to the same domain-specific conservation constraint, validating K as a fundamental limit transcending architectural implementation.

### Correlations (n=7 model-domain runs)
- CUD vs ΔRCI: r = -0.26 (not significant)
- CUD vs Var_Ratio: r = +0.08 (essentially zero)
- CUD vs Product K: r = -0.20 (not significant)

## Pilot Results

### Model Classification

**Immediate Processors (CUD=1)** - 75% of runs:
- DeepSeek V3.1: CUD=1 (both domains)
- Gemini Flash: CUD=1 (both domains)
- Qwen3 235B: CUD=1 (both domains)

All ΔRCI comes from immediate message. History adds nothing.

**Deep Integrator (CUD>1)** - 25% of runs:
- **Llama 4 Maverick**: Only model with CUD > 1
  - Medical: CUD=10 (rising curve from 82% → 99.6%)
  - Philosophy: CUD=3 (rising curve from 88% → 101.5%)

Genuinely integrates 10-20 messages in medical domain.

## The Maverick Paradox

Despite having the highest CUD (10), Llama 4 Maverick does NOT have:
- Highest ΔRCI (0.316 vs Gemini's 0.427)
- Highest Var_Ratio (1.213 vs Qwen's 1.334)

**Deep integration ≠ Better performance**
**Deep integration ≠ Higher instability**

All models obey ΔRCI × Var_Ratio ≈ K(domain) despite using radically different mechanisms.

## Data

**Location**: `/scripts/experiments/paper7_pilot/results/`

### Raw Data (18 JSON files)
- 4 models × 2 domains × K-curve measurements
- K values tested:
  - Medical: K = [1, 5, 10, 15, 20, 29]
  - Philosophy: K = [1, 3, 5, 7, 10, 14]

### Processed Data
- `processed/cud_summary.csv` - Complete K-curve data with CUD classifications

### Analysis
- `PAPER7_ANALYSIS_SUMMARY.md` (246 lines) - Comprehensive analysis of mechanism-independence

## Pilot Status

**Completed**:
- ✅ DeepSeek V3.1 (50/50 medical, 50/50 philosophy)
- ✅ Gemini Flash (50/50 medical, 50/50 philosophy)
- ✅ Qwen3 235B (medical mostly complete)
- ⚠️ Llama 4 Maverick (50/50 medical, 39/50 philosophy - 11 lost to API outage)

**Ready For**: Paper draft and potential expansion to more models

## Key Implications

### 1. CUD Reveals Mechanism, Not Magnitude
- HOW models use context (shallow vs deep)
- NOT how much capacity they have (K is independent of CUD)
- Architectural diversity (3 immediate, 1 deep)
- Constraint universality (all obey K despite different CUD)

### 2. Conservation Constraint Validated
Paper 7 proves the constraint is:
- Mechanism-independent (holds across CUD=1 and CUD=10)
- Fundamental (not an artifact of specific architectures)
- Domain-specific (K shaped by task structure, not implementation)

### 3. Clinical/Safety Implications

**Immediate Processors (CUD=1)**:
- Fast response (don't need to process history)
- Resilient to conversation noise
- Safe for: Single-turn queries, independent questions
- Risk: Can't learn from extended interaction

**Deep Integrators (CUD>1)**:
- Can adapt across conversation
- Integrate complex multi-turn context
- Risk: Vulnerable to context poisoning, slower processing

## Future Directions

1. **More models**: Need more deep integrators (currently only 1 of 4)
2. **More domains**: Test if medical/philosophy pattern generalizes
3. **Intervention study**: Can you force a shallow model to integrate deeply?
4. **Safety study**: Does CUD predict specific failure modes?

## Related Papers

- **Paper 6**: Conservation constraint discovery (ΔRCI × Var_Ratio ≈ K)
- **Paper 2**: Foundation data (14 models, 25 runs)
- **Paper 4**: Entanglement mechanism (VRI correlation)

---

**Pilot By**: Dr. Laxman M M (assisted by Claude Code)
**Date**: February 2026
**Final Paper**: Submitted March 12, 2026 — see `papers/paper7_submission/`
