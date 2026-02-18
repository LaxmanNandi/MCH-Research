# Paper 7: Context Utilization Depth (CUD)

## Core Question

How many context messages does a model actually *need* to achieve its full context sensitivity?

Papers 1-6 established that context matters (ΔRCI), that it trades off with variance (Paper 6 conservation), and that domain structure shapes the pattern. Paper 7 asks the next question: **how deep into the conversation history does context sensitivity actually reach?**

## Metric Definition

**CUD (Context Utilization Depth)** = minimum K where:

```
ΔRCI_TRUNCATED(K) >= 0.90 × ΔRCI_TRUE
```

Where:
- **ΔRCI_TRUE** = 1.0 - cosine_sim(response_TRUE, response_COLD)
  - response_TRUE: generated with full conversation history
  - response_COLD: generated with no history (single prompt)
- **ΔRCI_TRUNCATED(K)** = 1.0 - cosine_sim(response_TRUNCATED(K), response_COLD)
  - response_TRUNCATED(K): generated with only the last K message pairs as context

## K-Curve

The K-curve plots ΔRCI_TRUNCATED(K) / ΔRCI_TRUE at each K value:
- **Rising curve**: model uses deeper context (CUD > 1)
- **Flat curve at ~1.0**: model gets most signal from recent context (CUD ≈ 1)
- **Noisy/flat near zero**: no meaningful context sensitivity (CUD undefined)

## Method

Consistent with MCH Papers 1-6:
- **Embedding**: all-MiniLM-L6-v2 (384D)
- **Similarity**: cosine similarity of response embeddings
- **RCI**: response-to-response (TRUE vs COLD, TRUNCATED vs COLD)
- **Temperature**: 0.7
- **Trials**: 50 per model-domain-position condition
- **Checkpoint**: every 5 trials with resume capability

### Positions and K Values

| Domain | Position | K Values | Max History |
|--------|----------|----------|-------------|
| Medical | P30 | 1, 5, 10, 15, 20, 29 | 29 message pairs |
| Philosophy | P15 | 1, 3, 5, 7, 10, 14 | 14 message pairs |

### API Calls Per Trial

Each trial generates a fresh TRUE conversation (MCH protocol: new conversation per trial), then probes the target position under COLD and each K condition:
- Medical P30: 30 (TRUE conversation) + 1 (COLD) + 6 (K values) = 37 calls
- Philosophy P15: 15 (TRUE conversation) + 1 (COLD) + 6 (K values) = 22 calls

## Pilot Design

4 models selected for vendor diversity and accessibility:

| Model | Vendor | Parameters | API |
|-------|--------|------------|-----|
| DeepSeek V3.1 | DeepSeek | 685B MoE | Together AI |
| Gemini 2.5 Flash | Google | Undisclosed | Google AI |
| Llama 4 Maverick | Meta | 17B-128E MoE | Together AI |
| Qwen3 235B | Alibaba | 235B MoE (22B active) | Together AI |

Additionally, Llama 4 Scout was run as part of early pilot exploration.

## Preliminary Findings (from completed data)

### 1. Domain Contrast Replicates

All pilot models confirm Papers 1-6:
- **Medical**: Strong ΔRCI (~0.82-0.95), meaningful K-curves
- **Philosophy**: Near-zero ΔRCI (~0.05-0.22), K-curves are noise

CUD is only meaningful where ΔRCI is substantial (medical domain).

### 2. Model CUD Classes Emerge

| Model | Medical CUD | K-Curve Shape | Interpretation |
|-------|-------------|---------------|----------------|
| DeepSeek V3.1 | 1 | Flat at ~100% | Recency-dominant |
| Gemini Flash | 1 | Flat at ~100% | Recency-dominant |
| Qwen3 235B | 1 | Flat at ~88-96% | Recency-dominant, slight gradient |
| Llama 4 Maverick | >1 | Rising: 73% → 89% | Genuine depth utilization |

### 3. Key Insight: Sensitivity vs Accuracy Dissociation

Two metrics reveal different things:
- **ΔRCI_TRUNCATED(K)**: How different is the response from COLD? (sensitivity)
- **sim_trunc_true(K)**: How similar is the response to full-context TRUE? (accuracy)

Sensitivity saturates at K=1 for most models (they move away from COLD immediately). But accuracy (convergence to the TRUE response) requires more context. This dissociation suggests models can appear "context-sensitive" with minimal history while still not reproducing the full-context response.

### 4. Maverick Shows Genuine Depth

Llama 4 Maverick is the only pilot model with a clear rising K-curve in medical:
- K=1: ~73% of TRUE sensitivity
- K=5: ~78%
- K=10: ~82%
- K=15: ~85%
- K=20: ~87%
- K=29: ~89%

This suggests architectural differences in how Maverick processes conversation history — possibly related to its 128-expert MoE design.

## Pilot Status

| Model | Medical P30 | Philosophy P15 | Status |
|-------|------------|----------------|--------|
| DeepSeek V3.1 | 50/50 | 50/50 | COMPLETE |
| Gemini Flash | 50/50 | 50/50 | COMPLETE |
| Llama 4 Maverick | 50/50 | 39/50 | Needs 11 trial rerun |
| Qwen3 235B | IN PROGRESS | 0/50 | Running |

## Known Issues

- Together AI intermittent Cloudflare 500 errors cause trial failures
- Qwen3 235B required client-level `timeout=120.0` on OpenAI() init
- Philosophy K-curves too noisy for meaningful CUD — may need different analysis
- Maverick philosophy lost 11 trials to Together AI outage

## Cross-Paper Hypothesis: CUD as Depth Signature for Var_Ratio

### The Observation

Papers 1-6 identified the Llama safety anomaly: Var_Ratio up to 7.46 at medical P30,
far exceeding other models. This was interpreted as a safety-relevant instability.

Paper 7 now reveals that Llama 4 Maverick — a different generation of the same vendor
family — is the only pilot model with a genuine rising K-curve (CUD > 1). All other
models (DeepSeek, Gemini Flash, Qwen3) show flat K-curves with CUD ≈ 1.

### The Hypothesis

High Var_Ratio and high CUD may be two measurements of the same underlying property:
**deep context dependence**.

- A model that genuinely processes deep conversation history (high CUD) generates
  responses that are more sensitive to trial-to-trial variations in that history.
  Each trial produces a slightly different conversation, and a deep-context model
  amplifies those differences → high Var_Ratio.

- A recency-dominant model (CUD ≈ 1) ignores most of the conversation history,
  so trial-to-trial variations in earlier messages have minimal effect on the
  response → low Var_Ratio.

### Predicted Pattern

| CUD Class | Expected Var_Ratio | Models |
|-----------|-------------------|--------|
| High CUD (deep) | High Var_Ratio | Llama family |
| Low CUD (recency) | Low Var_Ratio | DeepSeek, Gemini Flash, Qwen3 |

### Testable With Existing Data

This hypothesis can be verified by computing Var_Ratio for the Paper 7 pilot models
from their 50-trial data and checking whether Maverick's Var_Ratio exceeds that of
the flat-CUD models. The original Papers 1-6 Llama data (high Var_Ratio) and the
Paper 7 Maverick data (high CUD) constitute independent observations from different
model generations, strengthening the cross-paper link.

### Implications

- The Llama "safety anomaly" may not be anomalous — it may be a structural property
  of deep context processing, visible as high variance from one angle and as rising
  K-curves from another.
- Paper 6's conservation constraint (ΔRCI × Var_Ratio ≈ K) may decompose further:
  models with high CUD concentrate the conservation product in deeper context layers,
  while recency-dominant models concentrate it in K=1 alone.
- The DIVERGENT class in Paper 5's taxonomy may correspond to high-CUD models.

## Connection to Papers 1-6

- **Paper 2**: CUD extends the ΔRCI benchmark by asking not just "how much" but "how deep"
- **Paper 3**: Temporal dynamics showed position-level patterns; CUD adds the K dimension
- **Paper 4**: VRI measures variance reduction; CUD measures context depth — potentially related
- **Paper 5**: The IDEAL/EMPTY/DIVERGENT/RICH taxonomy may map to CUD classes
- **Paper 6**: Conservation constraint ΔRCI × Var_Ratio ≈ K(domain) — does CUD interact with the conservation product?

## Future Directions

- Test CUD on the original 14 Paper 2 models
- Examine CUD at multiple positions (not just P30/P15)
- Investigate CUD × conservation constraint interaction
- Compute Var_Ratio for Paper 7 pilot models to test CUD-variance hypothesis
- Test whether CUD predicts practical deployment performance
- Explore CUD in additional domains beyond medical/philosophy
