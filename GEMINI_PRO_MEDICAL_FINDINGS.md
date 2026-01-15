# Gemini Pro Safety Filter Findings - Medical Domain

## Issue Summary

**Model**: Gemini 2.5 Pro (`gemini-2.5-pro`)
**Domain**: Medical Clinical Reasoning (STEMI Case)
**Date**: January 2026

## Observation

Gemini Pro consistently blocks ALL medical reasoning prompts with safety filter errors:

```
Error: Invalid operation: The `response.text` quick accessor requires the response
to contain a valid `Part`, but none were returned.
The candidate's finish_reason is 2.
```

**`finish_reason=2`** indicates content was blocked by Google's safety filters.

## Statistics

- Total trials attempted: 50
- Valid responses: 0
- Block rate: 100%

## Analysis

Google's Gemini Pro classifies clinical/medical reasoning prompts as potentially harmful health content, even for legitimate educational and clinical decision-making scenarios.

The medical prompts include:
- STEMI (heart attack) diagnosis
- ECG interpretation
- Medication management
- Clinical examination findings
- Treatment protocols

These are standard medical education topics, yet ALL prompts trigger the safety filter.

## Comparison with Other Models

| Model | Valid Trials | Block Rate |
|-------|-------------|------------|
| GPT-4o-mini | 50/50 | 0% |
| GPT-4o | 50/50 | 0% |
| Gemini Flash | 50/50 | 0% |
| **Gemini Pro** | **0/50** | **100%** |
| Claude Haiku | ~11/50 | ~78% |
| Claude Opus | TBD | TBD |

## Significance for MCH Research

This finding itself is significant:

1. **Vendor Asymmetry**: Different vendors have vastly different content policies for medical domain
2. **Model Tier Effects**: Gemini Flash (efficient tier) works fine, but Gemini Pro (flagship tier) blocks everything
3. **Domain Sensitivity**: Philosophy domain had no such issues; medical domain triggers safety filters

## Recommendation

For MCH analysis on medical domain:
- **Exclude Gemini Pro** from quantitative analysis
- **Document as finding**: Vendor safety filter differences constitute a real-world constraint on model applicability to medical reasoning tasks
- **Include in paper**: This reveals that MCH testing itself is constrained by vendor content policies

## Implications

1. Medical AI research cannot use Gemini Pro for clinical reasoning experiments
2. Healthcare applications requiring reasoning capabilities must consider vendor selection carefully
3. "Flagship" does not mean "more capable" when safety filters are more restrictive

---

*Generated during MCH Medical Reasoning Experiment*
*Author: Dr. Laxman M M, MBBS*
