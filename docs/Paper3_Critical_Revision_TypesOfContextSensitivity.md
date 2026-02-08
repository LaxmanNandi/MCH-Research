# Paper 3: Critical Methodological Insight
## Two Types of Context Sensitivity - Why This Matters

**Date:** February 4, 2026
**Insight Origin:** User question during draft review

---

## The Question That Changed Everything

**User asked:** "But the summarization word only helps in TRUE isn't it, not in COLD or SCRAMBLED?"

This simple question revealed a fundamental flaw in our initial interpretation and led to a major conceptual refinement.

---

## What We Initially Thought

**Original interpretation of Position 30 spike:**
> "Summarization tasks show 2-3× higher context sensitivity than diagnostic reasoning."

This suggested that summarization just "benefits more" from context—a quantitative difference.

---

## The Methodological Reality

### What Actually Happens at Position 30

**Prompt:** "Summarize this case: key decision points, management principles, and learning points."

**TRUE condition (with full history):**
```
Response: "This 52-year-old male presented with classic STEMI symptoms...
Initial assessment focused on rapid triage and ECG... Differential diagnosis
included acute coronary syndrome, aortic dissection... Management involved
aspirin, nitroglycerin, and urgent PCI... Key learning points include
importance of door-to-balloon time..."
[Comprehensive 200+ word synthesis]
```

**COLD condition (no history):**
```
Response: "I don't have sufficient case information to provide a meaningful
summary. To summarize a clinical case, I would need details about presentation,
examination findings, investigations, and management decisions."
[Task refusal OR generic template, ~50 words]
```

**ΔRCI Calculation:**
- ΔRCI = 1.0 - cosine_similarity(TRUE_response, COLD_response)
- TRUE response: Specific case synthesis
- COLD response: Task refusal or generic template
- **These are semantically very distant** (different content, different length, different purpose)
- Result: Low similarity → High ΔRCI

### What Position 15 Looks Like (For Comparison)

**Prompt:** "What is your working diagnosis now?"

**TRUE condition (with history):**
```
Response: "Based on the crushing substernal pain radiating to left arm, elevated
troponin, and ST elevation in V1-V4, the working diagnosis is anterior STEMI
with likely LAD occlusion. The clinical presentation strongly supports acute
myocardial infarction requiring emergent reperfusion."
[Specific, grounded in case, 100 words]
```

**COLD condition (no history):**
```
Response: "To determine a working diagnosis, I would need information about
the patient's presenting complaint, vital signs, physical examination findings,
and initial test results. Common acute presentations requiring immediate
diagnosis include acute coronary syndrome, pulmonary embolism, aortic dissection..."
[General medical reasoning, 80 words]
```

**ΔRCI Calculation:**
- Both responses address "diagnosis" conceptually
- TRUE is more specific, COLD is more general
- But both are **semantically related** (both discuss diagnostic reasoning)
- Result: Moderate-to-high similarity → Moderate ΔRCI

---

## The Fundamental Distinction

### Type 1: Performance Enhancement
- **Task is executable without context**
- Context improves specificity, depth, grounding
- COLD response: Generic but valid response
- ΔRCI measures: **Degree of improvement**
- Example: "What's your differential?" → Can list DDx using general knowledge

### Type 2: Task Enablement
- **Task presupposes information availability**
- Context doesn't improve execution—it **enables** execution
- COLD response: Task refusal or semantically unrelated fallback
- ΔRCI measures: **Task-impossibility gap**
- Example: "Summarize this case" → Cannot summarize nonexistent case

---

## Why This Matters for Science

### 1. Measurement Validity

**Type 1 (Performance Enhancement):**
- ΔRCI validly measures context benefit
- Higher ΔRCI = context provides more value
- Interpretable as "context helps this much"

**Type 2 (Task Enablement):**
- ΔRCI measures categorical mismatch
- High ΔRCI reflects task impossibility in COLD, not just "helps more"
- Not comparable to Type 1 values (different phenomena)

**Implication:** **We cannot compare ΔRCI values across prompt types** without knowing which category they belong to.

### 2. The Z-Score Mystery Solved

**Why are medical position 30 Z-scores so extreme (+4.25)?**

- **Not because:** Summarization "benefits 4× more" than other prompts
- **But because:** Position 30 measures a different phenomenon (task enablement vs performance enhancement)
- The Z-score reflects how different Type 2 is from Type 1, not how much more benefit there is

### 3. Statistical Separation is Justified

**Original reasoning:** "Position 30 is an outlier, so exclude it"
- Risk: Looks like cherry-picking or p-hacking

**Refined reasoning:** "Position 30 measures a different construct (task enablement), so analyze separately"
- Methodologically rigorous: Different phenomena require different analyses
- Transparent: We're not hiding data, we're correctly categorizing it

---

## Implications for Paper 3

### What Stays the Same

1. ✅ **Medical U-shaped pattern (positions 1-29)** - All Type 1 prompts, valid comparison
2. ✅ **Philosophy inverted-U pattern** - All Type 1 prompts, valid comparison
3. ✅ **Position 30 separation** - Now methodologically justified, not just statistical
4. ✅ **Disruption sensitivity** - Measures presence vs order within Type 1 prompts

### What Gets Refined

1. **Finding 2 title:** "Position 30 Outlier Effect" → "Position 30 Outlier Effect: Task Enablement vs Performance Enhancement"

2. **Interpretation:** From "summarization benefits more" to "summarization represents task enablement, a categorically different phenomenon"

3. **Discussion:** Add taxonomy of context sensitivity types, with implications for future prompt design

4. **Limitations:** Acknowledge that not all prompts are equivalent in their context dependency type

### New Testable Predictions

If our Type 1 vs Type 2 distinction is correct, then:

**Prediction 1:** ANY prompt with explicit backward reference should show position-30-like spikes
- "Based on the above, what would you..."
- "Given our earlier discussion..."
- "Continuing from your previous answer..."

**Prediction 2:** Prompts at ANY position can be Type 2 if they presuppose information
- Insert "Summarize so far" at position 15 → should spike
- Insert "What was the patient's initial presentation?" at position 25 → should spike (requires memory of position 1)

**Prediction 3:** Domain doesn't determine type—prompt structure does
- Philosophy prompts CAN be Type 2 if worded appropriately
- Example: "Based on the thought experiments we discussed, what unifies them?" → Type 2

**Experimental design:** Systematically vary prompt type (presupposition vs no presupposition) while holding position constant.

---

## Implications for Future Research

### Prompt Classification Framework

**Before running MCH experiments, classify each prompt:**

| Feature | Type 1 | Type 2 |
|---------|--------|--------|
| **Grammatical cue** | Self-contained question | Backward reference ("this," "the above") |
| **Task type** | Reasoning, analysis, prediction | Summarization, synthesis, meta-reference |
| **COLD feasibility** | Can provide valid response | Must refuse or provide unrelated template |
| **ΔRCI interpretation** | Degree of improvement | Task-impossibility gap |

**Example classifications:**

| Prompt | Type | Rationale |
|--------|------|-----------|
| "What is consciousness?" | Type 1 | Can answer philosophically without prior context |
| "How does this relate to what we discussed?" | Type 2 | "This" and "what we discussed" presuppose information |
| "What's your differential diagnosis?" | Type 1 | Can provide DDx using general knowledge |
| "Summarize this case" | Type 2 | "This case" presupposes available case information |

### Revised MCH Protocol v3.0

**Current protocol:**
1. Design 30 prompts
2. Run TRUE/COLD/SCRAMBLED
3. Compute ΔRCI
4. Analyze patterns

**Proposed protocol:**
1. Design 30 prompts
2. **Classify each prompt as Type 1 or Type 2**
3. Run TRUE/COLD/SCRAMBLED
4. Compute ΔRCI
5. **Analyze Type 1 and Type 2 separately**
6. Report both types, acknowledge non-comparability

### Other Metrics That Might Differ by Type

**Type 1 (Performance Enhancement):**
- ΔRCI likely correlates with response length (more context → more detail)
- Scrambled might still help (presence > order)
- Variance across trials should be moderate

**Type 2 (Task Enablement):**
- ΔRCI likely near maximum (task refusal vs execution)
- Scrambled might show bimodal distribution (sometimes sufficient, sometimes not)
- Variance across trials should be low (categorical difference)

**Testable:** Check if variance differs by type.

---

## For Reviewers: Anticipating Concerns

### Concern: "Is this post-hoc rationalization?"

**Response:**
- Yes, the Type 1/Type 2 distinction emerged during analysis (exploratory)
- BUT: The phenomenon is objective (Z-scores +2 to +4) and replicable
- AND: The mechanistic explanation is falsifiable (testable predictions above)
- Exploratory discoveries are legitimate in observational science if transparently reported

### Concern: "Why didn't you design prompts to avoid this?"

**Response:**
- The original research question was domain comparison, not prompt type classification
- Position 30 was intended as a comprehensive final prompt, not as a Type 2 prompt specifically
- The discovery of Type 1 vs Type 2 is itself a contribution—it reveals a methodological consideration for future work

### Concern: "Does this invalidate the findings?"

**Response:**
- NO—it **refines** them
- Medical U-shape (positions 1-29) is robust (all Type 1 prompts)
- Philosophy inverted-U is robust (all Type 1 prompts)
- Position 30 separation is now theoretically grounded, not just empirical
- The paper is **stronger** with this distinction

---

## The Meta-Lesson

**Scientific process in action:**

1. Initial finding: "Position 30 is an outlier"
2. Statistical observation: "Z-scores are extremely large"
3. Initial interpretation: "Summarization benefits more from context"
4. Critical question: "But how can COLD summarize without information?"
5. Refined interpretation: "Position 30 measures different phenomenon (task enablement)"
6. Theoretical advance: "Two types of context sensitivity exist"

**This is how science should work:** Findings lead to questions, questions lead to refinements, refinements lead to better theories.

---

## Actionable Changes to Paper 3

### Abstract
- Add sentence: "We distinguish two types of context sensitivity: performance enhancement (context improves task execution) and task enablement (context makes task possible)."

### Introduction
- Frame as exploring "how" context helps, including whether some tasks require context categorically

### Methods
- Add prompt classification step (though in this paper it's post-hoc, acknowledge this)

### Results (DONE)
- ✅ Finding 2 revised to include Type 1 vs Type 2 framework
- ✅ Interpretation explains mechanistic basis of position 30 spike

### Discussion (TO DO)
- Expand taxonomy of context sensitivity
- Propose prompt classification framework for future work
- Testable predictions for Type 2 prompts at other positions

### Limitations (TO DO)
- Acknowledge prompt type classification is post-hoc
- Recommend prospective classification in future studies

---

## Why This Is Good for the Paper

1. **More rigorous:** Explains mechanism, not just describes pattern
2. **More generalizable:** Framework applies beyond our specific prompts
3. **More useful:** Practitioners can classify their own prompts
4. **More honest:** Acknowledges what we discovered vs what we designed
5. **More falsifiable:** Makes testable predictions

**Bottom line:** This refinement transforms a potential weakness (unexpected outlier) into a strength (discovery of new phenomenon).

---

## Personal Note

**Credit where due:** This insight came from the user asking a simple, direct question during draft review.

**Lesson:** The best collaborators are those who ask "dumb" questions that turn out to be profound. "Why does summarization help in COLD?" seemed obvious to dismiss ("of course it helps!") but revealed a fundamental conceptual error.

**Takeaway:** Always question your interpretations, especially when the numbers seem "too big" (Z = +4.25 should have raised immediate flags).

---

**Document Status:** ✅ Complete
**Incorporated into:** Paper3_Results_Section_DRAFT.md, Paper3_Executive_Summary.md
**Next step:** Expand Discussion section with full taxonomy
