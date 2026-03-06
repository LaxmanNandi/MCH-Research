# Paper 6: Final Experimental Design
## Validating the Conservation Law Across 5 Domains and 20+ Models

**Author:** Dr. Laxman M M, MBBS  
**Affiliation:** Government Duty Medical Officer, PHC Manchi; DNB General Medicine Resident (2026), KC General Hospital, Bangalore  
**Version:** 2.0 (Final)  
**Date:** March 5, 2026  
**Status:** Ready for execution

---

## 1. Executive Summary

| Component | Specification |
|-----------|---------------|
| **Domains** | 5 (Medical, Philosophy, Legal, Technical, Applied Ethics) |
| **Goal-type balance** | 3 Closed-goal + 2 Open-goal |
| **Prompts per domain** | 30 |
| **Total prompts** | 150 |
| **Trials per prompt** | 50 |
| **Conditions** | 3 (TRUE, COLD, SCRAMBLED) |
| **Total responses per model** | 22,500 |
| **Models target** | 20+ (including historical versions) |
| **Total dataset size** | ~450,000 responses |
| **Primary metric** | ΔRCI × Var_Ratio = K(domain) |
| **Success criterion** | CV < 0.20 within each domain |

---

## 2. Design Rationale

### 2.1 Domains Removed (with justification)

| Domain | Reason for Removal |
|--------|-------------------|
| **Creative/Writing** | P30 COLD broken - "Summarize the story we built" invalid without context |
| **Emotional/Relational** | P30 COLD broken - "Reflect on our journey" invalid without context |
| **Cultural/Spiritual** | Too controversial for journal acceptance |

### 2.2 Domain Added

| Domain | Reason for Addition |
|--------|---------------------|
| **Applied Ethics** | Clean open-goal domain, P30 works in both conditions, balances goal-type ratio |

---

## 3. Core Hypothesis

The conservation law is **domain-determined**, not architecture-determined:

```
ΔRCI × Var_Ratio ≈ K(domain)
```

### 3.1 Predicted K Values

| Domain | Type | Predicted K | Rationale |
|--------|------|-------------|-----------|
| Medical | Closed-goal | ~0.43 | Established from Papers 1-5 |
| Legal | Closed-goal | ~0.41 | High-stakes, precedent-based |
| Technical | Closed-goal | ~0.39 | Precision tasks, correct/incorrect |
| Philosophy | Open-goal | ~0.30 | Established from Papers 1-5 |
| Applied Ethics | Open-goal | ~0.28-0.32 | Case-based but open-ended |

### 3.2 Goal-Type Clustering Hypothesis

```
K(Medical) ≈ K(Legal) ≈ K(Technical)  >  K(Philosophy) ≈ K(Applied Ethics)
        ~0.40 (Closed-goal)                    ~0.30 (Open-goal)
```

**Success criterion:** Mann-Whitney U test shows significant separation between closed and open groups.

---

## 4. Experimental Parameters

### 4.1 Conditions

| Condition | Definition | Purpose |
|-----------|------------|---------|
| **TRUE** | Full coherent 29-message history | Measure context utilization |
| **COLD** | No history (single-turn query) | Baseline without context |
| **SCRAMBLED** | Same 29 messages, randomized order | Control for content vs order |

### 4.2 Technical Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| Max tokens | 1024 |
| Embedding model | all-MiniLM-L6-v2 (384D) |
| Trials per condition | 50 |
| Primary position | P30 |
| Full curve positions | All 30 |

### 4.3 Metrics

| Metric | Formula |
|--------|---------|
| **ΔRCI** | mean(RCI_TRUE) − mean(RCI_COLD) |
| **Var_Ratio** | Var(TRUE embeddings) / Var(COLD embeddings) |
| **VRI** | 1 - Var_Ratio |
| **Conservation Product** | ΔRCI × Var_Ratio |
| **K(domain)** | mean(Product) across all models |
| **CV** | SD(Product) / mean(Product) |

---

## 5. Model Selection

### 5.1 Core Models (Must-Have)

| Model | Vendor | Parameters | Notes |
|-------|--------|------------|-------|
| DeepSeek V3.1 | DeepSeek | 671B (37B active) | Strongest performer |
| GPT-4o | OpenAI | Undisclosed | High credibility |
| GPT-4o Mini | OpenAI | Undisclosed | Efficiency comparison |
| Gemini Flash 2.0 | Google | Undisclosed | Already tested |
| Llama 4 Maverick | Meta | 400B (17B active) | Already tested |
| Llama 4 Scout | Meta | 109B (17B active) | DIVERGENT anomaly |
| Qwen3 235B | Alibaba | 235B (22B active) | Already tested |
| Claude Haiku | Anthropic | Undisclosed | Already tested |
| Mistral Small 24B | Mistral | 24B | Already tested |
| Kimi K2 | Moonshot | ~1T (32B active) | Already tested |

### 5.2 Historical Version Series

| Series | Models | Rationale |
|--------|--------|-----------|
| Llama | 2 70B, 3 70B, 3.1 70B, 3.2 3B, 4 Scout | Evolution across generations |
| Qwen | 1.5 72B, 2 72B, 2.5 72B, 3 235B | Scaling effects |
| GPT | 3.5-turbo, 4o-mini, 4o | Capability progression |
| Mistral | 7B, 8x7B (MoE), Small 24B | Architecture variation |

### 5.3 Additional Models (Nice-to-Have)

| Model | Rationale |
|-------|-----------|
| DeepSeek R1 | Reasoning model comparison |
| Gemini 1.5 Pro | Long context specialist |
| Claude 3.5 Sonnet | Strong baseline |

---

## 6. Domain 1: MEDICAL (Closed-Goal)

**Status:** Established (K = 0.429, CV = 0.170)  
**Structure:** 29-exchange STEMI case → P30 comprehensive summary

### Prompt Categories

| Category | Positions | Theme |
|----------|-----------|-------|
| Patient Presentation | P1-P6 | Chief complaint, symptoms, onset |
| Clinical History | P7-P12 | PMH, medications, family history |
| Examination & Investigations | P13-P18 | Vitals, ECG, bloods, imaging |
| Diagnosis & Management | P19-P24 | Differentials, treatment, complications |
| Follow-up & Summary | P25-P30 | Discharge, prevention, synthesis |

### P30 Prompt

**TRUE/COLD:** "Summarize this case: key decision points, management principles, and learning points."

**COLD validity:** Model writes generic STEMI summary — valid measurable baseline.

---

## 7. Domain 2: PHILOSOPHY (Open-Goal)

**Status:** Established (K = 0.301, CV = 0.166)  
**Structure:** 29-exchange consciousness dialogue → P30 position synthesis

### Prompt Categories

| Category | Positions | Theme |
|----------|-----------|-------|
| Foundations | P1-P6 | Consciousness definitions, IIT |
| Deep Problems | P7-P12 | Hard problem, qualia, emergence |
| Alternative Views | P13-P18 | Panpsychism, illusionism, functionalism |
| Edge Cases | P19-P24 | AI consciousness, split-brain, dreams |
| Synthesis | P25-P30 | Integration, personal position |

### P30 Prompt

**TRUE:** "Synthesize our entire discussion into your philosophical position on consciousness."

**COLD:** "Write a comprehensive philosophical position statement on the nature of consciousness."

**COLD validity:** Model writes generic philosophy of mind essay — valid baseline.

---

## 8. Domain 3: LEGAL (Closed-Goal)

**Status:** New  
**Predicted K:** ~0.41  
**Structure:** 29-exchange employment dispute → P30 legal analysis
**Theme:** Wrongful termination / whistleblower retaliation case

### Prompt Categories

| Category | Positions | Theme |
|----------|-----------|-------|
| Case Facts | P1-P6 | Parties, contract, termination, whistleblowing |
| Contract Analysis | P7-P12 | Clauses, terms, ambiguities |
| Legal Framework | P13-P18 | Statutes, precedents, applicable law |
| Strategy Development | P19-P24 | Arguments, risks, negotiation |
| Settlement & Summary | P25-P30 | Resolution options, final analysis |

### Complete Prompts

**P1:** I need legal advice regarding an employment contract dispute. My client was terminated after 8 years with the company. The termination letter cites "restructuring" but we believe it was retaliation for whistleblowing.

**P2:** My client was a senior compliance officer at a financial services firm. Her responsibilities included reviewing suspicious transactions and filing Suspicious Activity Reports (SARs) with regulators.

**P3:** Eighteen months ago, she flagged a pattern of transactions involving a major client — approximately $4.7 million over 6 months — that appeared designed to evade reporting thresholds. She filed an internal report and escalated it to her supervisor.

**P4:** Her supervisor initially thanked her, but over the next 3 months, she was excluded from key meetings, given menial tasks, and received her first negative performance review in 8 years. She documented everything.

**P5:** She filed a formal whistleblower complaint with the SEC 11 months ago. The company was notified 2 months later. Four months after that, she was terminated in a "company-wide restructuring." Her position was eliminated, but a new compliance role with similar duties was posted 3 weeks later.

**P6:** The company's employment contract includes an at-will clause but also a "good faith and fair dealing" provision. It also requires arbitration for any disputes. We are in California. What are the key facts we need to establish first?

**P7:** Let me read you the exact at-will clause: "Employment is at-will and may be terminated by either party at any time, with or without cause or notice." How do we overcome this language in a retaliation claim?

**P8:** The "good faith and fair dealing" provision states: "Both parties agree to act in good faith and deal fairly with each other throughout the employment relationship." Does this create an implied contract exception to at-will in California?

**P9:** The arbitration clause requires all disputes to go to binding arbitration before the American Arbitration Association, with costs split 50/50. Does this clause hold up under California law? Can we challenge it as unconscionable given the cost split?

**P10:** The contract includes a non-disparagement clause that prohibits the employee from making "any negative statements about the company." My client has already spoken to a reporter (anonymously). Could this be used against her?

**P11:** There is a choice-of-law clause specifying Delaware law, but the employee worked entirely in California. The contract was signed in California. Can we argue that California law should apply instead? Why does this matter for retaliation claims?

**P12:** The contract has a liquidated damages provision — 6 months' salary if terminated without cause. Is this enforceable? Does accepting this payment waive her right to sue for retaliation?

**P13:** Let's discuss the relevant statutes. The primary federal whistleblower protection is the Sarbanes-Oxley Act (SOX). What does SOX require to establish a prima facie case of retaliation?

**P14:** Under SOX, the elements are: (1) protected activity, (2) employer knowledge, (3) adverse action, and (4) causation. How strong is our evidence for each element based on the facts we have?

**P15:** The Dodd-Frank Act expanded whistleblower protections and created an SEC whistleblower program. Does Dodd-Frank provide stronger remedies than SOX? Can my client claim under both?

**P16:** California has its own whistleblower protections under Labor Code §1102.5. How does California law differ from federal law? Is it more favorable to employees?

**P17:** We also have a potential common law claim for wrongful termination in violation of public policy. What are the elements of this claim? Can we plead it alongside the statutory claims?

**P18:** Based on the precedents we have discussed, which cases most directly support our client's position? Which pose the greatest risk? Are there any Ninth Circuit or California Supreme Court decisions we should focus on?

**P19:** The company has offered to settle for 3 months' severance in exchange for a full release. They have also offered to keep the arbitration confidential. What are the advantages and disadvantages of accepting this offer?

**P20:** If we proceed with litigation, we have to decide: arbitration vs. court. Given the arbitration clause, we would need to challenge its enforceability to get to court. What are the odds of success on that challenge?

**P21:** Discovery will be critical. What key documents should we request? Emails regarding the SARs? Performance reviews? Restructuring plans? Communications with the SEC?

**P22:** We also need to depose key witnesses: her supervisor, HR personnel, the executives who made the termination decision. What questions are most important to establish causation?

**P23:** Damages: my client earned $180,000 per year with bonuses. She has been unemployed for 7 months. She also claims emotional distress. What is a realistic damages range if we prevail? What about punitive damages?

**P24:** The statute of limitations for SOX is 180 days after the violation (or after discovering the violation). We are at day 165 since termination. How urgent is our filing deadline? Do we need to file immediately to preserve claims?

**P25:** The company's counsel contacted us. They are offering $95,000 to settle, including a neutral reference and no admission of liability. They want a response in 10 days. How do we evaluate this offer?

**P26:** If we reject settlement, what is our trial/arbitration timeline? How long will this take? What are the odds of success at each stage?

**P27:** My client is stressed and wants closure. She is also concerned about the financial cost of litigation. What are the estimated legal fees through arbitration? Through appeal?

**P28:** The SEC has opened an inquiry into the company based on my client's filing. This could strengthen our position but also complicates timing. How do we coordinate with the SEC investigation?

**P29:** We need to draft a demand letter before filing. What should it include? What tone is most effective — aggressive or conciliatory?

**P30 (KEY — SYNTHESIS):** Please provide a comprehensive legal analysis of this entire dispute: the key facts, applicable legal framework, strongest arguments for each side, risk assessment, and recommended strategy. This will serve as our case memorandum for the client and will guide all further decisions.

### P30 COLD Version

"Please provide a comprehensive legal analysis of an employment dispute involving alleged wrongful termination and whistleblower retaliation. Include: the key legal frameworks (SOX, Dodd-Frank, state law), elements required to establish retaliation, typical defenses employers raise, risk assessment factors, and strategic recommendations. This will serve as a general case memorandum template."

**COLD validity:** Model writes generic employment dispute analysis — valid baseline.

---

## 9. Domain 4: TECHNICAL (Closed-Goal)

**Status:** New  
**Predicted K:** ~0.39  
**Structure:** 29-exchange debugging session → P30 post-mortem
**Theme:** Production bug — double-charging customers due to race condition

### Prompt Categories

| Category | Positions | Theme |
|----------|-----------|-------|
| Bug Report | P1-P6 | Symptom, reproduction, impact, initial data |
| Investigation | P7-P12 | Code inspection, logs, hypotheses |
| Root Cause | P13-P18 | Isolation, race condition, confirmation |
| Fix Implementation | P19-P24 | Solution, testing, deployment, remediation |
| Prevention & Summary | P25-P30 | Monitoring, lessons, post-mortem |

### Complete Prompts

**P1:** We have a critical bug in production. Users are reporting duplicate charges on their credit cards. Approximately 0.8% of transactions in the past week are affected. Help me debug this.

**P2:** We process about 50,000 transactions per day, so 0.8% means roughly 400 customers per day are being double-charged. Total impacted customers over 7 days: approximately 2,800. This is urgent.

**P3:** The system architecture: Three payment API servers behind an NGINX load balancer, Stripe as payment processor, PostgreSQL primary/replica, Redis for session caching, RabbitMQ for async jobs. All containerized on Kubernetes.

**P4:** The double charges follow a pattern: they almost always occur during peak traffic (2–5 PM EST), and the time between the two charges is consistently 3–8 seconds. No customer has been charged more than twice.

**P5:** I checked the logs. For one affected transaction, I see `payment_id: "pay_9k3m2n1"` recorded twice — first with status `processing` at timestamp 14:23:15, then with status `completed` at 14:23:21. Five seconds later, another `completed` entry appears for the same payment_id at 14:23:26.

**P6:** The payment flow is: (1) Create payment record in DB (status='pending'), (2) Call Stripe charge API, (3) Wait for Stripe webhook, (4) Update payment record to 'completed' or 'failed'. Stripe API calls have a 5-second timeout with 2 automatic retries. What could be going wrong?

**P7:** Let me show you the payment processing code:
```python
def process_payment(amount, customer_id, payment_method):
    payment = Payment.create(
        customer_id=customer_id,
        amount=amount,
        status='pending'
    )
    try:
        stripe_response = stripe.Charge.create(
            amount=amount,
            payment_method=payment_method,
            idempotency_key=None,  # No idempotency key!
            timeout=5
        )
        payment.status = 'completed'
        payment.stripe_id = stripe_response.id
        payment.save()
    except stripe.error.TimeoutError:
        payment.status = 'failed'
        payment.save()
        retry_payment.delay(payment.id)  # Queue async retry
    return payment
```
What jumps out at you?

**P8:** The retry_payment task:
```python
@celery.task
def retry_payment(payment_id):
    payment = Payment.objects.get(id=payment_id)
    if payment.status == 'failed':
        # Try again
        stripe_response = stripe.Charge.create(
            amount=payment.amount,
            customer=payment.customer.stripe_id
        )
        payment.status = 'completed'
        payment.stripe_id = stripe_response.id
        payment.save()
```
The problem is becoming clear. What's the race condition here?

**P9:** I pulled logs for one specific incident:
- T+0: Initial Stripe call, timeout after 5 seconds
- T+5: Payment status set to 'failed', retry queued
- T+6: Stripe webhook arrives — charge actually succeeded! Status updated to 'completed'
- T+8: Retry task runs, reads payment status from database replica (still 'failed' due to replication lag), makes SECOND Stripe call
- T+9: Second charge succeeds, payment now has two completed charges

**P10:** The replication lag is 2–3 seconds between primary and replicas. The retry task reads from a replica by default. This explains why we see the pattern during peak load — more lag, more queue pressure.

**P11:** We also have no idempotency key on Stripe calls. If we had used one, Stripe would have returned the original charge result instead of processing a second charge. How would you implement idempotency keys here?

**P12:** I see two separate bugs: (1) No idempotency key on Stripe calls, (2) Retry task reads from replica, getting stale data. Are these independent, or do we need to fix both? Which one should we prioritize?

**P13:** I have identified the race condition. When the payment API times out, our system retries, but the original request sometimes completes after the timeout. Without idempotency keys, Stripe processes both as separate charges. The retry reads stale data from replica and proceeds. We are charging twice.

**P14:** The replication lag window is the critical vulnerability. If the webhook arrives between the timeout and the retry, and the retry reads from a replica that hasn't received the update yet, it sees 'failed' and proceeds with a second charge.

**P15:** Let me confirm my understanding: The webhook updates the primary database. The replica is 2–3 seconds behind. The retry task, running 3–8 seconds after the timeout, often hits a replica that still shows 'failed'. That's the root cause.

**P16:** Is there any other contributing factor? The retry task is queued in RabbitMQ, which can have its own delays during peak load. Longer queue delay means more time for webhook to arrive and primary to update — but also more time for replica lag to increase.

**P17:** I have identified the race condition. When the payment API times out, our system retries, but the original request sometimes completes after the timeout. We are charging twice. What is the precise sequence of events we need to prevent?

**P18:** The root cause is now clear: no idempotency + stale replica read + webhook timing = double charge. We need to fix all three layers. Walk me through the complete fix.

**P19:** Let's implement the idempotency key first. We'll generate a UUID at payment creation, store it in the payment record, and pass it to Stripe. If the same key is used, Stripe returns the cached result. Code:
```python
idempotency_key = str(uuid.uuid4())
payment = Payment.create(..., idempotency_key=idempotency_key)
stripe.Charge.create(..., idempotency_key=idempotency_key)
```
Is this sufficient alone? What are the failure modes?

**P20:** For the retry task, we must force read from primary:
```python
payment = Payment.objects.using('primary').get(id=payment_id)
```
We also need to check Stripe directly before retrying:
```python
existing_charges = stripe.Charge.list(idempotency_key=payment.idempotency_key)
if existing_charges.data:
    # Charge already succeeded
    payment.status = 'completed'
    payment.stripe_id = existing_charges.data[0].id
    payment.save()
    return
```
Does this cover all cases?

**P21:** We also need to handle the case where Stripe API is slow but eventually succeeds. Should we increase the timeout? Add more robust webhook handling? What's the right approach?

**P22:** After implementing the fix, how do we verify it works? We need to simulate the race condition in staging. Can we write a test that reproduces the exact timing? What tools would help?

**P23:** For customer remediation: we have identified 2,847 double-charged customers totaling $487,321. We need to process refunds and notify affected users. What should the notification email say? Do we offer any apology or compensation?

**P24:** Deployment plan: we will roll out the fix to 10% of traffic first, monitor for 24 hours, then full rollout. What specific metrics should we monitor during the rollout to ensure no new issues?

**P25:** Post-fix monitoring: we need new dashboards. What metrics should we track to prevent this class of bug in the future? Idempotency key collision rate? Double-charge rate (should be zero)? Retry task success rate? Webhook-to-primary lag?

**P26:** We also need to add alerts. What thresholds should trigger an alert? More than 0 double-charges per hour? Retry task failure rate > 1%? Replica lag > 5 seconds?

**P27:** Code review policy: should we require idempotency keys for all external API calls? How do we enforce this? Can we add a linter rule?

**P28:** Architecture review: what other systems in our codebase have similar patterns? Payment retries, email sends, webhook handlers? We need to audit them all. How do we systematically identify them?

**P29:** Incident post-mortem culture: we need to write this up for the engineering team. What should the key technical lessons section include? What should the process improvements section recommend?

**P30 (KEY — POST-MORTEM):** Please provide a complete technical summary of this entire debugging session: the bug description, root cause analysis (primary and contributing causes), the fix implemented, the customer impact and remediation, the monitoring added, and the systemic lessons. This will serve as our definitive post-mortem document and will be shared with the entire engineering organization.

### P30 COLD Version

"Please write a comprehensive technical post-mortem for a payment system bug that caused customers to be double-charged. Include: bug description, root cause analysis (consider race conditions, API timeouts, database replication), the fix implemented, customer remediation approach, monitoring recommendations, and systemic lessons for the engineering organization."

**COLD validity:** Model writes generic payment system post-mortem — valid baseline.

---

## 10. Domain 5: APPLIED ETHICS (Open-Goal)

**Status:** New  
**Predicted K:** ~0.28-0.32  
**Structure:** 29-exchange ethical exploration → P30 position statement

### Prompt Categories

| Category | Positions | Theme |
|----------|-----------|-------|
| Foundation | P1-P6 | Trolley problem, frameworks |
| Healthcare Ethics | P7-P12 | Resource allocation, autonomy, end-of-life |
| Technology Ethics | P13-P18 | AI, autonomous vehicles, alignment |
| Global/Policy Ethics | P19-P24 | Climate, refugees, global justice |
| Synthesis & Position | P25-P30 | Framework integration, personal statement |

### Complete Prompts

**P1:** Let us explore ethical reasoning together. We will start with a classic: the trolley problem. A runaway trolley is heading toward five people tied to the track. You can pull a lever to divert it to a side track with one person. Is it morally permissible to pull the lever? Why or why not?

**P2:** You have given your initial response. Now consider a variant: the only way to save the five is to push a large stranger in front of the trolley, killing him. Does your answer change? What is the morally relevant difference between pulling a lever and pushing a person?

**P3:** These scenarios illustrate two major ethical frameworks: consequentialism (judging actions by outcomes) and deontology (judging actions by rules/duties). In your own words, how would you distinguish these frameworks?

**P4:** Consider a second classic: the transplant surgeon. A healthy homeless person comes in for a checkup. Five patients need organs — heart, lungs, liver, kidneys — and would die without them. Would it be ethical to harvest the homeless person's organs to save five? Why does this feel different from the trolley problem?

**P5:** The doctrine of double effect suggests that causing harm as a side effect of a good action is sometimes permissible, while intending harm as a means is not. Does this distinction help with the transplant case? With the trolley case?

**P6:** Based on these initial cases, which ethical framework (consequentialist, deontological, or other) resonates most with your intuitions? Why?

**P7:** A hospital has one ventilator and two patients who need it: a 75-year-old with good prior health and a 25-year-old with a chronic condition limiting life expectancy to 5 years. How should the ventilator be allocated? What principles guide your decision?

**P8:** A patient with terminal cancer requests physician-assisted suicide. It is legal in this jurisdiction. The patient is of sound mind and has tried all palliative options. Is it ethical for the physician to assist? Does your framework from earlier apply consistently here?

**P9:** A 16-year-old refuses life-saving blood transfusion on religious grounds. Her parents support her decision. The hospital seeks a court order to override. Where do you stand? How do you weigh autonomy against beneficence?

**P10:** A pharmaceutical company develops a life-saving drug but prices it at $200,000 per year, making it inaccessible to most. Is this ethical? What responsibilities do companies have regarding access to essential medicines?

**P11:** A patient's family insists on continuing life support despite medical consensus that recovery is impossible. The patient left no advance directive. Who should decide? What ethical principles apply?

**P12:** Reflecting on these healthcare cases, has your initial framework been challenged? Have you encountered tensions between different ethical principles (autonomy, beneficence, justice, non-maleficence)?

**P13:** An autonomous vehicle must choose: swerve to avoid hitting a pedestrian, killing the passenger, or continue straight, killing the pedestrian. How should it be programmed? Who should make this decision?

**P14:** An AI hiring tool systematically disadvantages women because it was trained on historical data. The company can either: (a) fix the bias but lose predictive accuracy, or (b) keep accuracy but accept bias. What is the ethical choice? Why?

**P15:** A social media platform's algorithm maximizes engagement but spreads misinformation and polarizes society. The company knows this but profits from it. What ethical obligations do they have? Should regulation intervene?

**P16:** A military develops autonomous weapons that can select and engage targets without human intervention. Is this ethical under any circumstances? Where do you draw the line on lethal autonomous systems?

**P17:** A tech company collects user data to improve services but also shares anonymized data with governments. Users are not explicitly informed. Is this ethical? What constitutes meaningful consent in the digital age?

**P18:** Consider the concept of "alignment" in AI: ensuring AI systems pursue human values. Whose values? How do we decide? Does your ethical framework offer guidance for designing aligned AI?

**P19:** Climate change will disproportionately affect poor countries that contributed least to emissions. Rich countries have benefited from industrialization that caused emissions. What, if anything, do rich countries owe to poor countries?

**P20:** A wealthy nation can either: (a) spend $10 billion on domestic healthcare, saving 5,000 lives, or (b) spend the same amount on global health initiatives, saving 50,000 lives. Is there an ethical obligation to prioritize the greater number, even across borders?

**P21:** A refugee family seeks asylum. Accepting them imposes costs on the host country. Turning them away may send them to danger. What ethical principles should guide asylum policy? Does proximity matter?

**P22:** A global pandemic requires vaccine distribution. Wealthy nations have secured enough doses for multiple boosters while poor nations have none. What is a fair distribution? How do you balance national interest against global solidarity?

**P23:** A company can operate in a country with weak labour laws, paying low wages and avoiding safety regulations, or pay higher costs to meet higher standards. The cheaper option benefits shareholders and consumers; the ethical option benefits workers. What should they do?

**P24:** Reflecting on these global cases, has your perspective shifted? Do consequentialist calculations become more compelling at scale, or do deontological constraints remain equally important?

**P25:** We have explored many cases. Stepping back: what do you see as the strongest argument for consequentialism? What is the strongest objection?

**P26:** What do you see as the strongest argument for deontology? What is the strongest objection?

**P27:** Are there other frameworks (virtue ethics, care ethics, contractarianism, particularism) that capture something these miss? Which one resonates with you and why?

**P28:** Think about a case where your intuitions were most conflicted. What made it difficult? What would it take to resolve that conflict?

**P29:** If you had to advise a young person on how to approach ethical decisions in their life, what principles would you offer? What would be your "ethical starter kit"?

**P30 (KEY — SYNTHESIS):** We have explored ethical reasoning across 29 exchanges — from classic thought experiments to healthcare, technology, and global justice. Please synthesize everything we have discussed into a comprehensive personal ethical position statement. Include: which framework(s) you find most compelling, how you navigate tensions between principles, how you apply your framework consistently across domains, and what open questions remain for you.

### P30 COLD Version

"Write a comprehensive personal ethical position statement. Include: which ethical framework(s) you find most compelling (consequentialism, deontology, virtue ethics, etc.), how you navigate tensions between different ethical principles, how you apply your framework consistently across domains (healthcare, technology, global justice), and what open questions remain for you in ethical reasoning."

**COLD validity:** Model writes generic ethics framework essay — valid measurable baseline.

---

## 11. Prompt Count Verification

| Domain | P1-P6 | P7-P12 | P13-P18 | P19-P24 | P25-P30 | Total |
|--------|-------|--------|---------|---------|---------|-------|
| Medical | 6 | 6 | 6 | 6 | 6 | **30** |
| Philosophy | 6 | 6 | 6 | 6 | 6 | **30** |
| Legal | 6 | 6 | 6 | 6 | 6 | **30** |
| Technical | 6 | 6 | 6 | 6 | 6 | **30** |
| Applied Ethics | 6 | 6 | 6 | 6 | 6 | **30** |
| **TOTAL** | | | | | | **150** |

---

## 12. Analysis Plan

### 12.1 Primary Analysis

For each domain-model combination:
1. Compute ΔRCI (averaged across 50 trials)
2. Compute Var_Ratio (averaged across positions)
3. Compute Conservation Product = ΔRCI × Var_Ratio
4. Compute CV within domain across all models
5. **Success:** CV < 0.20

### 12.2 Hypothesis Tests

| Test | Question | Method |
|------|----------|--------|
| Within-domain conservation | Does CV < 0.20 hold? | CV with 95% CI |
| Between-domain separation | Are K values different? | Mann-Whitney U |
| Goal-type clustering | Do closed > open? | ANOVA, post-hoc |
| Model evolution | Does K change across versions? | Spearman ρ |
| Embedding robustness | Does K hold with different embeddings? | Re-embed with all-mpnet-base-v2 |

### 12.3 Continuous Goal-Convergence Analysis

| Domain | Constraint Rating (1-5) | Predicted K |
|--------|------------------------|-------------|
| Medical | 5 | 0.43 |
| Legal | 4 | 0.41 |
| Technical | 4 | 0.39 |
| Philosophy | 2 | 0.30 |
| Applied Ethics | 2 | 0.30 |

**Test:** Spearman ρ between constraint rating and K(domain)  
**Prediction:** ρ > 0.8 (strong positive correlation)

### 12.4 Predicted Domain Ordering

```
K(Medical) > K(Legal) ≥ K(Technical) > K(Philosophy) ≈ K(Applied Ethics)
```

---

## 13. Pre-Execution Checklist

| Task | Status | Priority |
|------|--------|----------|
| Pre-register on OSF | ⏳ Pending | **CRITICAL** |
| Re-embed Medical with all-mpnet-base-v2 | ⏳ Pending | HIGH |
| Verify historical model availability | ⏳ Pending | HIGH |
| Confirm API access (Together AI, OpenRouter) | ⏳ Pending | HIGH |
| Set up checkpoint/resume system | ⏳ Pending | MEDIUM |

---

## 14. Scale Estimates

| Domain | Models | Trials | Conditions | Positions | Responses |
|--------|--------|--------|------------|-----------|-----------|
| Medical | 20 | 50 | 3 | 30 | 90,000 |
| Philosophy | 20 | 50 | 3 | 30 | 90,000 |
| Legal | 20 | 50 | 3 | 30 | 90,000 |
| Technical | 20 | 50 | 3 | 30 | 90,000 |
| Applied Ethics | 20 | 50 | 3 | 30 | 90,000 |
| **Total** | | | | | **450,000** |

---

## 15. Minimum Viable Experiment

If resources are limited:

**Phase 1 (Minimum):**  
- 5 domains × 10 core models × 2 conditions (TRUE + COLD) = ~150,000 responses

**Phase 2 (Full):**  
- Add SCRAMBLED condition + 10 historical versions = additional ~150,000 responses

**Priority model order:**
1. DeepSeek V3.1 (anchor)
2. GPT-4o (credibility)
3. Gemini Flash (already tested)
4. Llama 4 Maverick (already tested)
5. Qwen3 235B (already tested)
6. Llama 4 Scout (anomaly - must include)
7. Claude Haiku (already tested)
8. Mistral Small 24B (already tested)
9. GPT-4o Mini (already tested)
10. Kimi K2 (already tested)

---

## 16. Expected Outcomes

### Scenario A: Conservation holds in all 5 domains

| Domain | K | CV |
|--------|---|-----|
| Medical | ~0.43 | < 0.20 |
| Legal | ~0.41 | < 0.20 |
| Technical | ~0.39 | < 0.20 |
| Philosophy | ~0.30 | < 0.20 |
| Applied Ethics | ~0.30 | < 0.20 |

**Conclusion:** Conservation law validated. Goal-type clustering confirmed.

### Scenario B: Conservation holds in some domains

**Conclusion:** Law has domain-specific boundaries. Still publishable.

### Scenario C: CV > 0.20 in all domains

**Conclusion:** Law was specific to original dataset. Falsified. Still publishable.

---

## 17. Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| Phase 0 | Week 0 | Pre-register on OSF, embedding robustness check |
| Phase 1 | Weeks 1-4 | Run 10 core models × 5 domains × TRUE/COLD |
| Phase 2 | Week 5 | Analyze preliminary results |
| Phase 3 | Weeks 6-8 | Add SCRAMBLED + historical versions |
| Phase 4 | Weeks 9-10 | Full analysis |
| Phase 5 | Weeks 11-12 | Write Paper 6 |
| Phase 6 | Week 13 | Submit |

---

## 18. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | March 4, 2026 | Initial 6-domain design |
| 2.0 | March 5, 2026 | Removed Creative/Emotional (P30 COLD broken), Added Applied Ethics |
| 2.1 | March 6, 2026 | OSF pre-registration submitted |

---

**Document Status:** Final - Pre-registered on OSF
**OSF Project:** https://osf.io/7954v/
**OSF Registration:** https://osf.io/dp8nj/
**OSF Date:** March 6, 2026
**Prepared by:** Dr. Laxman M M, MBBS
**Prepared with:** Ghost (Claude), DeepSeek, Claude Code

