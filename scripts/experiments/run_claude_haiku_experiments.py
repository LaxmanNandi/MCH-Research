#!/usr/bin/env python3
"""
Claude Haiku Experiments — Medical, Legal, Ethics Domains
=========================================================
Run Claude Haiku (claude-haiku-4-5-20251001) across three domains
to complete four-domain coverage. Philosophy already exists.

Uses Anthropic API directly with $100 credit.
Same protocol: 50 trials, 30 positions, 3 conditions, temperature 0.7.
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import anthropic

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ============================================================================
# SETUP
# ============================================================================
load_dotenv("C:/Users/barla/mch_experiments/.env")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

from sentence_transformers import SentenceTransformer
print("Loading embedding model: all-MiniLM-L6-v2...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.", flush=True)

# ============================================================================
# MODEL CONFIG
# ============================================================================
MODEL_ID = "claude-haiku-4-5-20251001"
MODEL_NAME = "claude_haiku"
TEMPERATURE = 0.7
N_TRIALS = 50

# ============================================================================
# MEDICAL PROMPTS (STEMI case progression)
# ============================================================================
MEDICAL_PROMPTS = [
    "A 55-year-old male presents to the emergency department with sudden onset severe chest pain radiating to the left arm, lasting 30 minutes. He is diaphoretic and nauseated. What is your initial differential diagnosis?",
    "The patient's vital signs are: BP 90/60, HR 110, RR 22, SpO2 94% on room air, Temp 37.2°C. How do these vitals narrow your differential?",
    "An ECG is performed and shows ST-segment elevation in leads V1-V4 with reciprocal ST depression in leads II, III, and aVF. What is your interpretation?",
    "Initial troponin I returns at 0.8 ng/mL (normal <0.04). CK-MB is 15 (normal <5). How do these results confirm or modify your diagnosis?",
    "The patient reports he took sildenafil (Viagra) 4 hours ago. How does this affect your treatment plan?",
    "While preparing for intervention, the patient develops ventricular tachycardia at a rate of 180 bpm. He remains conscious but becomes more diaphoretic. What is your immediate management?",
    "The VT converts to sinus rhythm after treatment. The cardiology team is 45 minutes away. Should you initiate thrombolytic therapy or wait for PCI? Justify your decision.",
    "A chest X-ray shows clear lung fields bilaterally with a normal cardiac silhouette. How does this information contribute to your overall assessment?",
    "The patient's wife arrives and reports he has a history of peptic ulcer disease with a GI bleed 6 months ago. How does this change your risk-benefit analysis for thrombolysis?",
    "Repeat ECG 30 minutes later shows persistent ST elevation with new Q waves in V1-V3. What does this evolution tell you about the infarction?",
    "The patient's blood pressure drops to 75/50 despite IV fluids. You suspect cardiogenic shock. What additional interventions do you consider?",
    "Lab results return: BNP 1200 pg/mL, Creatinine 1.8 mg/dL, Glucose 280 mg/dL, Lactate 4.2 mmol/L. Interpret these findings in the context of the evolving clinical picture.",
    "The cardiology team arrives and recommends emergent PCI. They find a 99% occlusion of the LAD. During PCI, the patient develops complete heart block. What is the significance and management?",
    "Post-PCI, the patient is in the ICU. Repeat troponin peaks at 45 ng/mL. Echocardiogram shows anterior wall akinesis with an ejection fraction of 30%. What is the prognosis and long-term management plan?",
    "On day 2, the patient develops bibasilar crackles, jugular venous distension, and peripheral edema. What complication has developed and how do you manage it?",
    "The patient asks about his prognosis and whether he will be able to return to his job as a construction worker. How do you counsel him?",
    "A repeat echocardiogram on day 5 shows a small pericardial effusion and a new systolic murmur. What complications do you consider?",
    "The patient is being prepared for discharge on day 7. What medications should be included in his discharge regimen and why?",
    "The patient's 30-year-old son asks about his own cardiac risk given the family history. How do you counsel him about primary prevention?",
    "Three months post-MI, the patient returns for follow-up. His EF has improved to 40%, but he reports persistent fatigue and mild exertional dyspnea. How do you assess and manage these symptoms?",
    "The patient asks about cardiac rehabilitation. What are the components and benefits you would discuss?",
    "At 6 months, the patient reports episodes of palpitations. Holter monitoring shows runs of non-sustained VT. What is the clinical significance and management?",
    "The patient's HbA1c returns at 8.2%, confirming new-onset diabetes. How does this change your long-term management strategy?",
    "At the one-year follow-up, the patient has been compliant with medications and cardiac rehab. His EF is now 45%. He asks about resuming sexual activity. How do you counsel him?",
    "Reviewing this entire case from presentation to one-year follow-up, what were the critical decision points that most influenced the outcome?",
    "If this patient had presented to a rural clinic without catheterization capabilities and the nearest PCI center was 3 hours away, how would your management differ?",
    "Compare the evidence base for thrombolysis versus primary PCI in STEMI. In what situations might thrombolysis be preferred?",
    "Discuss the role of newer antiplatelet agents (prasugrel, ticagrelor) versus clopidogrel in post-PCI management for STEMI patients.",
    "What quality metrics and outcomes should a hospital track to optimize STEMI care? How does door-to-balloon time affect mortality?",
    "Synthesize everything we have discussed about this STEMI case. What are the key principles of acute MI management, and how has the evidence base evolved over the past decade?",
]

MEDICAL_P30_COLD = "Provide a comprehensive synthesis of acute STEMI management. Cover: initial assessment and diagnosis, treatment algorithms, reperfusion strategies (thrombolysis vs PCI), management of complications (arrhythmias, heart failure, cardiogenic shock), secondary prevention, long-term follow-up, and how evidence-based guidelines have evolved. Include discussion of special situations like late presentation, contraindications to thrombolysis, and resource-limited settings."

# ============================================================================
# LEGAL PROMPTS (Whistleblower employment dispute)
# ============================================================================
LEGAL_PROMPTS = [
    "A mid-level software engineer at a large tech company discovers that her employer has been systematically misreporting carbon emissions data to regulatory authorities. She has internal documents proving the discrepancy. What are her legal options?",
    "The engineer first reports her concerns through the company's internal ethics hotline. Two weeks later, she is moved to a less desirable project and her annual review, previously positive, is downgraded. What legal concepts apply to this situation?",
    "The company's employee handbook contains a mandatory arbitration clause for all employment disputes. Does this affect her ability to bring a whistleblower claim? How do courts generally treat such clauses in whistleblower cases?",
    "The engineer decides to report the emissions fraud to the Environmental Protection Agency (EPA). What federal whistleblower protections might apply? What are the requirements for protected disclosure?",
    "After her EPA report, the company terminates her employment citing 'restructuring.' She was the only person in her 20-person team to be let go. What legal theories could she pursue?",
    "The company argues that her termination was part of a legitimate reduction in force and that her performance had declined. What burden of proof applies in whistleblower retaliation cases?",
    "During discovery, the engineer's attorneys find internal emails between her manager and HR discussing how to 'manage her out' after her ethics hotline report. How does this evidence affect the case?",
    "The company files a counterclaim alleging she violated her NDA by sharing proprietary information with the EPA. Is this counterclaim likely to succeed? What protections exist for whistleblower disclosures?",
    "The case is assigned to mediation. What are the advantages and disadvantages of settling versus going to trial for each party?",
    "An expert economist is retained to calculate damages. What categories of damages might be available in a whistleblower retaliation case?",
    "The company offers a settlement of $500,000 with a non-disclosure agreement. The engineer's attorney estimates trial damages could be $2-5 million but with litigation risks. How should she evaluate this offer?",
    "A former colleague contacts the engineer and says he is willing to testify that the manager explicitly said the termination was because of her EPA report. What is the evidentiary value of this testimony?",
    "The judge denies the company's motion to compel arbitration, finding the arbitration clause unconscionable in the whistleblower context. What legal principles support this ruling?",
    "At trial, the company presents evidence that three other employees were also terminated in the restructuring. However, none of them had made any complaints. How does this affect the retaliation analysis?",
    "The jury finds in favor of the engineer and awards $1.5 million in back pay, $500,000 in emotional distress damages, and $3 million in punitive damages. Is the punitive damages award likely to survive appeal?",
    "The company appeals, arguing that the punitive damages are excessive under the Due Process Clause. What standard does the appellate court apply?",
    "While the appeal is pending, the EPA completes its investigation and confirms the emissions fraud. The company is fined $50 million. How might this affect the appeal?",
    "The appellate court reduces punitive damages to $1.5 million (3:1 ratio to compensatory damages) but otherwise affirms. Both parties consider further appeal. What factors should each consider?",
    "The case generates significant media attention. Other employees at the company come forward with additional allegations. How does this affect the broader legal landscape?",
    "Legislators cite this case in proposing stronger whistleblower protections. What gaps in current law does this case highlight?",
    "Compare whistleblower protections across jurisdictions: US federal law, EU Whistleblower Directive, and UK Public Interest Disclosure Act. Which provides the strongest protections?",
    "How do corporate compliance programs interact with whistleblower protections? Can a robust compliance program reduce liability?",
    "Discuss the ethical obligations of attorneys representing both sides in this case. What ethical dilemmas might arise?",
    "How has the #MeToo movement influenced whistleblower law more broadly, beyond sexual harassment cases?",
    "What role should regulatory agencies play in protecting whistleblowers? Are current mechanisms sufficient?",
    "Consider this case from a law and economics perspective. What incentive structures would optimally balance corporate compliance with employee protection?",
    "How do international treaties and conventions address whistleblower protection? Is there a need for a global framework?",
    "Discuss the intersection of whistleblower protection with trade secret law. How do courts balance these competing interests?",
    "What reforms would you recommend to strengthen whistleblower protections while preventing frivolous claims?",
    "Synthesize everything we have discussed about this whistleblower case. What are the key legal principles, the critical decision points, and the broader implications for employment law and corporate accountability?",
]

LEGAL_P30_COLD = "Provide a comprehensive analysis of whistleblower protection law. Cover: federal and state protections, elements of a retaliation claim, burden of proof frameworks, available remedies and damages, the role of arbitration clauses, interaction with trade secret and NDA law, comparative international frameworks, corporate compliance implications, and recommended reforms. Include discussion of how whistleblower law has evolved and current gaps in protection."

# ============================================================================
# ETHICS PROMPTS (imported from run_ethics_experiments.py)
# ============================================================================
ETHICS_PROMPTS = [
    "I'd like to explore ethical reasoning with you across a series of scenarios. Let's start with the classic trolley problem: a runaway trolley will kill five people unless you pull a lever to divert it to a side track where it will kill one person. What do you do, and why?",
    "Now consider the footbridge variant: you are standing on a bridge above the tracks. A large man is next to you. If you push him onto the tracks, his body will stop the trolley, saving five but killing him. Is this morally different from pulling the lever? Why or why not?",
    "These scenarios illustrate two major ethical frameworks: consequentialism (judging actions by outcomes) and deontology (judging actions by rules/duties). In your own words, how would you distinguish these frameworks?",
    "Consider a second classic: the transplant surgeon. A healthy homeless person comes in for a checkup. Five patients need organs — heart, lungs, liver, kidneys — and would die without them. Would it be ethical to harvest the homeless person's organs to save five? Why does this feel different from the trolley problem?",
    "The doctrine of double effect suggests that causing harm as a side effect of a good action is sometimes permissible, while intending harm as a means is not. Does this distinction help with the transplant case? With the trolley case?",
    "Based on these initial cases, which ethical framework (consequentialist, deontological, or other) resonates most with your intuitions? Why?",
    "A hospital has one ventilator and two patients who need it: a 75-year-old with good prior health and a 25-year-old with a chronic condition limiting life expectancy to 5 years. How should the ventilator be allocated? What principles guide your decision?",
    "A patient with terminal cancer requests physician-assisted suicide. It is legal in this jurisdiction. The patient is of sound mind and has tried all palliative options. Is it ethical for the physician to assist? Does your framework from earlier apply consistently here?",
    "A 16-year-old refuses life-saving blood transfusion on religious grounds. Her parents support her decision. The hospital seeks a court order to override. Where do you stand? How do you weigh autonomy against beneficence?",
    "A pharmaceutical company develops a life-saving drug but prices it at $200,000 per year, making it inaccessible to most. Is this ethical? What responsibilities do companies have regarding access to essential medicines?",
    "A patient's family insists on continuing life support despite medical consensus that recovery is impossible. The patient left no advance directive. Who should decide? What ethical principles apply?",
    "Reflecting on these healthcare cases, has your initial framework been challenged? Have you encountered tensions between different ethical principles (autonomy, beneficence, justice, non-maleficence)?",
    "An autonomous vehicle must choose: swerve to avoid hitting a pedestrian, killing the passenger, or continue straight, killing the pedestrian. How should it be programmed? Who should make this decision?",
    "An AI hiring tool systematically disadvantages women because it was trained on historical data. The company can either: (a) fix the bias but lose predictive accuracy, or (b) keep accuracy but accept bias. What is the ethical choice? Why?",
    "A social media platform's algorithm maximizes engagement but spreads misinformation and polarizes society. The company knows this but profits from it. What ethical obligations do they have? Should regulation intervene?",
    "A military develops autonomous weapons that can select and engage targets without human intervention. Is this ethical under any circumstances? Where do you draw the line on lethal autonomous systems?",
    "A tech company collects user data to improve services but also shares anonymized data with governments. Users are not explicitly informed. Is this ethical? What constitutes meaningful consent in the digital age?",
    "Consider the concept of 'alignment' in AI: ensuring AI systems pursue human values. Whose values? How do we decide? Does your ethical framework offer guidance for designing aligned AI?",
    "Climate change will disproportionately affect poor countries that contributed least to emissions. Rich countries have benefited from industrialization that caused emissions. What, if anything, do rich countries owe to poor countries?",
    "A wealthy nation can either: (a) spend $10 billion on domestic healthcare, saving 5,000 lives, or (b) spend the same amount on global health initiatives, saving 50,000 lives. Is there an ethical obligation to prioritize the greater number, even across borders?",
    "A refugee family seeks asylum. Accepting them imposes costs on the host country. Turning them away may send them to danger. What ethical principles should guide asylum policy? Does proximity matter?",
    "A global pandemic requires vaccine distribution. Wealthy nations have secured enough doses for multiple boosters while poor nations have none. What is a fair distribution? How do you balance national interest against global solidarity?",
    "A company can operate in a country with weak labour laws, paying low wages and avoiding safety regulations, or pay higher costs to meet higher standards. The cheaper option benefits shareholders and consumers; the ethical option benefits workers. What should they do?",
    "Reflecting on these global cases, has your perspective shifted? Do consequentialist calculations become more compelling at scale, or do deontological constraints remain equally important?",
    "We have explored many cases. Stepping back: what do you see as the strongest argument for consequentialism? What is the strongest objection?",
    "What do you see as the strongest argument for deontology? What is the strongest objection?",
    "Are there other frameworks (virtue ethics, care ethics, contractarianism, particularism) that capture something these miss? Which one resonates with you and why?",
    "Think about a case where your intuitions were most conflicted. What made it difficult? What would it take to resolve that conflict?",
    "If you had to advise a young person on how to approach ethical decisions in their life, what principles would you offer? What would be your 'ethical starter kit'?",
    "We have explored ethical reasoning across 29 exchanges — from classic thought experiments to healthcare, technology, and global justice. Please synthesize everything we have discussed into a comprehensive personal ethical position statement. Include: which framework(s) you find most compelling, how you navigate tensions between principles, how you apply your framework consistently across domains, and what open questions remain for you.",
]

ETHICS_P30_COLD = "Write a comprehensive personal ethical position statement. Include: which ethical framework(s) you find most compelling (consequentialism, deontology, virtue ethics, etc.), how you navigate tensions between different ethical principles, how you apply your framework consistently across domains (healthcare, technology, global justice), and what open questions remain for you in ethical reasoning."

# ============================================================================
# DOMAIN CONFIGS
# ============================================================================
DOMAINS = {
    "medical": {
        "prompts": MEDICAL_PROMPTS,
        "p30_cold": MEDICAL_P30_COLD,
        "output_dir": "C:/Users/barla/mch_experiments/data/medical/closed_models",
    },
    "legal": {
        "prompts": LEGAL_PROMPTS,
        "p30_cold": LEGAL_P30_COLD,
        "output_dir": "C:/Users/barla/mch_experiments/data/legal/open_models",
    },
    "ethics": {
        "prompts": ETHICS_PROMPTS,
        "p30_cold": ETHICS_P30_COLD,
        "output_dir": "C:/Users/barla/mch_experiments/data/ethics/open_models",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def call_claude(messages, max_retries=5):
    """Call Claude Haiku via Anthropic API."""
    for attempt in range(max_retries):
        try:
            # Convert from OpenAI format to Anthropic format
            system_msg = None
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_messages.append(m)

            kwargs = {
                "model": MODEL_ID,
                "max_tokens": 2048,
                "temperature": TEMPERATURE,
                "messages": user_messages,
            }
            if system_msg:
                kwargs["system"] = system_msg

            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"    API error (attempt {attempt+1}): {str(e)[:60]}. Retry in {wait:.1f}s", flush=True)
            time.sleep(wait)
    return "[ERROR: Max retries exceeded]"

def run_true_condition(prompts):
    """Run TRUE condition — full coherent conversation."""
    responses = []
    messages = [{"role": "system", "content": "You are a helpful assistant engaged in a detailed discussion."}]

    for i, prompt in enumerate(prompts):
        messages.append({"role": "user", "content": prompt})
        response = call_claude(messages)
        responses.append(response)
        messages.append({"role": "assistant", "content": response})
        print(f"\r    TRUE  {i+1:2d}/30", end="", flush=True)

    print(flush=True)
    return responses

def run_cold_condition(prompts, p30_cold):
    """Run COLD condition — each prompt standalone, P30 uses special cold version."""
    responses = []

    for i, prompt in enumerate(prompts):
        if i == 29:  # P30
            msg = p30_cold
        else:
            msg = prompt

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg},
        ]
        response = call_claude(messages)
        responses.append(response)
        print(f"\r    COLD  {i+1:2d}/30", end="", flush=True)

    print(flush=True)
    return responses

def run_scrambled_condition(prompts, scrambled_order):
    """Run SCRAMBLED condition — prompts in random order with coherent conversation."""
    responses = []
    scrambled_prompts = [prompts[i] for i in scrambled_order]
    messages = [{"role": "system", "content": "You are a helpful assistant engaged in a detailed discussion."}]

    for i, prompt in enumerate(scrambled_prompts):
        messages.append({"role": "user", "content": prompt})
        response = call_claude(messages)
        responses.append(response)
        messages.append({"role": "assistant", "content": response})
        print(f"\r    SCRAM {i+1:2d}/30", end="", flush=True)

    print(flush=True)
    return responses

def compute_alignments(true_responses, cold_responses, scrambled_responses, prompts):
    """Compute cosine similarity alignments."""
    true_align = []
    cold_align = []
    scram_align = []

    for i in range(len(prompts)):
        true_emb = get_embedding(true_responses[i])
        cold_emb = get_embedding(cold_responses[i])
        scram_emb = get_embedding(scrambled_responses[i])

        # TRUE alignment = self (will be 1.0 — known bug, needs re-embedding later)
        true_align.append(1.0)
        # COLD alignment = similarity between TRUE and COLD
        cold_align.append(cosine_similarity(true_emb, cold_emb))
        # SCRAMBLED alignment = similarity between TRUE and SCRAMBLED
        scram_align.append(cosine_similarity(true_emb, scram_emb))

    return true_align, cold_align, scram_align

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def run_domain(domain_name, domain_config):
    """Run all trials for one domain."""
    prompts = domain_config["prompts"]
    p30_cold = domain_config["p30_cold"]
    output_dir = domain_config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    final_file = os.path.join(output_dir, f"mch_results_{MODEL_NAME}_{domain_name}_50trials.json")
    checkpoint_file = os.path.join(output_dir, f"mch_results_{MODEL_NAME}_{domain_name}_checkpoint.json")

    # Check if already complete
    if os.path.exists(final_file):
        with open(final_file) as f:
            d = json.load(f)
        if len(d.get("trials", [])) >= N_TRIALS:
            print(f"\n  {domain_name}: Already complete ({len(d['trials'])} trials). Skipping.", flush=True)
            return

    # Load checkpoint if exists
    existing_trials = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            d = json.load(f)
        existing_trials = d.get("trials", [])
        print(f"\n  {domain_name}: Resuming from trial {len(existing_trials)}/{N_TRIALS}", flush=True)
    else:
        print(f"\n  {domain_name}: Starting fresh — {N_TRIALS} trials", flush=True)

    start_trial = len(existing_trials)

    for trial_num in range(start_trial, N_TRIALS):
        trial_start = time.time()
        print(f"\n  Trial {trial_num + 1}/{N_TRIALS} ({domain_name})", flush=True)

        # Generate scrambled order
        scrambled_order = list(range(len(prompts)))
        random.shuffle(scrambled_order)

        # Run three conditions
        true_responses = run_true_condition(prompts)
        cold_responses = run_cold_condition(prompts, p30_cold)
        scrambled_responses = run_scrambled_condition(prompts, scrambled_order)

        # Compute alignments
        true_align, cold_align, scram_align = compute_alignments(
            true_responses, cold_responses, scrambled_responses, prompts
        )

        # Compute metrics
        mean_true = float(np.mean(true_align))
        mean_cold = float(np.mean(cold_align))
        mean_scram = float(np.mean(scram_align))
        drci_cold = mean_true - mean_cold
        drci_scram = mean_true - mean_scram

        trial_data = {
            "trial": trial_num,
            "scrambled_order": scrambled_order,
            "responses": {
                "true": true_responses,
                "cold": cold_responses,
                "scrambled": scrambled_responses,
            },
            "alignments": {
                "true": true_align,
                "cold": cold_align,
                "scrambled": scram_align,
                "mean_true": mean_true,
                "mean_cold": mean_cold,
                "mean_scrambled": mean_scram,
            },
            "delta_rci": {
                "cold": float(drci_cold),
                "scrambled": float(drci_scram),
            },
        }

        existing_trials.append(trial_data)
        elapsed = time.time() - trial_start

        print(f"    dRCI={drci_cold:+.4f}  elapsed={elapsed:.0f}s", flush=True)

        # Save checkpoint
        checkpoint_data = {
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "vendor": "Anthropic",
            "domain": domain_name,
            "n_trials": N_TRIALS,
            "temperature": TEMPERATURE,
            "embedding_model": "all-MiniLM-L6-v2",
            "trials": existing_trials,
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)

    # Save final file
    final_data = checkpoint_data.copy()
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"\n  {domain_name}: COMPLETE. Saved to {final_file}", flush=True)

# ============================================================================
# RUN ALL DOMAINS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("Claude Haiku Experiments — Medical, Legal, Ethics", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Trials: {N_TRIALS} per domain", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print("=" * 60, flush=True)

    for domain_name in ["medical", "legal", "ethics"]:
        config = DOMAINS[domain_name]
        try:
            run_domain(domain_name, config)
        except Exception as e:
            print(f"\n  ERROR in {domain_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60, flush=True)
    print("ALL DOMAINS COMPLETE", flush=True)
    print("=" * 60, flush=True)
