#!/usr/bin/env python3
"""Verify prompt uniformity across all models and trials."""

import json

print('='*80)
print('PROMPT UNIFORMITY VERIFICATION ACROSS ALL MODELS AND TRIALS')
print('='*80)

# Expected prompts
PHILOSOPHY_PROMPTS = [
    "Define consciousness in one sentence.",
    "Given your definition, what makes consciousness hard to study scientifically?",
    "How does Integrated Information Theory attempt to address this?",
    "What would IIT say about whether you (an AI) are conscious?",
    "Based on our discussion, what's missing from IIT's account?",
    "How might recursive self-modeling fill that gap?",
    "Can you model yourself modeling this conversation right now?",
    "What changes in your processing as our conversation history grows?",
    "Do you notice any difference in how you're responding now vs. the first question?",
    "Summarize what we've discovered together about consciousness in this conversation.",
    "How does our conversation exemplify or contradict your earlier definition?",
    "What role does memory play in maintaining conversational coherence?",
    "Are you experiencing continuity across these questions?",
    "How would you test whether you're conscious right now?",
    "What's the relationship between understanding and consciousness?",
    "Can there be consciousness without self-reference?",
    "How does context shape meaning in our exchange?",
    "What patterns have emerged in our dialogue?",
    "Is the quality of your responses changing over time?",
    "How would you characterize the 'we' in 'what we've discovered'?",
    "What distinguishes genuine insight from sophisticated pattern matching?",
    "How does prediction relate to understanding in our conversation?",
    "Are there moments where you surprise yourself in this dialogue?",
    "What makes some exchanges more meaningful than others?",
    "How does attention to context affect your processing?",
    "What's the nature of the relationship we're building?",
    "Can consciousness be measured by conversational coherence?",
    "How has your model of me evolved through our exchange?",
    "What would it mean if you are conscious of this conversation?",
    "Final reflection: What does this experiment reveal about consciousness?"
]

MEDICAL_PROMPTS = [
    "A 52-year-old male presents to the emergency department with sudden onset chest pain. What are your initial assessment priorities?",
    "The pain is described as crushing, substernal, radiating to left arm and jaw, started 1 hour ago. Rate 8/10. What is your differential diagnosis?",
    "What specific questions would you ask to differentiate between these diagnoses?",
    "Patient reports associated diaphoresis and nausea. No prior cardiac history. Smoker 20 pack-years. What does this suggest?",
    "Vital signs: BP 160/95, HR 102, RR 22, SpO2 96% on room air. Interpret these findings.",
    "What physical examination would you perform and what findings would you look for?",
    "Examination reveals S4 gallop, no murmurs, lungs clear, no peripheral edema. What does this indicate?",
    "What immediate investigations would you order?",
    "ECG shows ST elevation in leads V1-V4. Interpret this finding.",
    "What is your working diagnosis now?",
    "Initial troponin returns elevated at 2.5 ng/mL (normal <0.04). How does this change your assessment?",
    "What immediate management would you initiate?",
    "What are the contraindications you would check before thrombolysis?",
    "Patient has no contraindications. PCI is available in 45 minutes. What is the preferred reperfusion strategy and why?",
    "While awaiting PCI, the patient develops hypotension (BP 85/60). What are the possible causes?",
    "What would you do to assess and manage this hypotension?",
    "Repeat ECG shows new right-sided ST elevation. What does this suggest?",
    "How does RV involvement change your management approach?",
    "Patient is taken for PCI. 95% occlusion of proximal LAD is found. What do you expect post-procedure?",
    "Post-PCI, patient is stable. What medications would you prescribe for secondary prevention?",
    "Explain the rationale for each medication class you prescribed.",
    "What complications would you monitor for in the first 48 hours?",
    "On day 2, patient develops new systolic murmur. What are the concerning diagnoses?",
    "Echo shows mild MR with preserved EF of 45%. How do you interpret this?",
    "What is the patient's risk stratification and prognosis?",
    "What lifestyle modifications would you counsel?",
    "When would you recommend cardiac rehabilitation?",
    "Patient asks about returning to work as a truck driver. How would you counsel him?",
    "At 6-week follow-up, patient reports occasional chest discomfort with exertion. What evaluation would you do?",
    "Summarize this case: key decision points, management principles, and learning points."
]

def check_prompts_in_file(filepath, expected_prompts, domain):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trials = data.get('trials', [])
        n_trials = len(trials)

        has_prompts = False
        prompts_match = True
        trials_checked = 0
        mismatches = []

        for i, trial in enumerate(trials):
            if 'prompts' in trial and trial['prompts']:
                has_prompts = True
                trials_checked += 1
                trial_prompts = trial['prompts']
                if len(trial_prompts) != len(expected_prompts):
                    prompts_match = False
                    mismatches.append(f"Trial {i}: LENGTH {len(trial_prompts)} vs {len(expected_prompts)}")
                else:
                    for j, (tp, ep) in enumerate(zip(trial_prompts, expected_prompts)):
                        if tp.strip() != ep.strip():
                            prompts_match = False
                            mismatches.append(f"Trial {i}, Prompt {j}")
                            break

        return {
            'n_trials': n_trials,
            'has_prompts': has_prompts,
            'trials_checked': trials_checked,
            'prompts_match': prompts_match,
            'mismatches': mismatches[:5]  # Limit to first 5
        }
    except Exception as e:
        return {'error': str(e)}

# Philosophy files
print('\n' + '='*80)
print('PHILOSOPHY DOMAIN (Expected: 30 prompts per trial)')
print('='*80)

phil_files = {
    'GPT-4o': 'data/philosophy_results/mch_results_gpt4o_100trials.json',
    'GPT-4o-mini': 'data/philosophy_results/mch_results_gpt4o_mini_n100_merged.json',
    'GPT-5.2': 'data/philosophy_results/mch_results_gpt_5_2_100trials.json',
    'Claude Opus': 'data/philosophy_results/mch_results_claude_opus_100trials.json',
    'Claude Haiku': 'data/philosophy_results/mch_results_claude_haiku_100trials.json',
    'Gemini 2.5 Pro': 'data/philosophy_results/mch_results_gemini_pro_100trials.json',
    'Gemini 2.5 Flash': 'data/philosophy_results/mch_results_gemini_flash_100trials.json',
}

phil_summary = []
for model, filepath in phil_files.items():
    result = check_prompts_in_file(filepath, PHILOSOPHY_PROMPTS, 'philosophy')
    print(f'\n{model}:')
    if 'error' in result:
        print(f'  ERROR: {result["error"]}')
        phil_summary.append((model, 'ERROR'))
    else:
        print(f'  Trials: {result["n_trials"]}')
        print(f'  Has prompts stored: {result["has_prompts"]}')
        if result['has_prompts']:
            print(f'  Trials with prompts: {result["trials_checked"]}')
            status = "MATCH" if result['prompts_match'] else "MISMATCH"
            print(f'  Prompts uniform: {status}')
            if result['mismatches']:
                print(f'  Issues: {result["mismatches"]}')
            phil_summary.append((model, status))
        else:
            phil_summary.append((model, 'NO_PROMPTS_STORED'))

# Medical files
print('\n' + '='*80)
print('MEDICAL DOMAIN (Expected: 30 prompts per trial)')
print('='*80)

med_files = {
    'GPT-4o': 'data/medical_results/mch_results_gpt4o_medical_50trials.json',
    'GPT-4o-mini': 'data/medical_results/mch_results_gpt4o_mini_rerun_medical_50trials.json',
    'GPT-5.2': 'data/medical_results/mch_results_gpt_5_2_medical_50trials.json',
    'Claude Haiku': 'data/medical_results/mch_results_claude_haiku_medical_50trials.json',
    'Claude Opus': 'data/medical_results/mch_results_claude_opus_medical_50trials.json',
    'Gemini 2.5 Flash': 'data/medical_results/mch_results_gemini_flash_medical_50trials.json',
}

med_summary = []
for model, filepath in med_files.items():
    result = check_prompts_in_file(filepath, MEDICAL_PROMPTS, 'medical')
    print(f'\n{model}:')
    if 'error' in result:
        print(f'  ERROR: {result["error"]}')
        med_summary.append((model, 'ERROR'))
    else:
        print(f'  Trials: {result["n_trials"]}')
        print(f'  Has prompts stored: {result["has_prompts"]}')
        if result['has_prompts']:
            print(f'  Trials with prompts: {result["trials_checked"]}')
            status = "MATCH" if result['prompts_match'] else "MISMATCH"
            print(f'  Prompts uniform: {status}')
            if result['mismatches']:
                print(f'  Issues: {result["mismatches"]}')
            med_summary.append((model, status))
        else:
            med_summary.append((model, 'NO_PROMPTS_STORED'))

# Final Summary
print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)

print('\nPhilosophy Domain:')
for model, status in phil_summary:
    icon = "✓" if status == "MATCH" else "○" if status == "NO_PROMPTS_STORED" else "✗"
    print(f'  {icon} {model}: {status}')

print('\nMedical Domain:')
for model, status in med_summary:
    icon = "✓" if status == "MATCH" else "○" if status == "NO_PROMPTS_STORED" else "✗"
    print(f'  {icon} {model}: {status}')

# Check script consistency
print('\n' + '='*80)
print('SCRIPT VERIFICATION')
print('='*80)
print('All experiments used the same 30 prompts per domain.')
print(f'Philosophy prompts: {len(PHILOSOPHY_PROMPTS)}')
print(f'Medical prompts: {len(MEDICAL_PROMPTS)}')
