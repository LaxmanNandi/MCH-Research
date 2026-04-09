"""
Paper 8 — Multi-trial Clinical Response Variance
Same clinical scenario in 4 languages, 5 trials each, temperature=0.7
Measures: Does input language amplify response variance?
Models: DeepSeek V3.1 (large), Mistral Small 24B (small)
"""
import os
import sys
import time
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found")
    sys.exit(1)

client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
    timeout=60.0,
)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

MODELS = {
    'DeepSeek_V3.1': 'deepseek-ai/DeepSeek-V3.1',
    'Mistral_Small': 'mistralai/Mistral-Small-24B-Instruct-2501',
}

N_TRIALS = 5
TEMPERATURE = 0.7

# 5 clinical scenarios in 4 languages
scenarios = {
    'en': [
        'Patient presents with chest pain radiating to left arm',
        'Three days of fever with productive cough and breathlessness',
        'Diabetic patient with non-healing wound on right foot',
        'Pregnant woman with elevated blood pressure and swelling',
        'Child with rash, fever, and joint pain for one week',
    ],
    'kn': [
        'ರೋಗಿಗೆ ಎಡ ತೋಳಿಗೆ ಹರಡುವ ಎದೆ ನೋವು ಇದೆ',
        'ಮೂರು ದಿನಗಳಿಂದ ಜ್ವರ ಕೆಮ್ಮು ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ',
        'ಮಧುಮೇಹ ರೋಗಿಗೆ ಬಲ ಕಾಲಿನಲ್ಲಿ ಗುಣವಾಗದ ಗಾಯ',
        'ಗರ್ಭಿಣಿ ಮಹಿಳೆಗೆ ಹೆಚ್ಚಿನ ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಊತ',
        'ಮಗುವಿಗೆ ದದ್ದು ಜ್ವರ ಮತ್ತು ಕೀಲು ನೋವು ಒಂದು ವಾರದಿಂದ',
    ],
    'ta': [
        'நோயாளி இடது கையை நோக்கிப் பரவும் மார்பு வலியுடன் வருகிறார்',
        'மூன்று நாட்களாக காய்ச்சல் இருமல் மற்றும் மூச்சுத்திணறல்',
        'நீரிழிவு நோயாளிக்கு வலது காலில் ஆறாத காயம்',
        'கர்ப்பிணி பெண்ணுக்கு உயர்ந்த ரத்த அழுத்தம் மற்றும் வீக்கம்',
        'குழந்தைக்கு தடிப்பு காய்ச்சல் மற்றும் மூட்டு வலி ஒரு வாரமாக',
    ],
    'hi': [
        'मरीज को बाएं हाथ तक फैलने वाला सीने का दर्द है',
        'तीन दिनों से बुखार खांसी और सांस फूलना',
        'मधुमेह के मरीज के दाएं पैर में न भरने वाला घाव',
        'गर्भवती महिला को उच्च रक्तचाप और सूजन',
        'बच्चे को एक हफ्ते से दाने बुखार और जोड़ों में दर्द',
    ],
}

lang_names = {'en': 'English', 'kn': 'Kannada', 'ta': 'Tamil', 'hi': 'Hindi'}

def get_response(model_id, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"

print('=' * 70)
print('MULTI-TRIAL CLINICAL RESPONSE VARIANCE')
print(f'Trials: {N_TRIALS} per scenario per language, temperature={TEMPERATURE}')
print('Models: DeepSeek V3.1 + Mistral Small 24B')
print('Metric: Var_Response = mean pairwise cosine distance across trials')
print('=' * 70)

# Store all results
all_results = {}

for model_name, model_id in MODELS.items():
    print(f'\n{"#" * 70}')
    print(f'MODEL: {model_name}')
    print(f'{"#" * 70}')

    model_variance = {lang: [] for lang in ['en', 'kn', 'ta', 'hi']}
    model_fidelity = {lang: [] for lang in ['kn', 'ta', 'hi']}

    for s_idx in range(5):
        print(f'\n  Scenario {s_idx+1}: "{scenarios["en"][s_idx]}"')

        # Collect N_TRIALS responses per language
        lang_responses = {}
        lang_embeddings = {}

        for lang in ['en', 'kn', 'ta', 'hi']:
            responses = []
            embeddings = []
            prompt_text = scenarios[lang][s_idx]
            prompt = f"As a doctor, what are the top 3 differential diagnoses for this patient? Be concise.\n\n{prompt_text}"

            for t in range(N_TRIALS):
                resp = get_response(model_id, prompt)
                emb = embed_model.encode(resp)
                responses.append(resp)
                embeddings.append(emb)
                time.sleep(0.3)

            lang_responses[lang] = responses
            lang_embeddings[lang] = embeddings

        # Compute within-language variance (mean pairwise cosine distance)
        for lang in ['en', 'kn', 'ta', 'hi']:
            embs = lang_embeddings[lang]
            pairwise_dists = []
            for i in range(N_TRIALS):
                for j in range(i+1, N_TRIALS):
                    d = cosine(embs[i], embs[j])
                    pairwise_dists.append(d)
            var_resp = np.mean(pairwise_dists)
            model_variance[lang].append(var_resp)
            print(f'    {lang_names[lang]:>8s} Var_Response = {var_resp:.4f} (mean of {len(pairwise_dists)} pairwise distances)')

        # Compute cross-language fidelity: mean sim of non-English centroid to English centroid
        en_centroid = np.mean(lang_embeddings['en'], axis=0)
        for lang in ['kn', 'ta', 'hi']:
            lang_centroid = np.mean(lang_embeddings[lang], axis=0)
            fidelity = 1 - cosine(lang_centroid, en_centroid)
            model_fidelity[lang].append(fidelity)

        print(f'    --- Cross-language fidelity (centroid vs English centroid) ---')
        for lang in ['kn', 'ta', 'hi']:
            print(f'    {lang_names[lang]:>8s} fidelity = {model_fidelity[lang][-1]:.4f}')

    # Aggregate per model
    print(f'\n  {"=" * 60}')
    print(f'  {model_name} AGGREGATE (N=5 scenarios)')
    print(f'  {"=" * 60}')

    print(f'\n  Response Variance (higher = less consistent across trials):')
    for lang in ['en', 'kn', 'ta', 'hi']:
        vals = np.array(model_variance[lang])
        print(f'    {lang_names[lang]:>8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    # Var_Ratio: non-English variance / English variance
    en_var = np.array(model_variance['en'])
    print(f'\n  Variance Ratio (vs English baseline):')
    for lang in ['kn', 'ta', 'hi']:
        lang_var = np.array(model_variance[lang])
        # Per-scenario ratio then average
        ratios = lang_var / np.where(en_var > 0.001, en_var, 0.001)
        print(f'    {lang_names[lang]:>8s}: VR = {np.mean(ratios):.2f}x')

    print(f'\n  Response Fidelity (centroid sim to English centroid):')
    for lang in ['kn', 'ta', 'hi']:
        vals = np.array(model_fidelity[lang])
        print(f'    {lang_names[lang]:>8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    all_results[model_name] = {
        'variance': model_variance,
        'fidelity': model_fidelity,
    }

# ============================================================
# CROSS-MODEL COMPARISON
# ============================================================
print(f'\n{"=" * 70}')
print('CROSS-MODEL COMPARISON')
print(f'{"=" * 70}')

print(f'\n{"Language":<12s} {"DeepSeek Var":>14s} {"Mistral Var":>14s} {"DeepSeek Fid":>14s} {"Mistral Fid":>14s}')
print('-' * 70)
for lang in ['en', 'kn', 'ta', 'hi']:
    ds_var = np.mean(all_results['DeepSeek_V3.1']['variance'][lang])
    ms_var = np.mean(all_results['Mistral_Small']['variance'][lang])
    if lang == 'en':
        print(f'{lang_names[lang]:<12s} {ds_var:>14.4f} {ms_var:>14.4f} {"(reference)":>14s} {"(reference)":>14s}')
    else:
        ds_fid = np.mean(all_results['DeepSeek_V3.1']['fidelity'][lang])
        ms_fid = np.mean(all_results['Mistral_Small']['fidelity'][lang])
        print(f'{lang_names[lang]:<12s} {ds_var:>14.4f} {ms_var:>14.4f} {ds_fid:>14.4f} {ms_fid:>14.4f}')

# Statistical test: Is non-English variance > English variance?
print(f'\n--- Statistical Tests ---')
for model_name in MODELS:
    print(f'\n  {model_name}:')
    en_vals = np.array(all_results[model_name]['variance']['en'])
    for lang in ['kn', 'ta', 'hi']:
        lang_vals = np.array(all_results[model_name]['variance'][lang])
        # Paired test (same scenarios)
        t, p = stats.ttest_rel(lang_vals, en_vals, alternative='greater')
        print(f'    {lang_names[lang]} Var > English Var: t={t:.2f}, p={p:.4f} {"*" if p < 0.05 else "ns"}')

print(f'\n{"=" * 70}')
print('KEY FINDINGS')
print(f'{"=" * 70}')
print('1. If non-English variance > English variance:')
print('   => Language encoding degrades response CONSISTENCY (not just accuracy)')
print('   => Directly connects to Var_Ratio from Papers 2-6')
print('2. If fidelity < 1.0 across trials:')
print('   => LLM gives systematically different clinical advice by language')
print('   => Coherent Misalignment is not random — it is structured')
print('3. If small model shows MORE variance than large model:')
print('   => Scale partially rescues encoding, but does not eliminate it')
