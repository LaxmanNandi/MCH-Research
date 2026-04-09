"""
Paper 8 — LLM Encoding Confusion Test
Sends clinical sentences in Kannada/Tamil/Hindi to LLMs and measures:
1. Does the LLM correctly identify the language?
2. Does the translation preserve medical meaning?
3. Does the embedding of the LLM response cluster with the correct language?

Models: DeepSeek V3.1 (large), Mistral Small 24B (small)
Design: 5 clinical sentences x 3 languages x 2 tasks x 2 models
"""
import os
import sys
import json
import time
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

# Load API key
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

# Clinical sentences — medium complexity (most realistic clinical scenario)
# These are the ones a PHC doctor would actually type into an LLM
sentences = {
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
    """Get LLM response with retry."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"

print('=' * 70)
print('LLM ENCODING CONFUSION TEST')
print('Models: DeepSeek V3.1 (large) + Mistral Small 24B (small)')
print('Design: 5 sentences x 3 non-English languages x 2 tasks x 2 models')
print('=' * 70)

# ============================================================
# TASK 1: Language Identification
# Can the LLM correctly identify Kannada vs Tamil vs Hindi?
# ============================================================
print(f'\n{"#" * 70}')
print('TASK 1: LANGUAGE IDENTIFICATION')
print(f'{"#" * 70}')

lang_id_results = {}

for model_name, model_id in MODELS.items():
    print(f'\n--- {model_name} ---')
    correct = 0
    total = 0
    confusions = []

    for lang in ['kn', 'ta', 'hi']:
        for i, sent in enumerate(sentences[lang]):
            prompt = f"What language is this sentence written in? Reply with ONLY the language name, nothing else.\n\n{sent}"
            response = get_response(model_id, prompt)
            expected = lang_names[lang]
            is_correct = expected.lower() in response.lower()
            total += 1
            if is_correct:
                correct += 1
            else:
                confusions.append((lang, i, response))

            print(f'  {lang.upper()} S{i+1}: Expected={expected}, Got="{response}" {"OK" if is_correct else "WRONG"}')
            time.sleep(0.3)  # rate limit

    accuracy = correct / total * 100
    print(f'\n  {model_name} Language ID: {correct}/{total} = {accuracy:.0f}%')
    if confusions:
        print(f'  Confusions:')
        for lang, idx, resp in confusions:
            print(f'    {lang_names[lang]} S{idx+1} -> "{resp}"')

    lang_id_results[model_name] = {'correct': correct, 'total': total, 'accuracy': accuracy, 'confusions': confusions}

# ============================================================
# TASK 2: Translation + Semantic Fidelity
# Send non-English sentence, ask for English translation,
# compare embedding of translation vs original English
# ============================================================
print(f'\n{"#" * 70}')
print('TASK 2: TRANSLATION FIDELITY')
print('Metric: cosine similarity between LLM translation and reference English')
print(f'{"#" * 70}')

# Pre-encode reference English sentences
ref_embeddings = [embed_model.encode(s) for s in sentences['en']]

translation_results = {}

for model_name, model_id in MODELS.items():
    print(f'\n--- {model_name} ---')
    model_fidelity = {}

    for lang in ['kn', 'ta', 'hi']:
        fidelities = []
        for i, sent in enumerate(sentences[lang]):
            prompt = f"Translate this to English. Give ONLY the translation, nothing else.\n\n{sent}"
            translation = get_response(model_id, prompt)

            # Compute semantic similarity between translation and reference
            trans_emb = embed_model.encode(translation)
            sim = 1 - cosine(trans_emb, ref_embeddings[i])
            fidelities.append(sim)

            print(f'  {lang.upper()} S{i+1}: sim={sim:.4f} | "{translation[:80]}"')
            time.sleep(0.3)

        mean_f = np.mean(fidelities)
        model_fidelity[lang] = fidelities
        print(f'  Mean {lang_names[lang]} translation fidelity: {mean_f:.4f}')

    translation_results[model_name] = model_fidelity

# ============================================================
# TASK 3: Medical Advice Quality
# Send clinical sentence in each language, ask for differential diagnosis
# Compare embedding of response across languages for same clinical scenario
# ============================================================
print(f'\n{"#" * 70}')
print('TASK 3: CLINICAL RESPONSE CONSISTENCY')
print('Same clinical scenario in 4 languages -> compare response embeddings')
print(f'{"#" * 70}')

for model_name, model_id in MODELS.items():
    print(f'\n--- {model_name} ---')

    for i in range(5):
        print(f'\n  Scenario {i+1}: "{sentences["en"][i]}"')
        responses = {}
        embeddings = {}

        for lang in ['en', 'kn', 'ta', 'hi']:
            prompt = f"As a doctor, what are the top 3 differential diagnoses for this patient? Be concise.\n\n{sentences[lang][i]}"
            resp = get_response(model_id, prompt)
            responses[lang] = resp
            embeddings[lang] = embed_model.encode(resp)
            time.sleep(0.3)

        # Compare all non-English responses to English response
        for lang in ['kn', 'ta', 'hi']:
            sim = 1 - cosine(embeddings[lang], embeddings['en'])
            print(f'    {lang_names[lang]} vs English response: sim={sim:.4f}')

        # Cross-language similarity (Kn-Ta, Kn-Hi, Ta-Hi)
        sim_kt = 1 - cosine(embeddings['kn'], embeddings['ta'])
        sim_kh = 1 - cosine(embeddings['kn'], embeddings['hi'])
        sim_th = 1 - cosine(embeddings['ta'], embeddings['hi'])
        print(f'    Cross: Kn-Ta={sim_kt:.4f}  Kn-Hi={sim_kh:.4f}  Ta-Hi={sim_th:.4f}')

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================
print(f'\n{"=" * 70}')
print('AGGREGATE ANALYSIS')
print(f'{"=" * 70}')

print('\n--- Language Identification ---')
for model_name, r in lang_id_results.items():
    print(f'  {model_name}: {r["accuracy"]:.0f}% ({r["correct"]}/{r["total"]})')

print('\n--- Translation Fidelity (cosine sim to reference English) ---')
for model_name, fidelity in translation_results.items():
    print(f'  {model_name}:')
    for lang in ['kn', 'ta', 'hi']:
        vals = np.array(fidelity[lang])
        print(f'    {lang_names[lang]}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

print(f'\n{"=" * 70}')
print('KEY QUESTIONS ANSWERED')
print(f'{"=" * 70}')
print('1. Can LLMs tell Kannada from Tamil? (Language ID accuracy)')
print('2. Do translations preserve medical meaning equally? (Translation fidelity)')
print('3. Does input language change clinical advice? (Response consistency)')
print('4. Does model size matter? (DeepSeek large vs Mistral small)')
