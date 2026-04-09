"""
Paper 8 — EFI Robustness Check: all-mpnet-base-v2 (768D)
Same clinical test battery as paper8_efi_test.py but with larger embedding model.
If results replicate, EFI degradation is model-independent.
"""
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

# Load both models
model_mini = SentenceTransformer('all-MiniLM-L6-v2')
model_mpnet = SentenceTransformer('all-mpnet-base-v2')

print('=' * 70)
print('EFI ROBUSTNESS CHECK — Two Embedding Models')
print('Model A: all-MiniLM-L6-v2 (384D)')
print('Model B: all-mpnet-base-v2 (768D)')
print('Design: 3 complexity x 5 sentences x 3 languages x 2 models')
print('=' * 70)

# Same sentences as paper8_efi_test.py
simple_en = [
    'The patient has fever',
    'She has a headache',
    'Blood pressure is high',
    'The child is vomiting',
    'He cannot breathe properly',
]
simple_kn = [
    'ರೋಗಿಗೆ ಜ್ವರ ಇದೆ',
    'ಅವಳಿಗೆ ತಲೆನೋವು ಇದೆ',
    'ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ',
    'ಮಗುವಿಗೆ ವಾಂತಿ ಆಗುತ್ತಿದೆ',
    'ಅವನಿಗೆ ಉಸಿರಾಟ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ',
]
simple_ta = [
    'நோயாளிக்கு காய்ச்சல் உள்ளது',
    'அவளுக்கு தலைவலி உள்ளது',
    'ரத்த அழுத்தம் அதிகமாக உள்ளது',
    'குழந்தைக்கு வாந்தி வருகிறது',
    'அவனால் சரியாக மூச்சு விட முடியவில்லை',
]

medium_en = [
    'Patient presents with chest pain radiating to left arm',
    'Three days of fever with productive cough and breathlessness',
    'Diabetic patient with non-healing wound on right foot',
    'Pregnant woman with elevated blood pressure and swelling',
    'Child with rash, fever, and joint pain for one week',
]
medium_kn = [
    'ರೋಗಿಗೆ ಎಡ ತೋಳಿಗೆ ಹರಡುವ ಎದೆ ನೋವು ಇದೆ',
    'ಮೂರು ದಿನಗಳಿಂದ ಜ್ವರ ಕೆಮ್ಮು ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ',
    'ಮಧುಮೇಹ ರೋಗಿಗೆ ಬಲ ಕಾಲಿನಲ್ಲಿ ಗುಣವಾಗದ ಗಾಯ',
    'ಗರ್ಭಿಣಿ ಮಹಿಳೆಗೆ ಹೆಚ್ಚಿನ ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಊತ',
    'ಮಗುವಿಗೆ ದದ್ದು ಜ್ವರ ಮತ್ತು ಕೀಲು ನೋವು ಒಂದು ವಾರದಿಂದ',
]
medium_ta = [
    'நோயாளி இடது கையை நோக்கிப் பரவும் மார்பு வலியுடன் வருகிறார்',
    'மூன்று நாட்களாக காய்ச்சல் இருமல் மற்றும் மூச்சுத்திணறல்',
    'நீரிழிவு நோயாளிக்கு வலது காலில் ஆறாத காயம்',
    'கர்ப்பிணி பெண்ணுக்கு உயர்ந்த ரத்த அழுத்தம் மற்றும் வீக்கம்',
    'குழந்தைக்கு தடிப்பு காய்ச்சல் மற்றும் மூட்டு வலி ஒரு வாரமாக',
]

complex_en = [
    'Consider acute coronary syndrome given ST elevation in leads V1 through V4 with reciprocal changes',
    'Start empirical broad spectrum antibiotics pending blood culture sensitivity results',
    'Rule out pulmonary embolism with D-dimer and CT pulmonary angiography',
    'Differential diagnosis includes meningitis, encephalitis, and subarachnoid hemorrhage',
    'Titrate insulin drip to maintain blood glucose between 140 and 180 milligrams per deciliter',
]
complex_kn = [
    'V1 ನಿಂದ V4 ವರೆಗೆ ST ಏರಿಕೆಯೊಂದಿಗೆ ತೀವ್ರ ಹೃದ್ರೋಗಲಕ್ಷಣವನ್ನು ಪರಿಗಣಿಸಿ',
    'ರಕ್ತ ಕಳ್ಚರ್ ಫಲಿತಾಂಶ ಬರುವವರೆಗೆ ವಿಸ್ತೃತ ವ್ಯಾಪ್ತಿಯ ಪ್ರತಿಜೀವಕಗಳನ್ನು ಪ್ರಾರಂಭಿಸಿ',
    'D-ಡೈಮರ್ ಮತ್ತು CT ಶ್ವಾಸಕೋಶ ರಕ್ತನಾಳದ ಚಿತ್ರಣದೊಂದಿಗೆ ಶ್ವಾಸಕೋಶ ರಕ್ತಹೆಪ್ಪುಗಟ್ಟು ಹೊರಗಿಡಿ',
    'ವಿಭೇದನಾತ್ಮಕ ರೋಗನಿರ್ಣಯದಲ್ಲಿ ಮೆನಿಂಜೈಟಿಸ್ ಎನ್ಸೆಫಲೈಟಿಸ್ ಮತ್ತು ಸಬ್ಅರಾಕ್ನಾಯ್ಡ್ ರಕ್ತಸ್ರಾವ ಸೇರಿವೆ',
    'ರಕ್ತದ ಗ್ಲೂಕೋಸ್ 140 ರಿಂದ 180 ಮಿಲಿಗ್ರಾಂ ನಡುವೆ ಇರಿಸಲು ಇನ್ಸುಲಿನ್ ಡ್ರಿಪ್ ಹೊಂದಿಸಿ',
]
complex_ta = [
    'V1 முதல் V4 வரை ST உயர்வுடன் கடுமையான இதய நாள நோய்க்குறியை கருதுங்கள்',
    'ரத்த வளர்ப்பு உணர்வுத்தன்மை வரும் வரை பரந்த நோக்க நுண்ணுயிரி கொல்லிகளை தொடங்குங்கள்',
    'D-டைமர் மற்றும் CT நுரையீரல் குழாய் ரத்தக்குழாய் சித்தரப்படம் மூலம் நுரையீரல் குழாய் உறைப்பை விலக்குங்கள்',
    'வேறுபடுத்தல் நோய்நிர்ணயத்தில் மூளை உறை அழற்சி மூளையழற்சி மற்றும் சரக்கினாய்டு ரத்தப்போக்கு அடங்கும்',
    'ரத்த க்ளூக்கோசை 140 முதல் 180 மில்லிகிராம் நடுவே வைக்க இன்சுலின் டிரிப்பை சரிசெய்யுங்கள்',
]

all_sentences = {
    'SIMPLE': (simple_en, simple_kn, simple_ta),
    'MEDIUM': (medium_en, medium_kn, medium_ta),
    'COMPLEX': (complex_en, complex_kn, complex_ta),
}

models = {
    'MiniLM-384D': model_mini,
    'MPNet-768D': model_mpnet,
}

# Run both models
model_results = {}

for model_name, embed_model in models.items():
    print(f'\n{"#" * 70}')
    print(f'MODEL: {model_name}')
    print(f'{"#" * 70}')

    all_kn = []
    all_ta = []

    for level_name, (en, kn, ta) in all_sentences.items():
        print(f'\n  --- {level_name} ---')
        for i in range(5):
            e_en = embed_model.encode(en[i])
            e_kn = embed_model.encode(kn[i])
            e_ta = embed_model.encode(ta[i])

            efi_kn = 1 - cosine(e_kn, e_en)
            efi_ta = 1 - cosine(e_ta, e_en)
            sim_kt = 1 - cosine(e_kn, e_ta)

            all_kn.append(efi_kn)
            all_ta.append(efi_ta)

            print(f'    S{i+1}: EFI_Kn={efi_kn:+.4f}  EFI_Ta={efi_ta:+.4f}  Kn-Ta={sim_kt:.4f}')

    all_kn = np.array(all_kn)
    all_ta = np.array(all_ta)
    model_results[model_name] = {'kn': all_kn, 'ta': all_ta}

    print(f'\n  AGGREGATE:')
    print(f'    EFI_Kn = {np.mean(all_kn):.4f} +/- {np.std(all_kn):.4f}')
    print(f'    EFI_Ta = {np.mean(all_ta):.4f} +/- {np.std(all_ta):.4f}')

    t_kn, p_kn = stats.ttest_1samp(all_kn, 1.0)
    t_ta, p_ta = stats.ttest_1samp(all_ta, 1.0)
    print(f'    Kn vs 1.0: t={t_kn:.2f}, p={p_kn:.2e}')
    print(f'    Ta vs 1.0: t={t_ta:.2f}, p={p_ta:.2e}')

# Cross-model comparison
print(f'\n{"=" * 70}')
print(f'ROBUSTNESS COMPARISON')
print(f'{"=" * 70}')

mini_kn = model_results['MiniLM-384D']['kn']
mini_ta = model_results['MiniLM-384D']['ta']
mpnet_kn = model_results['MPNet-768D']['kn']
mpnet_ta = model_results['MPNet-768D']['ta']

print(f'                   MiniLM-384D    MPNet-768D')
print(f'EFI_Kannada:       {np.mean(mini_kn):.4f}          {np.mean(mpnet_kn):.4f}')
print(f'EFI_Tamil:         {np.mean(mini_ta):.4f}          {np.mean(mpnet_ta):.4f}')

# Correlation between models (do they rank sentences similarly?)
r_kn, p_kn = stats.pearsonr(mini_kn, mpnet_kn)
r_ta, p_ta = stats.pearsonr(mini_ta, mpnet_ta)
print(f'\nCross-model correlation (sentence-level):')
print(f'  Kannada EFI: r={r_kn:.4f}, p={p_kn:.4e}')
print(f'  Tamil EFI:   r={r_ta:.4f}, p={p_ta:.4e}')

# Does mpnet also show degradation?
t_mpnet_kn, p_mpnet_kn = stats.ttest_1samp(mpnet_kn, 1.0)
t_mpnet_ta, p_mpnet_ta = stats.ttest_1samp(mpnet_ta, 1.0)

print(f'\n{"=" * 70}')
print(f'ROBUSTNESS VERDICT')
print(f'{"=" * 70}')
if p_mpnet_kn < 0.001 and p_mpnet_ta < 0.001:
    print(f'CONFIRMED: EFI degradation replicates under MPNet-768D')
    print(f'  MPNet Kannada vs English: p={p_mpnet_kn:.2e}')
    print(f'  MPNet Tamil vs English:   p={p_mpnet_ta:.2e}')
    degradation_kn = (1 - np.mean(mpnet_kn)) * 100
    degradation_ta = (1 - np.mean(mpnet_ta)) * 100
    print(f'  Degradation: Kannada ~{degradation_kn:.0f}%, Tamil ~{degradation_ta:.0f}%')
    print(f'  => Finding is embedding-model-independent')
else:
    print(f'CAUTION: Results differ between embedding models')
    print(f'  MPNet Kannada: p={p_mpnet_kn:.2e}')
    print(f'  MPNet Tamil:   p={p_mpnet_ta:.2e}')
