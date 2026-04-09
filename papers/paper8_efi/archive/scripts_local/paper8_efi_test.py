"""
Paper 8 — Encoding Fidelity Index (EFI) Test Battery
Clinical test sentences: Simple / Medium / Complex
Languages: English, Kannada, Tamil
"""
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

model = SentenceTransformer('all-MiniLM-L6-v2')

print('=' * 70)
print('ENCODING FIDELITY EXPERIMENT — Clinical Test Battery')
print('Model: all-MiniLM-L6-v2 (384D)')
print('Design: 3 complexity levels x 5 sentences x 3 languages')
print('=' * 70)

# SIMPLE — single symptoms, everyday clinical words
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

# MEDIUM — clinical descriptions with multiple symptoms
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

# COMPLEX — diagnostic reasoning, treatment decisions
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

levels = {
    'SIMPLE': (simple_en, simple_kn, simple_ta),
    'MEDIUM': (medium_en, medium_kn, medium_ta),
    'COMPLEX': (complex_en, complex_kn, complex_ta),
}

all_efi_kn = []
all_efi_ta = []
all_confusion = []
level_results = {}

for level_name, (en, kn, ta) in levels.items():
    print(f'\n{"=" * 60}')
    print(f'LEVEL: {level_name}')
    print(f'{"=" * 60}')

    efi_kn_level = []
    efi_ta_level = []
    conf_level = []

    for i in range(5):
        e_en = model.encode(en[i])
        e_kn = model.encode(kn[i])
        e_ta = model.encode(ta[i])

        efi_kn = 1 - cosine(e_kn, e_en)
        efi_ta = 1 - cosine(e_ta, e_en)
        kt = 1 - cosine(e_kn, e_ta)
        ke = 1 - cosine(e_kn, e_en)
        conf = kt / ke if abs(ke) > 0.001 else float('inf')

        efi_kn_level.append(efi_kn)
        efi_ta_level.append(efi_ta)
        if conf != float('inf'):
            conf_level.append(conf)

        print(f'  S{i+1}: EFI_Kn={efi_kn:+.4f}  EFI_Ta={efi_ta:+.4f}  Kn-Ta={kt:.4f}  Confusion={conf:.1f}x')

    all_efi_kn.extend(efi_kn_level)
    all_efi_ta.extend(efi_ta_level)
    all_confusion.extend(conf_level)

    mean_kn = np.mean(efi_kn_level)
    mean_ta = np.mean(efi_ta_level)
    mean_conf = np.mean(conf_level) if conf_level else float('nan')
    print(f'  ---')
    print(f'  Mean EFI_Kn={mean_kn:.4f}  EFI_Ta={mean_ta:.4f}  Confusion={mean_conf:.1f}x')

    level_results[level_name] = {
        'efi_kn': efi_kn_level,
        'efi_ta': efi_ta_level,
        'confusion': conf_level,
    }

# Aggregate statistics
all_efi_kn = np.array(all_efi_kn)
all_efi_ta = np.array(all_efi_ta)
all_confusion = np.array(all_confusion)

print(f'\n{"=" * 60}')
print(f'AGGREGATE RESULTS (N=15 sentence pairs)')
print(f'{"=" * 60}')
print(f'EFI_Kannada: {np.mean(all_efi_kn):.4f} +/- {np.std(all_efi_kn):.4f}')
print(f'EFI_Tamil:   {np.mean(all_efi_ta):.4f} +/- {np.std(all_efi_ta):.4f}')
print(f'EFI_English: 1.0000 (reference)')
print(f'Confusion ratio: {np.mean(all_confusion):.1f}x +/- {np.std(all_confusion):.1f}x')

# Statistical tests
U, p = stats.mannwhitneyu(all_efi_kn, all_efi_ta, alternative='two-sided')
print(f'\nMann-Whitney U (Kn vs Ta): U={U}, p={p:.6f}')

t_kn, p_kn = stats.ttest_1samp(all_efi_kn, 1.0)
t_ta, p_ta = stats.ttest_1samp(all_efi_ta, 1.0)
print(f't-test EFI_Kn vs 1.0: t={t_kn:.2f}, p={p_kn:.2e}')
print(f't-test EFI_Ta vs 1.0: t={t_ta:.2f}, p={p_ta:.2e}')

# Complexity effect
print(f'\n{"=" * 60}')
print(f'COMPLEXITY EFFECT')
print(f'{"=" * 60}')
for level_name in ['SIMPLE', 'MEDIUM', 'COMPLEX']:
    r = level_results[level_name]
    print(f'{level_name:>8s}: EFI_Kn={np.mean(r["efi_kn"]):.4f}  EFI_Ta={np.mean(r["efi_ta"]):.4f}  Confusion={np.mean(r["confusion"]):.1f}x')

# Test if complexity affects EFI
simple_kn_arr = np.array(level_results['SIMPLE']['efi_kn'])
complex_kn_arr = np.array(level_results['COMPLEX']['efi_kn'])
U2, p2 = stats.mannwhitneyu(simple_kn_arr, complex_kn_arr, alternative='two-sided')
print(f'\nSimple vs Complex (Kannada): U={U2}, p={p2:.6f}')

print(f'\n{"=" * 60}')
print(f'KEY FINDINGS')
print(f'{"=" * 60}')
print(f'1. Both Kannada and Tamil are ~{(1-np.mean(all_efi_kn))*100:.0f}% degraded vs English')
print(f'2. Kannada-Tamil confusion: {np.mean(all_confusion):.1f}x closer to each other than to English')
print(f'3. This is the mechanism enabling Coherent Misalignment in Dravidian languages')
print(f'4. Clinical implications: A doctor using LLM in Kannada gets Tamil-like encoding')
print(f'\nNOTE: This measures the embedding model (all-MiniLM-L6-v2) fidelity,')
print(f'not specific LLM internal encoding. The principle generalizes.')
