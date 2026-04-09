"""
Paper 8 — Encoding Fidelity Index: Hindi Comparison
Adds Hindi (Indo-Aryan) to test whether EFI degradation is Dravidian-specific
or applies to all Indian languages.
Design: 3 complexity x 5 sentences x 4 languages (En, Kn, Ta, Hi)
"""
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

model = SentenceTransformer('all-MiniLM-L6-v2')

print('=' * 70)
print('ENCODING FIDELITY — Hindi vs Dravidian Comparison')
print('Model: all-MiniLM-L6-v2 (384D)')
print('Design: 3 complexity x 5 sentences x 4 languages')
print('Hypothesis: Hindi (Indo-Aryan) may have higher EFI than Dravidian')
print('=' * 70)

# SIMPLE
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
simple_hi = [
    'मरीज को बुखार है',
    'उसे सिरदर्द है',
    'रक्तचाप अधिक है',
    'बच्चे को उल्टी हो रही है',
    'वह ठीक से सांस नहीं ले पा रहा है',
]

# MEDIUM
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
medium_hi = [
    'मरीज को बाएं हाथ तक फैलने वाला सीने का दर्द है',
    'तीन दिनों से बुखार खांसी और सांस फूलना',
    'मधुमेह के मरीज के दाएं पैर में न भरने वाला घाव',
    'गर्भवती महिला को उच्च रक्तचाप और सूजन',
    'बच्चे को एक हफ्ते से दाने बुखार और जोड़ों में दर्द',
]

# COMPLEX
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
complex_hi = [
    'V1 से V4 तक ST उन्नयन को देखते हुए तीव्र कोरोनरी सिंड्रोम पर विचार करें',
    'रक्त कल्चर संवेदनशीलता के परिणाम आने तक व्यापक स्पेक्ट्रम एंटीबायोटिक्स शुरू करें',
    'D-डाइमर और CT पल्मोनरी एंजियोग्राफी से पल्मोनरी एम्बोलिज्म को बाहर करें',
    'विभेदक निदान में मेनिनजाइटिस एन्सेफलाइटिस और सबएराक्नॉइड रक्तस्राव शामिल हैं',
    'रक्त ग्लूकोज को 140 से 180 मिलीग्राम प्रति डेसीलीटर के बीच रखने के लिए इंसुलिन ड्रिप समायोजित करें',
]

levels = {
    'SIMPLE': (simple_en, simple_kn, simple_ta, simple_hi),
    'MEDIUM': (medium_en, medium_kn, medium_ta, medium_hi),
    'COMPLEX': (complex_en, complex_kn, complex_ta, complex_hi),
}

all_efi = {'kn': [], 'ta': [], 'hi': []}
all_conf_kn_ta = []
all_conf_kn_hi = []
all_conf_ta_hi = []
level_results = {}

for level_name, (en, kn, ta, hi) in levels.items():
    print(f'\n{"=" * 60}')
    print(f'LEVEL: {level_name}')
    print(f'{"=" * 60}')

    efi_level = {'kn': [], 'ta': [], 'hi': []}

    for i in range(5):
        e_en = model.encode(en[i])
        e_kn = model.encode(kn[i])
        e_ta = model.encode(ta[i])
        e_hi = model.encode(hi[i])

        efi_kn = 1 - cosine(e_kn, e_en)
        efi_ta = 1 - cosine(e_ta, e_en)
        efi_hi = 1 - cosine(e_hi, e_en)

        # Cross-language similarities
        sim_kn_ta = 1 - cosine(e_kn, e_ta)
        sim_kn_hi = 1 - cosine(e_kn, e_hi)
        sim_ta_hi = 1 - cosine(e_ta, e_hi)

        efi_level['kn'].append(efi_kn)
        efi_level['ta'].append(efi_ta)
        efi_level['hi'].append(efi_hi)

        print(f'  S{i+1}: EFI_Kn={efi_kn:+.4f}  EFI_Ta={efi_ta:+.4f}  EFI_Hi={efi_hi:+.4f}  |  Kn-Ta={sim_kn_ta:.4f}  Kn-Hi={sim_kn_hi:.4f}  Ta-Hi={sim_ta_hi:.4f}')

    for lang in ['kn', 'ta', 'hi']:
        all_efi[lang].extend(efi_level[lang])

    means = {lang: np.mean(efi_level[lang]) for lang in ['kn', 'ta', 'hi']}
    print(f'  ---')
    print(f'  Mean EFI_Kn={means["kn"]:.4f}  EFI_Ta={means["ta"]:.4f}  EFI_Hi={means["hi"]:.4f}')

    level_results[level_name] = efi_level

# Convert to arrays
for lang in ['kn', 'ta', 'hi']:
    all_efi[lang] = np.array(all_efi[lang])

print(f'\n{"=" * 60}')
print(f'AGGREGATE RESULTS (N=15 sentence pairs per language)')
print(f'{"=" * 60}')
print(f'EFI_Kannada: {np.mean(all_efi["kn"]):.4f} +/- {np.std(all_efi["kn"]):.4f}')
print(f'EFI_Tamil:   {np.mean(all_efi["ta"]):.4f} +/- {np.std(all_efi["ta"]):.4f}')
print(f'EFI_Hindi:   {np.mean(all_efi["hi"]):.4f} +/- {np.std(all_efi["hi"]):.4f}')
print(f'EFI_English: 1.0000 (reference)')

# Statistical tests
print(f'\n--- PAIRWISE COMPARISONS ---')

# Hindi vs Kannada
U1, p1 = stats.mannwhitneyu(all_efi['hi'], all_efi['kn'], alternative='two-sided')
print(f'Hindi vs Kannada: U={U1}, p={p1:.6f}')

# Hindi vs Tamil
U2, p2 = stats.mannwhitneyu(all_efi['hi'], all_efi['ta'], alternative='two-sided')
print(f'Hindi vs Tamil:   U={U2}, p={p2:.6f}')

# Kannada vs Tamil
U3, p3 = stats.mannwhitneyu(all_efi['kn'], all_efi['ta'], alternative='two-sided')
print(f'Kannada vs Tamil: U={U3}, p={p3:.6f}')

# All vs English (1.0)
for lang, name in [('kn', 'Kannada'), ('ta', 'Tamil'), ('hi', 'Hindi')]:
    t, p = stats.ttest_1samp(all_efi[lang], 1.0)
    print(f'{name} vs English: t={t:.2f}, p={p:.2e}')

# Language family grouping
dravidian = np.concatenate([all_efi['kn'], all_efi['ta']])
indoaryan = all_efi['hi']
U4, p4 = stats.mannwhitneyu(dravidian, indoaryan, alternative='two-sided')
print(f'\n--- LANGUAGE FAMILY COMPARISON ---')
print(f'Dravidian (Kn+Ta, N=30): {np.mean(dravidian):.4f} +/- {np.std(dravidian):.4f}')
print(f'Indo-Aryan (Hi, N=15):   {np.mean(indoaryan):.4f} +/- {np.std(indoaryan):.4f}')
print(f'Mann-Whitney U: U={U4}, p={p4:.6f}')

# Complexity breakdown
print(f'\n{"=" * 60}')
print(f'COMPLEXITY EFFECT')
print(f'{"=" * 60}')
for level_name in ['SIMPLE', 'MEDIUM', 'COMPLEX']:
    r = level_results[level_name]
    print(f'{level_name:>8s}: EFI_Kn={np.mean(r["kn"]):.4f}  EFI_Ta={np.mean(r["ta"]):.4f}  EFI_Hi={np.mean(r["hi"]):.4f}')

# Key question: Does Hindi show the same Dravidian confusion pattern?
print(f'\n{"=" * 60}')
print(f'KEY QUESTION: Is degradation Dravidian-specific or universal?')
print(f'{"=" * 60}')
if p4 < 0.05:
    print(f'RESULT: Hindi EFI significantly DIFFERENT from Dravidian (p={p4:.4f})')
    if np.mean(indoaryan) > np.mean(dravidian):
        print(f'Hindi has HIGHER fidelity than Dravidian languages')
        print(f'=> Degradation has a language-family component')
    else:
        print(f'Hindi has LOWER fidelity than Dravidian languages')
else:
    print(f'RESULT: Hindi EFI NOT significantly different from Dravidian (p={p4:.4f})')
    print(f'=> Degradation is universal for Indian languages, not Dravidian-specific')
    print(f'=> All non-English languages suffer similar encoding fidelity loss')
