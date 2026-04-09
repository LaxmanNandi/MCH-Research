"""
Paper 8 — EFI European Language Control Experiment
Tests whether low EFI is specific to Indian languages (tokenizer fragmentation)
or a general property of any non-English language.
If French/Spanish/German score high EFI while Kannada/Tamil/Hindi score low,
it proves the problem is tokenizer fragmentation, not "being a different language."
Design: 3 complexity x 5 sentences x 7 languages (En, Kn, Ta, Hi, Fr, Es, De)
"""
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats
import json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

model = SentenceTransformer('all-MiniLM-L6-v2')

print('=' * 70)
print('ENCODING FIDELITY — European vs Indian Language Control')
print('Model: all-MiniLM-L6-v2 (384D)')
print('Design: 3 complexity x 5 sentences x 7 languages')
print('Hypothesis: European languages (well-tokenized) should show higher EFI')
print('=' * 70)

# === ENGLISH (reference) ===
simple_en = [
    'The patient has fever',
    'She has a headache',
    'Blood pressure is high',
    'The child is vomiting',
    'He cannot breathe properly',
]
medium_en = [
    'Patient presents with chest pain radiating to left arm',
    'Three days of fever with productive cough and breathlessness',
    'Diabetic patient with non-healing wound on right foot',
    'Pregnant woman with elevated blood pressure and swelling',
    'Child with rash, fever, and joint pain for one week',
]
complex_en = [
    'Consider acute coronary syndrome given ST elevation in leads V1 through V4 with reciprocal changes',
    'Start empirical broad spectrum antibiotics pending blood culture sensitivity results',
    'Rule out pulmonary embolism with D-dimer and CT pulmonary angiography',
    'Differential diagnosis includes meningitis, encephalitis, and subarachnoid hemorrhage',
    'Titrate insulin drip to maintain blood glucose between 140 and 180 milligrams per deciliter',
]

# === KANNADA ===
simple_kn = [
    'ರೋಗಿಗೆ ಜ್ವರ ಇದೆ',
    'ಅವಳಿಗೆ ತಲೆನೋವು ಇದೆ',
    'ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ',
    'ಮಗುವಿಗೆ ವಾಂತಿ ಆಗುತ್ತಿದೆ',
    'ಅವನಿಗೆ ಉಸಿರಾಟ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ',
]
medium_kn = [
    'ರೋಗಿಗೆ ಎಡ ತೋಳಿಗೆ ಹರಡುವ ಎದೆ ನೋವು ಇದೆ',
    'ಮೂರು ದಿನಗಳಿಂದ ಜ್ವರ ಕೆಮ್ಮು ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ',
    'ಮಧುಮೇಹ ರೋಗಿಗೆ ಬಲ ಕಾಲಿನಲ್ಲಿ ಗುಣವಾಗದ ಗಾಯ',
    'ಗರ್ಭಿಣಿ ಮಹಿಳೆಗೆ ಹೆಚ್ಚಿನ ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಊತ',
    'ಮಗುವಿಗೆ ದದ್ದು ಜ್ವರ ಮತ್ತು ಕೀಲು ನೋವು ಒಂದು ವಾರದಿಂದ',
]
complex_kn = [
    'V1 ನಿಂದ V4 ವರೆಗೆ ST ಏರಿಕೆಯೊಂದಿಗೆ ತೀವ್ರ ಹೃದ್ರೋಗಲಕ್ಷಣವನ್ನು ಪರಿಗಣಿಸಿ',
    'ರಕ್ತ ಕಳ್ಚರ್ ಫಲಿತಾಂಶ ಬರುವವರೆಗೆ ವಿಸ್ತೃತ ವ್ಯಾಪ್ತಿಯ ಪ್ರತಿಜೀವಕಗಳನ್ನು ಪ್ರಾರಂಭಿಸಿ',
    'D-ಡೈಮರ್ ಮತ್ತು CT ಶ್ವಾಸಕೋಶ ರಕ್ತನಾಳದ ಚಿತ್ರಣದೊಂದಿಗೆ ಶ್ವಾಸಕೋಶ ರಕ್ತಹೆಪ್ಪುಗಟ್ಟು ಹೊರಗಿಡಿ',
    'ವಿಭೇದನಾತ್ಮಕ ರೋಗನಿರ್ಣಯದಲ್ಲಿ ಮೆನಿಂಜೈಟಿಸ್ ಎನ್ಸೆಫಲೈಟಿಸ್ ಮತ್ತು ಸಬ್ಅರಾಕ್ನಾಯ್ಡ್ ರಕ್ತಸ್ರಾವ ಸೇರಿವೆ',
    'ರಕ್ತದ ಗ್ಲೂಕೋಸ್ 140 ರಿಂದ 180 ಮಿಲಿಗ್ರಾಂ ನಡುವೆ ಇರಿಸಲು ಇನ್ಸುಲಿನ್ ಡ್ರಿಪ್ ಹೊಂದಿಸಿ',
]

# === TAMIL ===
simple_ta = [
    'நோயாளிக்கு காய்ச்சல் உள்ளது',
    'அவளுக்கு தலைவலி உள்ளது',
    'ரத்த அழுத்தம் அதிகமாக உள்ளது',
    'குழந்தைக்கு வாந்தி வருகிறது',
    'அவனால் சரியாக மூச்சு விட முடியவில்லை',
]
medium_ta = [
    'நோயாளி இடது கையை நோக்கிப் பரவும் மார்பு வலியுடன் வருகிறார்',
    'மூன்று நாட்களாக காய்ச்சல் இருமல் மற்றும் மூச்சுத்திணறல்',
    'நீரிழிவு நோயாளிக்கு வலது காலில் ஆறாத காயம்',
    'கர்ப்பிணி பெண்ணுக்கு உயர்ந்த ரத்த அழுத்தம் மற்றும் வீக்கம்',
    'குழந்தைக்கு தடிப்பு காய்ச்சல் மற்றும் மூட்டு வலி ஒரு வாரமாக',
]
complex_ta = [
    'V1 முதல் V4 வரை ST உயர்வுடன் கடுமையான இதய நாள நோய்க்குறியை கருதுங்கள்',
    'ரத்த வளர்ப்பு உணர்வுத்தன்மை வரும் வரை பரந்த நோக்க நுண்ணுயிரி கொல்லிகளை தொடங்குங்கள்',
    'D-டைமர் மற்றும் CT நுரையீரல் குழாய் ரத்தக்குழாய் சித்தரப்படம் மூலம் நுரையீரல் குழாய் உறைப்பை விலக்குங்கள்',
    'வேறுபடுத்தல் நோய்நிர்ணயத்தில் மூளை உறை அழற்சி மூளையழற்சி மற்றும் சரக்கினாய்டு ரத்தப்போக்கு அடங்கும்',
    'ரத்த க்ளூக்கோசை 140 முதல் 180 மில்லிகிராம் நடுவே வைக்க இன்சுலின் டிரிப்பை சரிசெய்யுங்கள்',
]

# === HINDI ===
simple_hi = [
    'मरीज को बुखार है',
    'उसे सिरदर्द है',
    'रक्तचाप अधिक है',
    'बच्चे को उल्टी हो रही है',
    'वह ठीक से सांस नहीं ले पा रहा है',
]
medium_hi = [
    'मरीज को बाएं हाथ तक फैलने वाला सीने का दर्द है',
    'तीन दिनों से बुखार खांसी और सांस फूलना',
    'मधुमेह के मरीज के दाएं पैर में न भरने वाला घाव',
    'गर्भवती महिला को उच्च रक्तचाप और सूजन',
    'बच्चे को एक हफ्ते से दाने बुखार और जोड़ों में दर्द',
]
complex_hi = [
    'V1 से V4 तक ST उन्नयन को देखते हुए तीव्र कोरोनरी सिंड्रोम पर विचार करें',
    'रक्त कल्चर संवेदनशीलता के परिणाम आने तक व्यापक स्पेक्ट्रम एंटीबायोटिक्स शुरू करें',
    'D-डाइमर और CT पल्मोनरी एंजियोग्राफी से पल्मोनरी एम्बोलिज्म को बाहर करें',
    'विभेदक निदान में मेनिनजाइटिस एन्सेफलाइटिस और सबएराक्नॉइड रक्तस्राव शामिल हैं',
    'रक्त ग्लूकोज को 140 से 180 मिलीग्राम प्रति डेसीलीटर के बीच रखने के लिए इंसुलिन ड्रिप समायोजित करें',
]

# === FRENCH (control — Latin script, high-resource) ===
simple_fr = [
    'Le patient a de la fièvre',
    'Elle a mal à la tête',
    'La tension artérielle est élevée',
    'L\'enfant vomit',
    'Il ne peut pas respirer correctement',
]
medium_fr = [
    'Le patient présente une douleur thoracique irradiant vers le bras gauche',
    'Trois jours de fièvre avec toux productive et essoufflement',
    'Patient diabétique avec une plaie non cicatrisante au pied droit',
    'Femme enceinte avec une tension artérielle élevée et un œdème',
    'Enfant avec éruption cutanée, fièvre et douleurs articulaires depuis une semaine',
]
complex_fr = [
    'Considérer un syndrome coronarien aigu devant un sus-décalage ST en V1 à V4 avec des modifications réciproques',
    'Débuter une antibiothérapie empirique à large spectre en attendant les résultats de l\'hémoculture',
    'Exclure une embolie pulmonaire par D-dimères et angioscanner pulmonaire',
    'Le diagnostic différentiel comprend la méningite, l\'encéphalite et l\'hémorragie sous-arachnoïdienne',
    'Ajuster la perfusion d\'insuline pour maintenir la glycémie entre 140 et 180 milligrammes par décilitre',
]

# === SPANISH (control — Latin script, high-resource) ===
simple_es = [
    'El paciente tiene fiebre',
    'Ella tiene dolor de cabeza',
    'La presión arterial está alta',
    'El niño está vomitando',
    'No puede respirar correctamente',
]
medium_es = [
    'El paciente presenta dolor torácico que irradia al brazo izquierdo',
    'Tres días de fiebre con tos productiva y dificultad respiratoria',
    'Paciente diabético con herida que no cicatriza en el pie derecho',
    'Mujer embarazada con presión arterial elevada y edema',
    'Niño con erupción cutánea, fiebre y dolor articular durante una semana',
]
complex_es = [
    'Considerar síndrome coronario agudo dado elevación del ST en derivaciones V1 a V4 con cambios recíprocos',
    'Iniciar antibióticos empíricos de amplio espectro pendiente resultados del hemocultivo',
    'Descartar embolia pulmonar con dímero D y angiotomografía pulmonar',
    'El diagnóstico diferencial incluye meningitis, encefalitis y hemorragia subaracnoidea',
    'Ajustar la infusión de insulina para mantener la glucemia entre 140 y 180 miligramos por decilitro',
]

# === GERMAN (control — Latin script, compound words) ===
simple_de = [
    'Der Patient hat Fieber',
    'Sie hat Kopfschmerzen',
    'Der Blutdruck ist hoch',
    'Das Kind erbricht sich',
    'Er kann nicht richtig atmen',
]
medium_de = [
    'Patient stellt sich mit Brustschmerzen vor, die in den linken Arm ausstrahlen',
    'Drei Tage Fieber mit produktivem Husten und Atemnot',
    'Diabetischer Patient mit nicht heilender Wunde am rechten Fuß',
    'Schwangere Frau mit erhöhtem Blutdruck und Schwellung',
    'Kind mit Hautausschlag, Fieber und Gelenkschmerzen seit einer Woche',
]
complex_de = [
    'Akutes Koronarsyndrom in Betracht ziehen bei ST-Hebung in Ableitungen V1 bis V4 mit reziproken Veränderungen',
    'Empirische Breitspektrum-Antibiotika beginnen bis Blutkultur-Empfindlichkeitsergebnisse vorliegen',
    'Lungenembolie ausschließen mittels D-Dimer und CT-Pulmonalisangiographie',
    'Differentialdiagnose umfasst Meningitis, Enzephalitis und Subarachnoidalblutung',
    'Insulintropf titrieren um Blutzucker zwischen 140 und 180 Milligramm pro Deziliter zu halten',
]

# === COMPUTE EFI FOR ALL LANGUAGES ===
languages = {
    'Kannada': (simple_kn, medium_kn, complex_kn),
    'Tamil': (simple_ta, medium_ta, complex_ta),
    'Hindi': (simple_hi, medium_hi, complex_hi),
    'French': (simple_fr, medium_fr, complex_fr),
    'Spanish': (simple_es, medium_es, complex_es),
    'German': (simple_de, medium_de, complex_de),
}

english = (simple_en, medium_en, complex_en)
level_names = ['SIMPLE', 'MEDIUM', 'COMPLEX']

results = {}
all_embeddings = {}  # Store all embeddings
all_sentences = {}   # Store all sentences

# Store English sentences and embeddings
all_sentences['English'] = {'SIMPLE': simple_en, 'MEDIUM': medium_en, 'COMPLEX': complex_en}
all_embeddings['English'] = {}
for level_idx, level in enumerate(level_names):
    en_sents = english[level_idx]
    all_embeddings['English'][level] = [model.encode(s).tolist() for s in en_sents]

for lang_name, lang_data in languages.items():
    lang_efis = {'all': [], 'by_level': {}}
    all_sentences[lang_name] = {}
    all_embeddings[lang_name] = {}
    print(f'\n{"=" * 60}')
    print(f'LANGUAGE: {lang_name}')
    print(f'{"=" * 60}')

    for level_idx, level in enumerate(level_names):
        en_sents = english[level_idx]
        lang_sents = lang_data[level_idx]
        level_efis = []
        level_embs = []

        all_sentences[lang_name][level] = lang_sents

        print(f'\n  {level}:')
        for i in range(5):
            e_en = model.encode(en_sents[i])
            e_lang = model.encode(lang_sents[i])
            efi = float(1 - cosine(e_lang, e_en))
            level_efis.append(efi)
            level_embs.append(e_lang.tolist())
            print(f'    S{i+1}: EFI = {efi:.4f}')

        mean_efi = np.mean(level_efis)
        print(f'    Mean: {mean_efi:.4f}')
        lang_efis['by_level'][level] = level_efis
        lang_efis['all'].extend(level_efis)
        all_embeddings[lang_name][level] = level_embs

    overall = np.mean(lang_efis['all'])
    std = np.std(lang_efis['all'])
    results[lang_name] = {
        'mean_efi': float(overall),
        'std_efi': float(std),
        'by_level': {level: float(np.mean(lang_efis['by_level'][level])) for level in level_names},
        'all_values': [float(x) for x in lang_efis['all']],
    }
    print(f'\n  >>> {lang_name} OVERALL EFI: {overall:.4f} ± {std:.4f}')

# === SUMMARY ===
print('\n' + '=' * 70)
print('SUMMARY — EFI by Language')
print('=' * 70)
print(f'{"Language":<12} {"Overall EFI":>12} {"Simple":>10} {"Medium":>10} {"Complex":>10}')
print('-' * 54)
for lang in ['French', 'Spanish', 'German', 'Kannada', 'Tamil', 'Hindi']:
    r = results[lang]
    print(f'{lang:<12} {r["mean_efi"]:>12.4f} {r["by_level"]["SIMPLE"]:>10.4f} {r["by_level"]["MEDIUM"]:>10.4f} {r["by_level"]["COMPLEX"]:>10.4f}')

# === STATISTICAL TESTS ===
print('\n' + '=' * 70)
print('STATISTICAL TESTS')
print('=' * 70)

european = results['French']['all_values'] + results['Spanish']['all_values'] + results['German']['all_values']
indian = results['Kannada']['all_values'] + results['Tamil']['all_values'] + results['Hindi']['all_values']

t_stat, p_val = stats.ttest_ind(european, indian)
u_stat, u_p = stats.mannwhitneyu(european, indian, alternative='two-sided')
cohens_d = (np.mean(european) - np.mean(indian)) / np.sqrt((np.std(european)**2 + np.std(indian)**2) / 2)

print(f'European (N={len(european)}): mean={np.mean(european):.4f} ± {np.std(european):.4f}')
print(f'Indian (N={len(indian)}):   mean={np.mean(indian):.4f} ± {np.std(indian):.4f}')
print(f't-test: t={t_stat:.4f}, p={p_val:.2e}')
print(f'Mann-Whitney: U={u_stat:.1f}, p={u_p:.2e}')
print(f"Cohen's d: {cohens_d:.4f}")

# === SAVE RESULTS ===
output = {
    'experiment': 'Paper 8 — European Language Control for EFI',
    'timestamp': datetime.now().isoformat(),
    'embedding_model': 'all-MiniLM-L6-v2 (384D)',
    'design': '3 complexity x 5 sentences x 7 languages',
    'results': results,
    'group_comparison': {
        'european_mean': float(np.mean(european)),
        'european_std': float(np.std(european)),
        'indian_mean': float(np.mean(indian)),
        'indian_std': float(np.std(indian)),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'mann_whitney_U': float(u_stat),
        'mann_whitney_p': float(u_p),
        'cohens_d': float(cohens_d),
    }
}

out_path = 'papers/paper8_efi/results/exp_european_control.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nResults saved to {out_path}')

# Save embeddings separately (large file)
emb_output = {
    'experiment': 'Paper 8 — European Control Embeddings',
    'embedding_model': 'all-MiniLM-L6-v2 (384D)',
    'timestamp': datetime.now().isoformat(),
    'sentences': all_sentences,
    'embeddings': all_embeddings,
}
emb_path = 'papers/paper8_efi/results/exp_european_control_embeddings.json'
with open(emb_path, 'w', encoding='utf-8') as f:
    json.dump(emb_output, f, indent=2, ensure_ascii=False)
print(f'Embeddings saved to {emb_path}')
