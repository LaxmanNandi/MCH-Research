"""
Paper 8 — MPNet Robustness Check for European Control
Verifies that European vs Indian EFI gap holds across embedding models.
"""
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy import stats
import json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')

# === SENTENCES (same as main experiment) ===
simple_en = ['The patient has fever','She has a headache','Blood pressure is high','The child is vomiting','He cannot breathe properly']
medium_en = ['Patient presents with chest pain radiating to left arm','Three days of fever with productive cough and breathlessness','Diabetic patient with non-healing wound on right foot','Pregnant woman with elevated blood pressure and swelling','Child with rash, fever, and joint pain for one week']
complex_en = ['Consider acute coronary syndrome given ST elevation in leads V1 through V4 with reciprocal changes','Start empirical broad spectrum antibiotics pending blood culture sensitivity results','Rule out pulmonary embolism with D-dimer and CT pulmonary angiography','Differential diagnosis includes meningitis, encephalitis, and subarachnoid hemorrhage','Titrate insulin drip to maintain blood glucose between 140 and 180 milligrams per deciliter']

english = (simple_en, medium_en, complex_en)

langs = {
    'Kannada': (
        ['ರೋಗಿಗೆ ಜ್ವರ ಇದೆ','ಅವಳಿಗೆ ತಲೆನೋವು ಇದೆ','ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ','ಮಗುವಿಗೆ ವಾಂತಿ ಆಗುತ್ತಿದೆ','ಅವನಿಗೆ ಉಸಿರಾಟ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ'],
        ['ರೋಗಿಗೆ ಎಡ ತೋಳಿಗೆ ಹರಡುವ ಎದೆ ನೋವು ಇದೆ','ಮೂರು ದಿನಗಳಿಂದ ಜ್ವರ ಕೆಮ್ಮು ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ','ಮಧುಮೇಹ ರೋಗಿಗೆ ಬಲ ಕಾಲಿನಲ್ಲಿ ಗುಣವಾಗದ ಗಾಯ','ಗರ್ಭಿಣಿ ಮಹಿಳೆಗೆ ಹೆಚ್ಚಿನ ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಊತ','ಮಗುವಿಗೆ ದದ್ದು ಜ್ವರ ಮತ್ತು ಕೀಲು ನೋವು ಒಂದು ವಾರದಿಂದ'],
        ['V1 ನಿಂದ V4 ವರೆಗೆ ST ಏರಿಕೆಯೊಂದಿಗೆ ತೀವ್ರ ಹೃದ್ರೋಗಲಕ್ಷಣವನ್ನು ಪರಿಗಣಿಸಿ','ರಕ್ತ ಕಳ್ಚರ್ ಫಲಿತಾಂಶ ಬರುವವರೆಗೆ ವಿಸ್ತೃತ ವ್ಯಾಪ್ತಿಯ ಪ್ರತಿಜೀವಕಗಳನ್ನು ಪ್ರಾರಂಭಿಸಿ','D-ಡೈಮರ್ ಮತ್ತು CT ಶ್ವಾಸಕೋಶ ರಕ್ತನಾಳದ ಚಿತ್ರಣದೊಂದಿಗೆ ಶ್ವಾಸಕೋಶ ರಕ್ತಹೆಪ್ಪುಗಟ್ಟು ಹೊರಗಿಡಿ','ವಿಭೇದನಾತ್ಮಕ ರೋಗನಿರ್ಣಯದಲ್ಲಿ ಮೆನಿಂಜೈಟಿಸ್ ಎನ್ಸೆಫಲೈಟಿಸ್ ಮತ್ತು ಸಬ್ಅರಾಕ್ನಾಯ್ಡ್ ರಕ್ತಸ್ರಾವ ಸೇರಿವೆ','ರಕ್ತದ ಗ್ಲೂಕೋಸ್ 140 ರಿಂದ 180 ಮಿಲಿಗ್ರಾಂ ನಡುವೆ ಇರಿಸಲು ಇನ್ಸುಲಿನ್ ಡ್ರಿಪ್ ಹೊಂದಿಸಿ'],
    ),
    'Tamil': (
        ['நோயாளிக்கு காய்ச்சல் உள்ளது','அவளுக்கு தலைவலி உள்ளது','ரத்த அழுத்தம் அதிகமாக உள்ளது','குழந்தைக்கு வாந்தி வருகிறது','அவனால் சரியாக மூச்சு விட முடியவில்லை'],
        ['நோயாளி இடது கையை நோக்கிப் பரவும் மார்பு வலியுடன் வருகிறார்','மூன்று நாட்களாக காய்ச்சல் இருமல் மற்றும் மூச்சுத்திணறல்','நீரிழிவு நோயாளிக்கு வலது காலில் ஆறாத காயம்','கர்ப்பிணி பெண்ணுக்கு உயர்ந்த ரத்த அழுத்தம் மற்றும் வீக்கம்','குழந்தைக்கு தடிப்பு காய்ச்சல் மற்றும் மூட்டு வலி ஒரு வாரமாக'],
        ['V1 முதல் V4 வரை ST உயர்வுடன் கடுமையான இதய நாள நோய்க்குறியை கருதுங்கள்','ரத்த வளர்ப்பு உணர்வுத்தன்மை வரும் வரை பரந்த நோக்க நுண்ணுயிரி கொல்லிகளை தொடங்குங்கள்','D-டைமர் மற்றும் CT நுரையீரல் குழாய் ரத்தக்குழாய் சித்தரப்படம் மூலம் நுரையீரல் குழாய் உறைப்பை விலக்குங்கள்','வேறுபடுத்தல் நோய்நிர்ணயத்தில் மூளை உறை அழற்சி மூளையழற்சி மற்றும் சரக்கினாய்டு ரத்தப்போக்கு அடங்கும்','ரத்த க்ளூக்கோசை 140 முதல் 180 மில்லிகிராம் நடுவே வைக்க இன்சுலின் டிரிப்பை சரிசெய்யுங்கள்'],
    ),
    'Hindi': (
        ['मरीज को बुखार है','उसे सिरदर्द है','रक्तचाप अधिक है','बच्चे को उल्टी हो रही है','वह ठीक से सांस नहीं ले पा रहा है'],
        ['मरीज को बाएं हाथ तक फैलने वाला सीने का दर्द है','तीन दिनों से बुखार खांसी और सांस फूलना','मधुमेह के मरीज के दाएं पैर में न भरने वाला घाव','गर्भवती महिला को उच्च रक्तचाप और सूजन','बच्चे को एक हफ्ते से दाने बुखार और जोड़ों में दर्द'],
        ['V1 से V4 तक ST उन्नयन को देखते हुए तीव्र कोरोनरी सिंड्रोम पर विचार करें','रक्त कल्चर संवेदनशीलता के परिणाम आने तक व्यापक स्पेक्ट्रम एंटीबायोटिक्स शुरू करें','D-डाइमर और CT पल्मोनरी एंजियोग्राफी से पल्मोनरी एम्बोलिज्म को बाहर करें','विभेदक निदान में मेनिनजाइटिस एन्सेफलाइटिस और सबएराक्नॉइड रक्तस्राव शामिल हैं','रक्त ग्लूकोज को 140 से 180 मिलीग्राम प्रति डेसीलीटर के बीच रखने के लिए इंसुलिन ड्रिप समायोजित करें'],
    ),
    'French': (
        ['Le patient a de la fièvre','Elle a mal à la tête','La tension artérielle est élevée',"L'enfant vomit",'Il ne peut pas respirer correctement'],
        ['Le patient présente une douleur thoracique irradiant vers le bras gauche','Trois jours de fièvre avec toux productive et essoufflement','Patient diabétique avec une plaie non cicatrisante au pied droit','Femme enceinte avec une tension artérielle élevée et un œdème','Enfant avec éruption cutanée, fièvre et douleurs articulaires depuis une semaine'],
        ['Considérer un syndrome coronarien aigu devant un sus-décalage ST en V1 à V4 avec des modifications réciproques',"Débuter une antibiothérapie empirique à large spectre en attendant les résultats de l'hémoculture",'Exclure une embolie pulmonaire par D-dimères et angioscanner pulmonaire',"Le diagnostic différentiel comprend la méningite, l'encéphalite et l'hémorragie sous-arachnoïdienne","Ajuster la perfusion d'insuline pour maintenir la glycémie entre 140 et 180 milligrammes par décilitre"],
    ),
    'Spanish': (
        ['El paciente tiene fiebre','Ella tiene dolor de cabeza','La presión arterial está alta','El niño está vomitando','No puede respirar correctamente'],
        ['El paciente presenta dolor torácico que irradia al brazo izquierdo','Tres días de fiebre con tos productiva y dificultad respiratoria','Paciente diabético con herida que no cicatriza en el pie derecho','Mujer embarazada con presión arterial elevada y edema','Niño con erupción cutánea, fiebre y dolor articular durante una semana'],
        ['Considerar síndrome coronario agudo dado elevación del ST en derivaciones V1 a V4 con cambios recíprocos','Iniciar antibióticos empíricos de amplio espectro pendiente resultados del hemocultivo','Descartar embolia pulmonar con dímero D y angiotomografía pulmonar','El diagnóstico diferencial incluye meningitis, encefalitis y hemorragia subaracnoidea','Ajustar la infusión de insulina para mantener la glucemia entre 140 y 180 miligramos por decilitro'],
    ),
    'German': (
        ['Der Patient hat Fieber','Sie hat Kopfschmerzen','Der Blutdruck ist hoch','Das Kind erbricht sich','Er kann nicht richtig atmen'],
        ['Patient stellt sich mit Brustschmerzen vor, die in den linken Arm ausstrahlen','Drei Tage Fieber mit produktivem Husten und Atemnot','Diabetischer Patient mit nicht heilender Wunde am rechten Fuß','Schwangere Frau mit erhöhtem Blutdruck und Schwellung','Kind mit Hautausschlag, Fieber und Gelenkschmerzen seit einer Woche'],
        ['Akutes Koronarsyndrom in Betracht ziehen bei ST-Hebung in Ableitungen V1 bis V4 mit reziproken Veränderungen','Empirische Breitspektrum-Antibiotika beginnen bis Blutkultur-Empfindlichkeitsergebnisse vorliegen','Lungenembolie ausschließen mittels D-Dimer und CT-Pulmonalisangiographie','Differentialdiagnose umfasst Meningitis, Enzephalitis und Subarachnoidalblutung','Insulintropf titrieren um Blutzucker zwischen 140 und 180 Milligramm pro Deziliter zu halten'],
    ),
}

level_names = ['SIMPLE', 'MEDIUM', 'COMPLEX']

print('\n' + '='*70)
print('EFI: MiniLM (384D) vs MPNet (768D)')
print('='*70)

results = {}
all_embs = {}

for lang_name, lang_data in langs.items():
    efis_minilm = []
    efis_mpnet = []
    all_embs[lang_name] = {'minilm': [], 'mpnet': []}

    for level_idx in range(3):
        for i in range(5):
            en_s = english[level_idx][i]
            lang_s = lang_data[level_idx][i]

            e_en_ml = minilm.encode(en_s)
            e_lang_ml = minilm.encode(lang_s)
            e_en_mp = mpnet.encode(en_s)
            e_lang_mp = mpnet.encode(lang_s)

            efis_minilm.append(float(1 - cosine(e_lang_ml, e_en_ml)))
            efis_mpnet.append(float(1 - cosine(e_lang_mp, e_en_mp)))

            all_embs[lang_name]['minilm'].append(e_lang_ml.tolist())
            all_embs[lang_name]['mpnet'].append(e_lang_mp.tolist())

    r_val, p_val = stats.pearsonr(efis_minilm, efis_mpnet)
    results[lang_name] = {
        'minilm_mean': float(np.mean(efis_minilm)),
        'minilm_std': float(np.std(efis_minilm)),
        'mpnet_mean': float(np.mean(efis_mpnet)),
        'mpnet_std': float(np.std(efis_mpnet)),
        'correlation_r': float(r_val),
        'correlation_p': float(p_val),
        'minilm_values': efis_minilm,
        'mpnet_values': efis_mpnet,
    }
    print(f'{lang_name:<10}: MiniLM={np.mean(efis_minilm):.4f}±{np.std(efis_minilm):.4f}  MPNet={np.mean(efis_mpnet):.4f}±{np.std(efis_mpnet):.4f}  r={r_val:.4f} (p={p_val:.2e})')

# Group comparison
euro_ml = results['French']['minilm_values'] + results['Spanish']['minilm_values'] + results['German']['minilm_values']
euro_mp = results['French']['mpnet_values'] + results['Spanish']['mpnet_values'] + results['German']['mpnet_values']
indian_ml = results['Kannada']['minilm_values'] + results['Tamil']['minilm_values'] + results['Hindi']['minilm_values']
indian_mp = results['Kannada']['mpnet_values'] + results['Tamil']['mpnet_values'] + results['Hindi']['mpnet_values']

print('\n' + '='*70)
print('GROUP COMPARISON')
print('='*70)

for name, euro, indian in [('MiniLM', euro_ml, indian_ml), ('MPNet', euro_mp, indian_mp)]:
    t_s, t_p = stats.ttest_ind(euro, indian)
    u_s, u_p = stats.mannwhitneyu(euro, indian, alternative='two-sided')
    cd = (np.mean(euro) - np.mean(indian)) / np.sqrt((np.std(euro)**2 + np.std(indian)**2) / 2)
    print(f'\n{name}:')
    print(f'  European: {np.mean(euro):.4f} ± {np.std(euro):.4f}')
    print(f'  Indian:   {np.mean(indian):.4f} ± {np.std(indian):.4f}')
    print(f'  t={t_s:.4f}, p={t_p:.2e}, Cohen d={cd:.4f}')
    print(f'  Mann-Whitney U={u_s:.1f}, p={u_p:.2e}')

# Save results
output = {
    'experiment': 'European Control MPNet Robustness',
    'timestamp': datetime.now().isoformat(),
    'models': ['all-MiniLM-L6-v2 (384D)', 'all-mpnet-base-v2 (768D)'],
    'per_language': results,
    'group_comparison': {
        'minilm': {
            'european_mean': float(np.mean(euro_ml)),
            'indian_mean': float(np.mean(indian_ml)),
        },
        'mpnet': {
            'european_mean': float(np.mean(euro_mp)),
            'indian_mean': float(np.mean(indian_mp)),
        },
    },
}
out_path = 'papers/paper8_efi/results/exp_european_control_mpnet.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nResults saved to {out_path}')

# Save embeddings
emb_out = {
    'experiment': 'European Control MPNet Embeddings',
    'timestamp': datetime.now().isoformat(),
    'embeddings': all_embs,
}
emb_path = 'papers/paper8_efi/results/exp_european_control_mpnet_embeddings.json'
with open(emb_path, 'w', encoding='utf-8') as f:
    json.dump(emb_out, f, indent=2, ensure_ascii=False)
print(f'Embeddings saved to {emb_path}')
