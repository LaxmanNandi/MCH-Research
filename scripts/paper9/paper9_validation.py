#!/usr/bin/env python3
"""
Paper 9 Validation Experiments A, B, C
======================================
A: MuRIL degeneracy sanity check
B: LaBSE European language control
C: Variance ratio bootstrap CIs
"""

import sys
import json
import numpy as np
import os
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/paper9"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ============================================================================
# EXPERIMENT A: MuRIL Degeneracy Sanity Check
# ============================================================================

def experiment_a():
    print("=" * 70)
    print("EXPERIMENT A: MuRIL Degeneracy Sanity Check")
    print("=" * 70)

    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    model = AutoModel.from_pretrained("google/muril-base-cased")
    model.eval()

    # Step 1: Verify tokenization for Kannada/Tamil/Hindi
    print("\n--- Step 1: Tokenization verification ---")
    test_sentences = {
        "English": "The patient has a fever.",
        "Kannada": "ರೋಗಿಗೆ ಜ್ವರವಿದೆ.",
        "Tamil": "நோயாளிக்கு காய்ச்சல் உள்ளது.",
        "Hindi": "मरीज को बुखार है।",
    }

    for lang, sent in test_sentences.items():
        tokens = tokenizer.tokenize(sent)
        ids = tokenizer.encode(sent)
        print(f"  {lang:8s}: {len(tokens)} tokens, {len(ids)} ids")
        print(f"           Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    # Step 2-3: Encode with proper attention masks, check norms
    print("\n--- Step 2-3: Encoding with attention masks + norm check ---")

    def encode_muril(text, pooling="cls"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        assert "attention_mask" in inputs, "Missing attention mask!"
        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        elif pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            emb = emb.squeeze().numpy()
        elif pooling == "mean_normalized":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            emb = emb.squeeze().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-10)  # L2 normalize
        return emb

    # Compute norms for all pooling strategies
    for pooling in ["cls", "mean", "mean_normalized"]:
        norms = []
        for lang, sent in test_sentences.items():
            emb = encode_muril(sent, pooling)
            norms.append(np.linalg.norm(emb))
        print(f"  {pooling:20s}: norm mean={np.mean(norms):.4f} std={np.std(norms):.6f}")

    # Step 4: Full similarity matrix across conditions
    print("\n--- Step 4: Similarity matrix ---")

    conditions = {
        "EN_fever": "The patient has a fever.",
        "KN_fever": "ರೋಗಿಗೆ ಜ್ವರವಿದೆ.",
        "TA_fever": "நோயாளிக்கு காய்ச்சல் உள்ளது.",
        "HI_fever": "मरीज को बुखार है।",
        "EN_bp": "Blood pressure is elevated.",
        "EN_weather": "The weather is sunny today.",
        "EN_food": "I like to eat pizza for dinner.",
        "Random_str": "xkq jwm plz rvt bnh",
        "Single_char": "a",
        "Long_random": "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz "), 100)),
    }

    # Also add random numpy vectors
    np.random.seed(42)
    rand_vec_1 = np.random.randn(768).astype(np.float32)
    rand_vec_2 = np.random.randn(768).astype(np.float32)
    rand_cosine = cosine_similarity(rand_vec_1, rand_vec_2)

    results_a = {"pooling_strategies": {}}

    for pooling in ["cls", "mean", "mean_normalized"]:
        print(f"\n  Pooling: {pooling}")
        embs = {}
        for name, text in conditions.items():
            embs[name] = encode_muril(text, pooling)

        # Compute key similarities
        pairs = [
            ("Same lang same sent", "EN_fever", "EN_fever"),
            ("EN-KN same meaning", "EN_fever", "KN_fever"),
            ("EN-TA same meaning", "EN_fever", "TA_fever"),
            ("EN-HI same meaning", "EN_fever", "HI_fever"),
            ("EN diff meaning", "EN_fever", "EN_bp"),
            ("EN unrelated", "EN_fever", "EN_weather"),
            ("EN very unrelated", "EN_fever", "EN_food"),
            ("EN vs random string", "EN_fever", "Random_str"),
            ("EN vs single char", "EN_fever", "Single_char"),
            ("EN vs long random", "EN_fever", "Long_random"),
            ("Random vs weather", "Random_str", "EN_weather"),
        ]

        pair_results = {}
        for label, k1, k2 in pairs:
            if k1 == k2:
                sim = 1.0
            else:
                sim = cosine_similarity(embs[k1], embs[k2])
            pair_results[label] = float(sim)
            print(f"    {label:30s}: {sim:.6f}")

        pair_results["Random numpy vectors"] = float(rand_cosine)
        print(f"    {'Random numpy vectors':30s}: {rand_cosine:.6f}")

        # Verdict
        random_sim = cosine_similarity(embs["EN_fever"], embs["Random_str"])
        if random_sim > 0.95:
            verdict = "DEGENERATE — random input similarity > 0.95"
        elif random_sim > 0.7:
            verdict = "SUSPICIOUS — random input similarity > 0.7"
        else:
            verdict = "VALID — random input has low similarity"

        print(f"    VERDICT: {verdict}")

        results_a["pooling_strategies"][pooling] = {
            "pairs": pair_results,
            "verdict": verdict,
        }

    # Save
    results_a["random_numpy_cosine"] = float(rand_cosine)
    results_a["timestamp"] = datetime.now().isoformat()

    with open(os.path.join(OUTPUT_DIR, "paper9_exp_a_muril_degeneracy.json"), "w") as f:
        json.dump(results_a, f, indent=2)

    print(f"\n  Saved: paper9_exp_a_muril_degeneracy.json")
    return results_a


# ============================================================================
# EXPERIMENT B: LaBSE European Language Control
# ============================================================================

def experiment_b():
    print("\n" + "=" * 70)
    print("EXPERIMENT B: LaBSE European Language Control")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer

    # European clinical sentences (from Paper 8)
    european_sentences = {
        "French": [
            "Le patient a de la fièvre.",
            "La tension artérielle est élevée.",
            "Le patient signale une douleur thoracique.",
            "Le rythme cardiaque est irrégulier.",
            "Le patient a des difficultés à respirer.",
            "Le patient présente de la fièvre, des douleurs articulaires et une éruption maculaire diffuse sur le tronc.",
            "Les lectures de tension artérielle montrent une hypertension systolique avec des changements orthostatiques.",
            "Le patient signale une douleur thoracique rétrosternale irradiant vers le bras gauche, aggravée à l'effort.",
            "Rythme cardiaque irrégulier avec épisodes de palpitations et de quasi-syncope.",
            "Dyspnée progressive à l'effort avec œdème bilatéral des membres inférieurs.",
            "L'élévation du segment ST dans les dérivations V1-V4 avec des changements réciproques suggère un infarctus du myocarde antérieur aigu nécessitant un cathétérisme d'urgence.",
            "Un D-dimère élevé avec un score de Wells de 6 justifie une angiographie pulmonaire par CT pour exclure une embolie pulmonaire.",
            "Les mesures sérielles de troponine montrant un schéma ascendant avec des changements dynamiques de l'ECG indiquent un syndrome coronarien aigu.",
            "L'échocardiographie révèle une régurgitation mitrale sévère avec dilatation de l'oreillette gauche et hypertension pulmonaire.",
            "La surveillance hémodynamique montre un choc cardiogénique avec des pressions de remplissage élevées nécessitant un support inotrope.",
        ],
        "Spanish": [
            "El paciente tiene fiebre.",
            "La presión arterial está elevada.",
            "El paciente reporta dolor en el pecho.",
            "El ritmo cardíaco es irregular.",
            "El paciente tiene dificultad para respirar.",
            "El paciente presenta fiebre, dolor articular y un sarpullido macular difuso en el tronco.",
            "Las lecturas de presión arterial muestran hipertensión sistólica con cambios ortostáticos.",
            "El paciente reporta dolor torácico subesternal que irradia al brazo izquierdo, peor con el esfuerzo.",
            "Ritmo cardíaco irregular con episodios de palpitaciones y casi-síncope.",
            "Disnea progresiva al esfuerzo con edema bilateral de miembros inferiores.",
            "La elevación del ST en derivaciones V1-V4 con cambios recíprocos sugiere infarto agudo de miocardio anterior que requiere cateterismo de emergencia.",
            "Un D-dímero elevado con puntuación de Wells de 6 justifica una angiografía pulmonar por TC para descartar embolia pulmonar.",
            "Las mediciones seriadas de troponina que muestran un patrón ascendente con cambios dinámicos del ECG indican síndrome coronario agudo.",
            "La ecocardiografía revela regurgitación mitral severa con agrandamiento de la aurícula izquierda e hipertensión pulmonar.",
            "El monitoreo hemodinámico muestra shock cardiogénico con presiones de llenado elevadas que requieren soporte inotrópico.",
        ],
        "German": [
            "Der Patient hat Fieber.",
            "Der Blutdruck ist erhöht.",
            "Der Patient berichtet über Brustschmerzen.",
            "Die Herzfrequenz ist unregelmäßig.",
            "Der Patient hat Schwierigkeiten beim Atmen.",
            "Der Patient zeigt Fieber, Gelenkschmerzen und einen diffusen makulären Ausschlag am Rumpf.",
            "Die Blutdruckwerte zeigen systolische Hypertonie mit orthostatischen Veränderungen.",
            "Der Patient berichtet über retrosternale Brustschmerzen, die in den linken Arm ausstrahlen und bei Belastung schlimmer werden.",
            "Unregelmäßige Herzfrequenz mit Episoden von Palpitationen und Beinahe-Synkope.",
            "Progressive Belastungsdyspnoe mit bilateralem Unterschenkelödem.",
            "ST-Hebung in den Ableitungen V1-V4 mit reziproken Veränderungen deutet auf einen akuten Vorderwandinfarkt hin, der eine Notfallkatheterisierung erfordert.",
            "Ein erhöhter D-Dimer mit einem Wells-Score von 6 erfordert eine CT-Pulmonalangiographie zum Ausschluss einer Lungenembolie.",
            "Serielle Troponin-Messungen mit ansteigendem Muster bei dynamischen EKG-Veränderungen weisen auf ein akutes Koronarsyndrom hin.",
            "Die Echokardiographie zeigt eine schwere Mitralklappeninsuffizienz mit Vergrößerung des linken Vorhofs und pulmonaler Hypertonie.",
            "Das hämodynamische Monitoring zeigt einen kardiogenen Schock mit erhöhten Füllungsdrücken, der eine inotrope Unterstützung erfordert.",
        ],
    }

    # English reference (same as Paper 8/9)
    english_sentences = [
        "The patient has a fever.",
        "Blood pressure is elevated.",
        "The patient reports chest pain.",
        "Heart rate is irregular.",
        "The patient has difficulty breathing.",
        "Patient presents with fever, joint pain, and a diffuse macular rash on the trunk.",
        "Blood pressure readings show systolic hypertension with orthostatic changes.",
        "Patient reports substernal chest pain radiating to the left arm, worse on exertion.",
        "Irregular heart rate with episodes of palpitations and near-syncope.",
        "Progressive dyspnea on exertion with bilateral lower extremity edema.",
        "ST elevation in leads V1-V4 with reciprocal changes suggests acute anterior myocardial infarction requiring emergent catheterization.",
        "Elevated D-dimer with Wells score of 6 warrants CT pulmonary angiography to rule out pulmonary embolism.",
        "Serial troponin measurements showing a rising pattern with dynamic ECG changes indicate acute coronary syndrome.",
        "Echocardiography reveals severe mitral regurgitation with left atrial enlargement and pulmonary hypertension.",
        "Hemodynamic monitoring shows cardiogenic shock with elevated filling pressures requiring inotropic support.",
    ]

    # Indic sentences (from Paper 9)
    indic_sentences = {
        "Kannada": [
            "ರೋಗಿಗೆ ಜ್ವರವಿದೆ.", "ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ.", "ರೋಗಿಯು ಎದೆನೋವನ್ನು ವರದಿ ಮಾಡುತ್ತಾರೆ.",
            "ಹೃದಯ ಬಡಿತ ಅನಿಯಮಿತವಾಗಿದೆ.", "ರೋಗಿಗೆ ಉಸಿರಾಟದ ತೊಂದರೆ ಇದೆ.",
            "ರೋಗಿಯು ಜ್ವರ, ಕೀಲು ನೋವು ಮತ್ತು ಮುಂಡದ ಮೇಲೆ ವ್ಯಾಪಕ ಮ್ಯಾಕ್ಯುಲರ್ ರಾಶ್ ಅನ್ನು ಪ್ರಸ್ತುತಪಡಿಸುತ್ತಾರೆ.",
            "ರಕ್ತದೊತ್ತಡ ಓದುವಿಕೆಗಳು ಆರ್ಥೋಸ್ಟ್ಯಾಟಿಕ್ ಬದಲಾವಣೆಗಳೊಂದಿಗೆ ಸಿಸ್ಟೊಲಿಕ್ ಅಧಿಕ ರಕ್ತದೊತ್ತಡವನ್ನು ತೋರಿಸುತ್ತವೆ.",
            "ರೋಗಿಯು ಶ್ರಮದ ಮೇಲೆ ಕೆಟ್ಟದಾಗುವ ಎಡ ತೋಳಿಗೆ ವಿಕಿರಣಗೊಳ್ಳುವ ಉಪಸ್ಟರ್ನಲ್ ಎದೆ ನೋವನ್ನು ವರದಿ ಮಾಡುತ್ತಾರೆ.",
            "ಬಡಿತಗಳು ಮತ್ತು ಸಮೀಪ-ಸಿಂಕೋಪ್ ಎಪಿಸೋಡ್\u200cಗಳೊಂದಿಗೆ ಅನಿಯಮಿತ ಹೃದಯ ಬಡಿತ.",
            "ದ್ವಿಪಕ್ಷೀಯ ಕೆಳ ತುದಿಯ ಊತದೊಂದಿಗೆ ಶ್ರಮದ ಮೇಲೆ ಪ್ರಗತಿಶೀಲ ಡಿಸ್ಪ್ನಿಯಾ.",
            "ಲೀಡ್\u200cಗಳಲ್ಲಿ V1-V4 ನಲ್ಲಿ ST ಎತ್ತರವು ಪರಸ್ಪರ ಬದಲಾವಣೆಗಳೊಂದಿಗೆ ತೀವ್ರ ಮುಂಭಾಗದ ಮಯೋಕಾರ್ಡಿಯಲ್ ಇನ್ಫಾರ್ಕ್ಷನ್ ಅನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
            "6 ರ ವೆಲ್ಸ್ ಸ್ಕೋರ್\u200cನೊಂದಿಗೆ ಎತ್ತರದ D-ಡೈಮರ್ ಪಲ್ಮನರಿ ಎಂಬಾಲಿಸಮ್ ಅನ್ನು ತಳ್ಳಿಹಾಕಲು CT ಪಲ್ಮನರಿ ಆಂಜಿಯೋಗ್ರಫಿ ಅಗತ್ಯ.",
            "ಡೈನಾಮಿಕ್ ECG ಬದಲಾವಣೆಗಳೊಂದಿಗೆ ಏರುತ್ತಿರುವ ಮಾದರಿಯನ್ನು ತೋರಿಸುವ ಸರಣಿ ಟ್ರೋಪೋನಿನ್ ಅಳತೆಗಳು ತೀವ್ರ ಕರೋನರಿ ಸಿಂಡ್ರೋಮ್ ಅನ್ನು ಸೂಚಿಸುತ್ತವೆ.",
            "ಎಕೋಕಾರ್ಡಿಯೋಗ್ರಫಿ ಎಡ ಹೃತ್ಕರ್ಣ ಹಿಗ್ಗುವಿಕೆ ಮತ್ತು ಶ್ವಾಸಕೋಶದ ಅಧಿಕ ರಕ್ತದೊತ್ತಡದೊಂದಿಗೆ ತೀವ್ರ ಮೈಟ್ರಲ್ ರಿಗರ್ಗಿಟೇಶನ್ ಅನ್ನು ಬಹಿರಂಗಪಡಿಸುತ್ತದೆ.",
            "ಹಿಮೋಡೈನಾಮಿಕ್ ಮಾನಿಟರಿಂಗ್ ಇನೋಟ್ರೋಪಿಕ್ ಬೆಂಬಲದ ಅಗತ್ಯವಿರುವ ಎತ್ತರದ ಫಿಲ್ಲಿಂಗ್ ಒತ್ತಡಗಳೊಂದಿಗೆ ಕಾರ್ಡಿಯೋಜೆನಿಕ್ ಶಾಕ್ ಅನ್ನು ತೋರಿಸುತ್ತದೆ.",
        ],
        "Tamil": [
            "நோயாளிக்கு காய்ச்சல் உள்ளது.", "இரத்த அழுத்தம் அதிகமாக உள்ளது.",
            "நோயாளி நெஞ்சு வலியை தெரிவிக்கிறார்.", "இதய துடிப்பு ஒழுங்கற்றதாக உள்ளது.",
            "நோயாளிக்கு சுவாசிப்பதில் சிரமம் உள்ளது.",
            "நோயாளி காய்ச்சல், மூட்டு வலி மற்றும் உடலில் பரவலான மாகுலர் தடிப்புடன் வருகிறார்.",
            "இரத்த அழுத்த அளவீடுகள் ஆர்த்தோஸ்டாடிக் மாற்றங்களுடன் சிஸ்டாலிக் உயர் இரத்த அழுத்தத்தைக் காட்டுகின்றன.",
            "நோயாளி உடற்பயிற்சியில் மோசமாகும் இடது கையில் பரவும் மார்பு வலியைத் தெரிவிக்கிறார்.",
            "படபடப்பு மற்றும் மயக்கம் போன்ற அறிகுறிகளுடன் ஒழுங்கற்ற இதய துடிப்பு.",
            "இருபக்க கீழ் முனை வீக்கத்துடன் உடற்பயிற்சியில் முற்போக்கான மூச்சுத்திணறல்.",
            "V1-V4 லீட்களில் ST உயர்வு பரஸ்பர மாற்றங்களுடன் கடுமையான முன்புற மாரடைப்பைக் குறிக்கிறது.",
            "6 வெல்ஸ் மதிப்பெண்ணுடன் உயர்ந்த D-டைமர் நுரையீரல் தக்கையடைப்பை நிராகரிக்க CT நுரையீரல் ஆஞ்சியோகிராபி தேவை.",
            "டைனமிக் ECG மாற்றங்களுடன் உயரும் முறையைக் காட்டும் தொடர் ட்ரோப்போனின் அளவீடுகள் கடுமையான கரோனரி நோய்க்குறியைக் குறிக்கின்றன.",
            "எக்கோகார்டியோகிராபி இடது ஏட்ரிய விரிவாக்கம் மற்றும் நுரையீரல் உயர் இரத்த அழுத்தத்துடன் கடுமையான மைட்ரல் ரெகர்ஜிடேஷனை வெளிப்படுத்துகிறது.",
            "ஹீமோடைனமிக் கண்காணிப்பு இனோட்ரோபிக் ஆதரவு தேவைப்படும் உயர்ந்த நிரப்பு அழுத்தங்களுடன் கார்டியோஜெனிக் அதிர்ச்சியைக் காட்டுகிறது.",
        ],
        "Hindi": [
            "मरीज को बुखार है।", "रक्तचाप बढ़ा हुआ है।",
            "मरीज सीने में दर्द की शिकायत करता है।", "हृदय गति अनियमित है।",
            "मरीज को सांस लेने में कठिनाई है।",
            "मरीज बुखार, जोड़ों में दर्द और धड़ पर व्यापक मैक्युलर रैश के साथ प्रस्तुत होता है।",
            "रक्तचाप की रीडिंग ऑर्थोस्टैटिक परिवर्तनों के साथ सिस्टोलिक उच्च रक्तचाप दिखाती है।",
            "मरीज परिश्रम पर बिगड़ने वाले बाएं हाथ में विकिरण करने वाले सबस्टर्नल सीने में दर्द की रिपोर्ट करता है।",
            "धड़कन और निकट-बेहोशी के एपिसोड के साथ अनियमित हृदय गति।",
            "द्विपक्षीय निचले छोर के शोफ के साथ परिश्रम पर प्रगतिशील डिस्पनिया।",
            "V1-V4 लीड में ST उन्नयन पारस्परिक परिवर्तनों के साथ तीव्र पूर्वकाल मायोकार्डियल इन्फार्क्शन का सुझाव देता है।",
            "6 के वेल्स स्कोर के साथ उन्नत D-डाइमर फुफ्फुसीय अन्त:शल्यता को रद्द करने के लिए CT फुफ्फुसीय एंजियोग्राफी की आवश्यकता है।",
            "गतिशील ECG परिवर्तनों के साथ बढ़ते पैटर्न दिखाने वाले सीरियल ट्रोपोनिन माप तीव्र कोरोनरी सिंड्रोम का संकेत देते हैं।",
            "इकोकार्डियोग्राफी बाएं आलिंद विस्तार और फुफ्फुसीय उच्च रक्तचाप के साथ गंभीर माइट्रल रिगर्जिटेशन प्रकट करती है।",
            "हेमोडायनामिक मॉनिटरिंग इनोट्रोपिक सपोर्ट की आवश्यकता वाले ऊंचे फिलिंग प्रेशर के साथ कार्डियोजेनिक शॉक दिखाती है।",
        ],
    }

    MODELS_B = [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
        ("sentence-transformers/LaBSE", 768),
    ]

    all_languages = {**{"English": english_sentences}, **indic_sentences, **european_sentences}

    results_b = {"models": {}}

    for model_name, dim in MODELS_B:
        print(f"\n  Loading {model_name}...", flush=True)
        embedder = SentenceTransformer(model_name)

        eng_embs = np.array([embedder.encode(s) for s in english_sentences])

        model_efis = {}
        for lang, sents in all_languages.items():
            if lang == "English":
                continue
            lang_embs = np.array([embedder.encode(s) for s in sents])
            efis = [cosine_similarity(eng_embs[i], lang_embs[i]) for i in range(15)]
            model_efis[lang] = {
                "mean": float(np.mean(efis)),
                "std": float(np.std(efis)),
                "simple": float(np.mean(efis[:5])),
                "medium": float(np.mean(efis[5:10])),
                "complex": float(np.mean(efis[10:15])),
            }
            print(f"    {lang:10s}: EFI = {np.mean(efis):.4f} ± {np.std(efis):.4f}", flush=True)

        results_b["models"][model_name] = {"dim": dim, "languages": model_efis}

        del embedder
        import gc; gc.collect()

    # Print comparison table
    print(f"\n  {'Language':10s}", end="")
    for m, _ in MODELS_B:
        print(f"  {m.split('/')[-1]:>15s}", end="")
    print()
    print("  " + "-" * 60)

    for lang in ["Kannada", "Tamil", "Hindi", "French", "Spanish", "German"]:
        print(f"  {lang:10s}", end="")
        for m, _ in MODELS_B:
            efi = results_b["models"][m]["languages"].get(lang, {}).get("mean", 0)
            print(f"  {efi:15.4f}", end="")
        print()

    with open(os.path.join(OUTPUT_DIR, "paper9_exp_b_european_control.json"), "w") as f:
        json.dump(results_b, f, indent=2)
    print(f"\n  Saved: paper9_exp_b_european_control.json")
    return results_b


# ============================================================================
# EXPERIMENT C: Variance Ratio Bootstrap CIs
# ============================================================================

def experiment_c():
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Variance Ratio Bootstrap Confidence Intervals")
    print("=" * 70)

    # Load existing variance data
    with open("C:/Users/barla/mch_experiments/papers/paper8_efi/results/exp5_15trials_variance.json",
              encoding="utf-8") as f:
        d = json.load(f)

    raw = d["raw_responses"]

    from sentence_transformers import SentenceTransformer

    MODELS_C = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "sentence-transformers/LaBSE",
    ]

    N_BOOTSTRAP = 10000
    np.random.seed(42)

    results_c = {"n_bootstrap": N_BOOTSTRAP, "models": {}}

    for emb_name in MODELS_C:
        print(f"\n  Embedding: {emb_name}", flush=True)
        embedder = SentenceTransformer(emb_name)

        results_c["models"][emb_name] = {}

        for llm in ["DeepSeek_V3.1", "Mistral_Small"]:
            results_c["models"][emb_name][llm] = {}

            # Embed all responses for all languages
            lang_embeddings = {}
            for lang in ["English", "Kannada", "Tamil", "Hindi"]:
                all_embs = []  # list of (scenario, trial) embeddings
                scenarios = raw[llm][lang]
                for s_key in sorted(scenarios.keys()):
                    trials = scenarios[s_key]
                    for t in trials:
                        emb = embedder.encode(t["response"])
                        all_embs.append(emb)
                lang_embeddings[lang] = all_embs  # 75 embeddings (5 scenarios × 15 trials)

            # Compute pairwise variance for each language
            def compute_var(embs):
                sims = []
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        sims.append(cosine_similarity(embs[i], embs[j]))
                return np.var(sims)

            eng_var = compute_var(lang_embeddings["English"])

            for lang in ["Kannada", "Tamil", "Hindi"]:
                lang_var = compute_var(lang_embeddings[lang])
                vr_point = lang_var / eng_var if eng_var > 0 else float("inf")

                # Bootstrap
                n_total = len(lang_embeddings[lang])
                n_eng = len(lang_embeddings["English"])
                bootstrap_vrs = []

                for _ in range(N_BOOTSTRAP):
                    # Resample with replacement
                    idx_lang = np.random.choice(n_total, n_total, replace=True)
                    idx_eng = np.random.choice(n_eng, n_eng, replace=True)

                    boot_lang = [lang_embeddings[lang][i] for i in idx_lang]
                    boot_eng = [lang_embeddings["English"][i] for i in idx_eng]

                    boot_lang_var = compute_var(boot_lang)
                    boot_eng_var = compute_var(boot_eng)

                    if boot_eng_var > 0:
                        bootstrap_vrs.append(boot_lang_var / boot_eng_var)

                ci_lower = float(np.percentile(bootstrap_vrs, 2.5))
                ci_upper = float(np.percentile(bootstrap_vrs, 97.5))
                sig = ci_lower > 1.0  # VR > 1 is significant if lower CI > 1

                results_c["models"][emb_name][llm][lang] = {
                    "vr_point": float(vr_point),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "significant": bool(sig),
                }

                sig_str = "***" if sig else "ns"
                print(f"    {llm:15s} {lang:8s}: VR={vr_point:8.4f}  95% CI [{ci_lower:.4f}, {ci_upper:.4f}] {sig_str}", flush=True)

            del lang_embeddings

        del embedder
        import gc; gc.collect()

    # Kendall's tau for ranking consistency
    from scipy.stats import kendalltau
    print("\n  Kendall's tau for VR ranking consistency across models:")

    for llm in ["DeepSeek_V3.1", "Mistral_Small"]:
        model_names = list(results_c["models"].keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                vrs1 = [results_c["models"][m1][llm].get(l, {}).get("vr_point", 0) for l in ["Kannada", "Tamil", "Hindi"]]
                vrs2 = [results_c["models"][m2][llm].get(l, {}).get("vr_point", 0) for l in ["Kannada", "Tamil", "Hindi"]]
                tau, p = kendalltau(vrs1, vrs2)
                print(f"    {llm:15s} {m1.split('/')[-1]:15s} vs {m2.split('/')[-1]:15s}: tau={tau:.4f} p={p:.4f}")

    with open(os.path.join(OUTPUT_DIR, "paper9_exp_c_bootstrap_ci.json"), "w") as f:
        json.dump(results_c, f, indent=2)
    print(f"\n  Saved: paper9_exp_c_bootstrap_ci.json")
    return results_c


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("PAPER 9 VALIDATION EXPERIMENTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    results_a = experiment_a()
    results_b = experiment_b()
    results_c = experiment_c()

    # Write summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = []

    # Exp A verdict
    cls_verdict = results_a["pooling_strategies"]["cls"]["verdict"]
    mean_verdict = results_a["pooling_strategies"]["mean"]["verdict"]
    norm_verdict = results_a["pooling_strategies"]["mean_normalized"]["verdict"]
    summary.append(f"Experiment A: MuRIL CLS={cls_verdict}, Mean={mean_verdict}, Normalized={norm_verdict}")
    print(f"\n  A: {summary[-1]}")

    # Exp B verdict
    labse_langs = results_b["models"].get("sentence-transformers/LaBSE", {}).get("languages", {})
    indic_efis = [labse_langs.get(l, {}).get("mean", 0) for l in ["Kannada", "Tamil", "Hindi"]]
    euro_efis = [labse_langs.get(l, {}).get("mean", 0) for l in ["French", "Spanish", "German"]]
    indic_mean = np.mean(indic_efis) if indic_efis else 0
    euro_mean = np.mean(euro_efis) if euro_efis else 0
    gap = euro_mean - indic_mean
    summary.append(f"Experiment B: LaBSE Indic mean={indic_mean:.4f}, European mean={euro_mean:.4f}, gap={gap:.4f}")
    print(f"  B: {summary[-1]}")

    # Exp C verdict
    labse_results = results_c["models"].get("sentence-transformers/LaBSE", {})
    sig_count = sum(1 for llm in labse_results for lang in labse_results[llm]
                    if labse_results[llm][lang].get("significant", False))
    total = sum(1 for llm in labse_results for lang in labse_results[llm])
    summary.append(f"Experiment C: LaBSE VR>1 significant in {sig_count}/{total} language-LLM pairs")
    print(f"  C: {summary[-1]}")

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "paper9_validation_summary.json"), "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "summary": summary}, f, indent=2)

    print(f"\n  All results saved to {OUTPUT_DIR}/")
    print("  DONE.")


if __name__ == "__main__":
    main()
