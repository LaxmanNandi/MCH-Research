#!/usr/bin/env python3
"""
Paper 9 v2: Indic EFI Comparison with Degeneracy Checks
========================================================
Compares EFI across 7 models with built-in degeneracy validation.
"""

import sys
import json
import numpy as np
import os
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ============================================================================
# CLINICAL SENTENCES (from Paper 8)
# ============================================================================

SENTENCES = {
    "English": [
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
    ],
    "Kannada": [
        "ರೋಗಿಗೆ ಜ್ವರವಿದೆ.",
        "ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ.",
        "ರೋಗಿಯು ಎದೆನೋವನ್ನು ವರದಿ ಮಾಡುತ್ತಾರೆ.",
        "ಹೃದಯ ಬಡಿತ ಅನಿಯಮಿತವಾಗಿದೆ.",
        "ರೋಗಿಗೆ ಉಸಿರಾಟದ ತೊಂದರೆ ಇದೆ.",
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
        "நோயாளிக்கு காய்ச்சல் உள்ளது.",
        "இரத்த அழுத்தம் அதிகமாக உள்ளது.",
        "நோயாளி நெஞ்சு வலியை தெரிவிக்கிறார்.",
        "இதய துடிப்பு ஒழுங்கற்றதாக உள்ளது.",
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
        "मरीज को बुखार है।",
        "रक्तचाप बढ़ा हुआ है।",
        "मरीज सीने में दर्द की शिकायत करता है।",
        "हृदय गति अनियमित है।",
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

# Degeneracy check sentences
DEGENERACY_CHECK = {
    "same_en": "The patient has a fever.",
    "same_kn": "ರೋಗಿಗೆ ಜ್ವರವಿದೆ.",
    "unrelated_en": "The weather is sunny today.",
    "unrelated_kn": "ಇಂದು ಹವಾಮಾನ ಬಿಸಿಲಿದೆ.",
}

# ============================================================================
# MODELS
# ============================================================================

MODELS = [
    # Paper 8 baselines
    ("all-MiniLM-L6-v2", "sentence-transformers", 384, "Baseline"),
    ("all-mpnet-base-v2", "sentence-transformers", 768, "Baseline"),
    # Multilingual sentence models
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers", 384, "Multilingual-ST"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-transformers", 768, "Multilingual-ST"),
    ("sentence-transformers/LaBSE", "sentence-transformers", 768, "Multilingual-ST"),
    # Indic MLMs (expected degenerate — negative controls)
    ("google/muril-base-cased", "transformers-cls", 768, "Indic-MLM"),
    ("ai4bharat/IndicBERTv2-MLM-only", "transformers-cls", 768, "Indic-MLM"),
]

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def load_and_encode(name, model_type):
    if model_type == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        return lambda texts: np.array([model.encode(t) for t in texts])
    else:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        def encode(texts):
            embs = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    out = model(**inputs)
                embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
            return np.array(embs)
        return encode

def main():
    OUTPUT_DIR = "C:/Users/barla/mch_experiments/data/paper9"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for model_name, model_type, dim, category in MODELS:
        print(f"\n{'='*60}")
        print(f"{category}: {model_name} ({dim}D)")
        print(f"{'='*60}")

        try:
            encode_fn = load_and_encode(model_name, model_type)
            print("  Loaded.", flush=True)
        except Exception as e:
            print(f"  FAILED: {str(e)[:80]}")
            continue

        # DEGENERACY CHECK
        degen_embs = {k: encode_fn([v])[0] for k, v in DEGENERACY_CHECK.items()}
        same_sim = cosine_similarity(degen_embs["same_en"], degen_embs["same_kn"])
        unrel_sim = cosine_similarity(degen_embs["same_en"], degen_embs["unrelated_en"])
        cross_unrel = cosine_similarity(degen_embs["same_en"], degen_embs["unrelated_kn"])
        is_degenerate = unrel_sim > 0.7

        print(f"  Degeneracy check: same={same_sim:.4f} unrel={unrel_sim:.4f} cross_unrel={cross_unrel:.4f} → {'DEGENERATE' if is_degenerate else 'VALID'}", flush=True)

        # EFI COMPUTATION
        eng_embs = encode_fn(SENTENCES["English"])

        model_result = {
            "model": model_name,
            "type": model_type,
            "category": category,
            "dim": dim,
            "degeneracy": {
                "same_meaning_sim": float(same_sim),
                "unrelated_sim": float(unrel_sim),
                "cross_unrelated_sim": float(cross_unrel),
                "is_degenerate": bool(is_degenerate),
            },
            "languages": {},
        }

        for lang in ["Kannada", "Tamil", "Hindi"]:
            lang_embs = encode_fn(SENTENCES[lang])
            efis = [cosine_similarity(eng_embs[i], lang_embs[i]) for i in range(15)]

            model_result["languages"][lang] = {
                "mean_efi": float(np.mean(efis)),
                "std_efi": float(np.std(efis)),
                "simple_efi": float(np.mean(efis[:5])),
                "medium_efi": float(np.mean(efis[5:10])),
                "complex_efi": float(np.mean(efis[10:15])),
                "all_efis": [float(e) for e in efis],
            }

            d = "⚠" if is_degenerate else " "
            print(f"  {d}{lang:8s}: EFI = {np.mean(efis):.4f} ± {np.std(efis):.4f}  [S:{np.mean(efis[:5]):.3f} M:{np.mean(efis[5:10]):.3f} C:{np.mean(efis[10:15]):.3f}]", flush=True)

        all_results.append(model_result)

        import gc; gc.collect()

    # SUMMARY
    print(f"\n{'='*70}")
    print("PAPER 9: EFI COMPARISON WITH DEGENERACY VALIDATION")
    print(f"{'='*70}")
    print()
    print(f"{'Model':45s} {'Cat':12s} {'Degen':>6s} {'Kannada':>8s} {'Tamil':>8s} {'Hindi':>8s}")
    print("-" * 90)

    for r in all_results:
        d = "YES" if r["degeneracy"]["is_degenerate"] else "no"
        kn = r["languages"].get("Kannada", {}).get("mean_efi", 0)
        ta = r["languages"].get("Tamil", {}).get("mean_efi", 0)
        hi = r["languages"].get("Hindi", {}).get("mean_efi", 0)
        print(f"{r['model']:45s} {r['category']:12s} {d:>6s} {kn:8.4f} {ta:8.4f} {hi:8.4f}")

    # Save
    output = {
        "experiment": "Paper 9 v2: Indic EFI with Degeneracy Validation",
        "timestamp": datetime.now().isoformat(),
        "sentences_per_language": 15,
        "languages": ["English", "Kannada", "Tamil", "Hindi"],
        "degeneracy_threshold": 0.7,
        "results": all_results,
    }

    outfile = os.path.join(OUTPUT_DIR, "paper9_efi_v2_with_degeneracy.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {outfile}")

if __name__ == "__main__":
    main()
