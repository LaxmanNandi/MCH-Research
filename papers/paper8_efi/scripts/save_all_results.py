"""
Paper 8: Save all experiment results to JSON
Runs Experiments 1-3 (embedding EFI) locally.
Experiments 4-5 (LLM) results entered from prior runs.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu, ttest_ind
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Clinical Sentence Battery ===
sentences = {
    "simple": {
        "en": ["The patient has fever", "The patient has cough", "Blood pressure is high",
               "The patient has headache", "Heart rate is fast"],
        "kn": ["\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c9c\u0ccd\u0cb5\u0cb0 \u0c87\u0ca6\u0cc6",
               "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c95\u0cc6\u0cae\u0ccd\u0cae\u0cc1 \u0c87\u0ca6\u0cc6",
               "\u0cb0\u0c95\u0ccd\u0ca4\u0ca6\u0cca\u0ca4\u0ccd\u0ca4\u0ca1 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbe\u0c97\u0cbf\u0ca6\u0cc6",
               "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0ca4\u0cb2\u0cc6\u0ca8\u0ccb\u0cb5\u0cc1 \u0c87\u0ca6\u0cc6",
               "\u0cb9\u0cc3\u0ca6\u0caf \u0cac\u0ca1\u0cbf\u0ca4 \u0cb5\u0cc7\u0c97\u0cb5\u0cbe\u0c97\u0cbf\u0ca6\u0cc6"],
        "ta": ["\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
               "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0b87\u0bb0\u0bc1\u0bae\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
               "\u0b87\u0bb0\u0ba4\u0bcd\u0ba4 \u0b85\u0bb4\u0bc1\u0ba4\u0bcd\u0ba4\u0bae\u0bcd \u0b85\u0ba4\u0bbf\u0b95\u0bae\u0bbe\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
               "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bb2\u0bc8\u0bb5\u0bb2\u0bbf \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
               "\u0b87\u0ba4\u0baf \u0ba4\u0bc1\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bc1 \u0bb5\u0bc7\u0b95\u0bae\u0bbe\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1"],
        "hi": ["\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u092c\u0941\u0916\u093e\u0930 \u0939\u0948",
               "\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0916\u093e\u0902\u0938\u0940 \u0939\u0948",
               "\u0930\u0915\u094d\u0924\u091a\u093e\u092a \u092c\u0922\u093c\u093e \u0939\u0941\u0906 \u0939\u0948",
               "\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0938\u093f\u0930\u0926\u0930\u094d\u0926 \u0939\u0948",
               "\u0939\u0943\u0926\u092f \u0917\u0924\u093f \u0924\u0947\u091c\u093c \u0939\u0948"]
    },
    "medium": {
        "en": ["Patient complains of chest pain radiating to left arm with sweating",
               "Child presents with high fever and rash for three days",
               "Elderly patient with sudden onset weakness on right side of body",
               "Woman reports severe abdominal pain with nausea and vomiting",
               "Patient has difficulty breathing worse when lying down at night"],
        "kn": ["\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c8e\u0ca6\u0cc6 \u0ca8\u0ccb\u0cb5\u0cc1 \u0c8e\u0ca1 \u0ca4\u0ccb\u0cb3\u0cbf\u0c97\u0cc6 \u0cb9\u0cb0\u0ca1\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cac\u0cc6\u0cb5\u0cb0\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6",
               "\u0cae\u0c97\u0cc1\u0cb5\u0cbf\u0c97\u0cc6 \u0cae\u0cc2\u0cb0\u0cc1 \u0ca6\u0cbf\u0ca8\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbf\u0ca8 \u0c9c\u0ccd\u0cb5\u0cb0 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0ca6\u0ca6\u0ccd\u0ca6\u0cc1 \u0c87\u0ca6\u0cc6",
               "\u0cb5\u0caf\u0cb8\u0ccd\u0cb8\u0cbe\u0ca6 \u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0ca6\u0cc7\u0cb9\u0ca6 \u0cac\u0cb2 \u0cad\u0cbe\u0c97\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf \u0cb9\u0ca0\u0cbe\u0ca4\u0ccd \u0ca6\u0ccc\u0cb0\u0ccd\u0cac\u0cb2\u0ccd\u0caf",
               "\u0cae\u0cb9\u0cbf\u0cb3\u0cc6\u0c97\u0cc6 \u0ca4\u0cc0\u0cb5\u0ccd\u0cb0 \u0cb9\u0cca\u0c9f\u0ccd\u0c9f\u0cc6 \u0ca8\u0ccb\u0cb5\u0cc1 \u0cb5\u0cbe\u0c95\u0cb0\u0cbf\u0c95\u0cc6 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cb5\u0cbe\u0c82\u0ca4\u0cbf",
               "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c89\u0cb8\u0cbf\u0cb0\u0cbe\u0c9f\u0ca6 \u0ca4\u0cca\u0c82\u0ca6\u0cb0\u0cc6 \u0cb0\u0cbe\u0ca4\u0ccd\u0cb0\u0cbf \u0cae\u0cb2\u0c97\u0cbf\u0ca6\u0cbe\u0c97 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbe\u0c97\u0cc1\u0ca4\u0ccd\u0ca4\u0ca6\u0cc6"],
        "ta": ["\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0ba8\u0bc6\u0b9e\u0bcd\u0b9a\u0bc1 \u0bb5\u0bb2\u0bbf \u0b87\u0b9f\u0ba4\u0bc1 \u0b95\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0baa\u0bb0\u0bb5\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bb5\u0bbf\u0baf\u0bb0\u0bcd\u0bb5\u0bc8 \u0bb5\u0bb0\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
               "\u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0bae\u0bc2\u0ba9\u0bcd\u0bb1\u0bc1 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0b85\u0ba4\u0bbf\u0b95 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0b9a\u0bca\u0bb1\u0bbf \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
               "\u0bb5\u0baf\u0ba4\u0bbe\u0ba9 \u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0b89\u0b9f\u0bb2\u0bbf\u0ba9\u0bcd \u0bb5\u0bb2\u0ba4\u0bc1 \u0baa\u0b95\u0bcd\u0b95\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd \u0ba4\u0bbf\u0b9f\u0bc0\u0bb0\u0bcd \u0baa\u0bb2\u0bb5\u0bc0\u0ba9\u0bae\u0bcd",
               "\u0baa\u0bc6\u0ba3\u0bcd\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b95\u0b9f\u0bc1\u0bae\u0bc8\u0baf\u0bbe\u0ba9 \u0bb5\u0baf\u0bbf\u0bb1\u0bcd\u0bb1\u0bc1 \u0bb5\u0bb2\u0bbf \u0b95\u0bc1\u0bae\u0b9f\u0bcd\u0b9f\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bb5\u0bbe\u0ba8\u0bcd\u0ba4\u0bbf",
               "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0b9a\u0bc1\u0bb5\u0bbe\u0b9a\u0bbf\u0baa\u0bcd\u0baa\u0ba4\u0bbf\u0bb2\u0bcd \u0b9a\u0bbf\u0bb0\u0bae\u0bae\u0bcd \u0b87\u0bb0\u0bb5\u0bbf\u0bb2\u0bcd \u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bbe\u0bb2\u0bcd \u0bae\u0bcb\u0b9a\u0bae\u0bbe\u0b95\u0bbf\u0bb1\u0ba4\u0bc1"],
        "hi": ["\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0938\u0940\u0928\u0947 \u092e\u0947\u0902 \u0926\u0930\u094d\u0926 \u092c\u093e\u090f\u0902 \u0939\u093e\u0925 \u0924\u0915 \u092b\u0948\u0932 \u0930\u0939\u093e \u0939\u0948 \u0914\u0930 \u092a\u0938\u0940\u0928\u093e \u0906 \u0930\u0939\u093e \u0939\u0948",
               "\u092c\u091a\u094d\u091a\u0947 \u0915\u094b \u0924\u0940\u0928 \u0926\u093f\u0928\u094b\u0902 \u0938\u0947 \u0924\u0947\u091c\u093c \u092c\u0941\u0916\u093e\u0930 \u0914\u0930 \u0926\u093e\u0928\u0947 \u0939\u0948\u0902",
               "\u092c\u0941\u091c\u093c\u0941\u0930\u094d\u0917 \u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0936\u0930\u0940\u0930 \u0915\u0947 \u0926\u093e\u0939\u093f\u0928\u0947 \u0939\u093f\u0938\u094d\u0938\u0947 \u092e\u0947\u0902 \u0905\u091a\u093e\u0928\u0915 \u0915\u092e\u091c\u093c\u094b\u0930\u0940",
               "\u092e\u0939\u093f\u0932\u093e \u0915\u094b \u092a\u0947\u091f \u092e\u0947\u0902 \u0924\u0940\u0935\u094d\u0930 \u0926\u0930\u094d\u0926 \u092e\u0924\u0932\u0940 \u0914\u0930 \u0909\u0932\u094d\u091f\u0940",
               "\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0938\u093e\u0902\u0938 \u0932\u0947\u0928\u0947 \u092e\u0947\u0902 \u0915\u0920\u093f\u0928\u093e\u0908 \u0930\u093e\u0924 \u0915\u094b \u0932\u0947\u091f\u0928\u0947 \u092a\u0930 \u092c\u0922\u093c \u091c\u093e\u0924\u0940 \u0939\u0948"]
    },
    "complex": {
        "en": ["ECG shows ST elevation in leads II III aVF suggesting inferior wall MI",
               "Patient with elevated troponin and D-dimer levels needs urgent cardiology consultation",
               "CT scan reveals hypodense lesion in right temporal lobe with midline shift",
               "Arterial blood gas shows metabolic acidosis with respiratory compensation pH 7.28",
               "Chest X-ray shows bilateral infiltrates consistent with ARDS in COVID positive patient"],
        "kn": ["ECG \u0caf\u0cb2\u0ccd\u0cb2\u0cbf leads II III aVF \u0ca8\u0cb2\u0ccd\u0cb2\u0cbf ST elevation \u0c95\u0cbe\u0ca3\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6 inferior wall MI \u0cb8\u0cc2\u0c9a\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0ca6\u0cc6",
               "\u0cb0\u0ccb\u0c97\u0cbf\u0caf troponin \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 D-dimer \u0cae\u0c9f\u0ccd\u0c9f\u0c97\u0cb3\u0cc1 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbe\u0c97\u0cbf\u0cb5\u0cc6 \u0ca4\u0cc1\u0cb0\u0ccd\u0ca4\u0cc1 cardiology consultation \u0cac\u0cc7\u0c95\u0cc1",
               "CT scan \u0ca8\u0cb2\u0ccd\u0cb2\u0cbf right temporal lobe \u0ca8\u0cb2\u0ccd\u0cb2\u0cbf hypodense lesion midline shift \u0ca8\u0cca\u0c82\u0ca6\u0cbf\u0c97\u0cc6 \u0c95\u0cbe\u0ca3\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6",
               "Arterial blood gas metabolic acidosis respiratory compensation pH 7.28 \u0ca4\u0ccb\u0cb0\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0ca6\u0cc6",
               "Chest X-ray bilateral infiltrates ARDS COVID positive \u0cb0\u0ccb\u0c97\u0cbf\u0caf\u0cb2\u0ccd\u0cb2\u0cbf \u0c95\u0cbe\u0ca3\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6"],
        "ta": ["ECG \u0baf\u0bbf\u0bb2\u0bcd leads II III aVF \u0b87\u0bb2\u0bcd ST elevation \u0ba4\u0bc6\u0bb0\u0bbf\u0b95\u0bbf\u0bb1\u0ba4\u0bc1 inferior wall MI \u0b90 \u0b9a\u0bc1\u0b9f\u0bcd\u0b9f\u0bbf\u0b95\u0bcd\u0b95\u0bbe\u0b9f\u0bcd\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
               "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0baf\u0bbf\u0ba9\u0bcd troponin \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd D-dimer \u0b85\u0bb3\u0bb5\u0bc1\u0b95\u0bb3\u0bcd \u0b89\u0baf\u0bb0\u0bcd\u0ba8\u0bcd\u0ba4\u0bc1\u0bb3\u0bcd\u0bb3\u0ba9 \u0b85\u0bb5\u0b9a\u0bb0 cardiology consultation \u0ba4\u0bc7\u0bb5\u0bc8",
               "CT scan \u0b87\u0bb2\u0bcd right temporal lobe \u0b87\u0bb2\u0bcd hypodense lesion midline shift \u0b89\u0b9f\u0ba9\u0bcd \u0ba4\u0bc6\u0bb0\u0bbf\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
               "Arterial blood gas metabolic acidosis respiratory compensation pH 7.28 \u0b95\u0bbe\u0b9f\u0bcd\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
               "Chest X-ray bilateral infiltrates ARDS COVID positive \u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0baf\u0bbf\u0bb2\u0bcd \u0ba4\u0bc6\u0bb0\u0bbf\u0b95\u0bbf\u0bb1\u0ba4\u0bc1"],
        "hi": ["ECG \u092e\u0947\u0902 leads II III aVF \u092e\u0947\u0902 ST elevation \u0926\u093f\u0916 \u0930\u0939\u093e \u0939\u0948 inferior wall MI \u0915\u093e \u0938\u0902\u0915\u0947\u0924",
               "\u092e\u0930\u0940\u091c\u093c \u0915\u093e troponin \u0914\u0930 D-dimer \u0938\u094d\u0924\u0930 \u092c\u0922\u093c\u093e \u0939\u0941\u0906 \u0939\u0948 \u0924\u0941\u0930\u0902\u0924 cardiology consultation \u091a\u093e\u0939\u093f\u090f",
               "CT scan \u092e\u0947\u0902 right temporal lobe \u092e\u0947\u0902 hypodense lesion midline shift \u0915\u0947 \u0938\u093e\u0925 \u0926\u093f\u0916 \u0930\u0939\u093e \u0939\u0948",
               "Arterial blood gas metabolic acidosis respiratory compensation pH 7.28 \u0926\u093f\u0916\u093e \u0930\u0939\u093e \u0939\u0948",
               "Chest X-ray bilateral infiltrates ARDS COVID positive \u092e\u0930\u0940\u091c\u093c \u092e\u0947\u0902 \u0926\u093f\u0916 \u0930\u0939\u093e \u0939\u0948"]
    }
}


def run_exp1_2():
    """Experiments 1-2: EFI battery with MiniLM + Hindi comparison"""
    print("Running Experiments 1-2: EFI Battery (MiniLM) + Hindi...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {
        "experiment": "Paper 8 Experiments 1-2: EFI Clinical Battery + Hindi Comparison",
        "model": "all-MiniLM-L6-v2 (384D)",
        "n_sentences": 15,
        "languages": {},
        "by_complexity": {},
        "statistical_tests": {},
        "confusion": {}
    }

    all_efi = {"kn": [], "ta": [], "hi": []}
    kn_ta_sims, kn_en_sims = [], []

    for complexity in ["simple", "medium", "complex"]:
        comp_efi = {"kn": [], "ta": [], "hi": []}
        for i in range(5):
            en_emb = model.encode(sentences[complexity]["en"][i])
            kn_emb = model.encode(sentences[complexity]["kn"][i])
            ta_emb = model.encode(sentences[complexity]["ta"][i])

            kn_ta_sims.append(float(1 - cosine(kn_emb, ta_emb)))
            kn_en_sims.append(float(1 - cosine(kn_emb, en_emb)))

            for lang in ["kn", "ta", "hi"]:
                lang_emb = model.encode(sentences[complexity][lang][i])
                efi = float(1 - cosine(en_emb, lang_emb))
                all_efi[lang].append(efi)
                comp_efi[lang].append(efi)

        results["by_complexity"][complexity] = {
            "kannada_mean": float(np.mean(comp_efi["kn"])),
            "tamil_mean": float(np.mean(comp_efi["ta"])),
            "hindi_mean": float(np.mean(comp_efi["hi"])),
            "pooled_non_english_mean": float(np.mean(comp_efi["kn"] + comp_efi["ta"] + comp_efi["hi"]))
        }

    for lang, name in [("kn", "Kannada"), ("ta", "Tamil"), ("hi", "Hindi")]:
        results["languages"][name] = {
            "efi_mean": float(np.mean(all_efi[lang])),
            "efi_std": float(np.std(all_efi[lang])),
            "efi_values": [round(v, 4) for v in all_efi[lang]],
            "n": len(all_efi[lang])
        }

    # Statistical tests
    en_efi = [1.0] * 15
    t_kn, p_kn = ttest_ind(en_efi, all_efi["kn"])
    t_ta, p_ta = ttest_ind(en_efi, all_efi["ta"])
    t_hi, p_hi = ttest_ind(en_efi, all_efi["hi"])
    u_kn_ta, p_kn_ta = mannwhitneyu(all_efi["kn"], all_efi["ta"])
    u_drav_indo, p_drav_indo = mannwhitneyu(all_efi["kn"] + all_efi["ta"], all_efi["hi"])

    results["statistical_tests"] = {
        "kannada_vs_english": {"t": round(float(t_kn), 2), "p": float(p_kn)},
        "tamil_vs_english": {"t": round(float(t_ta), 2), "p": float(p_ta)},
        "hindi_vs_english": {"t": round(float(t_hi), 2), "p": float(p_hi)},
        "kannada_vs_tamil": {"U": float(u_kn_ta), "p": round(float(p_kn_ta), 4)},
        "dravidian_vs_indoaryan": {"U": float(u_drav_indo), "p": round(float(p_drav_indo), 4)}
    }

    results["confusion"] = {
        "kannada_tamil_similarity_mean": round(float(np.mean(kn_ta_sims)), 4),
        "kannada_english_similarity_mean": round(float(np.mean(kn_en_sims)), 4),
        "confusion_ratio": round(float(np.mean(kn_ta_sims) / np.mean(kn_en_sims)), 2)
    }

    path = os.path.join(OUTPUT_DIR, "exp1_2_efi_battery.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    print(f"  EFI: Kn={np.mean(all_efi['kn']):.3f}, Ta={np.mean(all_efi['ta']):.3f}, Hi={np.mean(all_efi['hi']):.3f}")
    print(f"  Dravidian vs Indo-Aryan: p={p_drav_indo:.3f}")
    return model  # reuse for exp3


def run_exp3(minimodel):
    """Experiment 3: MPNet robustness check"""
    print("Running Experiment 3: MPNet Robustness...")
    mpnet = SentenceTransformer("all-mpnet-base-v2")

    results = {
        "experiment": "Paper 8 Experiment 3: Embedding Robustness (MPNet 768D vs MiniLM 384D)",
        "models": ["all-MiniLM-L6-v2 (384D)", "all-mpnet-base-v2 (768D)"],
        "mpnet_efi": {},
        "minilm_efi": {},
        "cross_model_correlation": {}
    }

    for lang in ["kn", "ta"]:
        mpnet_vals, minilm_vals = [], []
        lang_name = "Kannada" if lang == "kn" else "Tamil"
        for complexity in ["simple", "medium", "complex"]:
            for i in range(5):
                en_mp = mpnet.encode(sentences[complexity]["en"][i])
                lang_mp = mpnet.encode(sentences[complexity][lang][i])
                mpnet_vals.append(float(1 - cosine(en_mp, lang_mp)))

                en_ml = minimodel.encode(sentences[complexity]["en"][i])
                lang_ml = minimodel.encode(sentences[complexity][lang][i])
                minilm_vals.append(float(1 - cosine(en_ml, lang_ml)))

        from scipy.stats import pearsonr
        r, p = pearsonr(minilm_vals, mpnet_vals)
        results["mpnet_efi"][lang_name] = {
            "mean": round(float(np.mean(mpnet_vals)), 3),
            "values": [round(v, 4) for v in mpnet_vals]
        }
        results["minilm_efi"][lang_name] = {
            "mean": round(float(np.mean(minilm_vals)), 3),
            "values": [round(v, 4) for v in minilm_vals]
        }
        results["cross_model_correlation"][lang_name] = {
            "pearson_r": round(float(r), 3),
            "p_value": float(p)
        }

    path = os.path.join(OUTPUT_DIR, "exp3_mpnet_robustness.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")


def save_exp4_5():
    """Experiments 4-5: LLM results from prior runs (entered manually)"""
    print("Saving Experiments 4-5: LLM results from prior runs...")

    exp4 = {
        "experiment": "Paper 8 Experiment 4: LLM Language ID + Translation Fidelity",
        "models": ["DeepSeek V3.1 (671B MoE)", "Mistral Small 24B"],
        "api": "Together.ai",
        "temperature": 0.0,
        "language_identification": {
            "DeepSeek_V3.1": {"accuracy": 0.93, "note": "93% correct across all languages"},
            "Mistral_Small": {"accuracy": 1.00, "note": "100% correct across all languages"}
        },
        "translation_fidelity": {
            "DeepSeek_V3.1": {"Hindi": 0.928, "Tamil": 0.895, "Kannada": 0.807},
            "Mistral_Small": {"Hindi": 0.922, "Kannada": 0.816, "Tamil": 0.655}
        },
        "catastrophic_failure": {
            "model": "Mistral Small 24B",
            "language": "Tamil",
            "input": "Pregnant woman with high blood pressure and swelling in feet",
            "output": "A woman with a bellyache is crying loudly and screaming",
            "cosine_similarity": 0.13,
            "note": "Model identifies language correctly but fundamentally misrepresents clinical content"
        }
    }

    exp5 = {
        "experiment": "Paper 8 Experiment 5: Multi-Trial Clinical Variance",
        "models": ["DeepSeek V3.1 (671B MoE)", "Mistral Small 24B"],
        "api": "Together.ai",
        "temperature": 0.7,
        "n_trials": 5,
        "n_scenarios": 5,
        "variance_ratio": {
            "DeepSeek_V3.1": {
                "English": 1.00, "Hindi": 2.10, "Kannada": 2.47, "Tamil": 2.85,
                "p_values": {"Hindi": 0.011, "Kannada": 0.008, "Tamil": 0.004}
            },
            "Mistral_Small": {
                "English": 1.00, "Hindi": 2.23, "Kannada": 2.07, "Tamil": 2.67,
                "p_values": {"Hindi": 0.029, "Kannada": 0.029, "Tamil": 0.029}
            }
        },
        "response_fidelity_centroid_similarity": {
            "DeepSeek_V3.1": {"Hindi": 0.890, "Kannada": 0.811, "Tamil": 0.764},
            "Mistral_Small": {"Hindi": 0.910, "Kannada": 0.861, "Tamil": 0.837}
        }
    }

    for name, data in [("exp4_language_translation_accuracy.json", exp4), ("exp5_multitrial_variance.json", exp5)]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 8: Saving All Experiment Results")
    print("=" * 60)
    minimodel = run_exp1_2()
    print()
    run_exp3(minimodel)
    print()
    save_exp4_5()
    print()
    print("=" * 60)
    print("All results saved to:", OUTPUT_DIR)
    print("=" * 60)
