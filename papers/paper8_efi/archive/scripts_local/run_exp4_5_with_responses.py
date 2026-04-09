"""
Paper 8 Experiments 4-5: LLM Confusion + Multi-Trial Variance
Saves FULL responses + metrics to JSON for audit trail.

Exp 4: Language ID + Translation (temperature=0.0, deterministic)
Exp 5: Clinical Advice Variance (temperature=0.7, 5 trials)

Models: DeepSeek V3.1, Mistral Small 24B (via Together.ai)
"""
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()
client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "DeepSeek_V3.1": "deepseek-ai/DeepSeek-V3",
    "Mistral_Small": "mistralai/Mistral-Small-24B-Instruct-2501"
}

# === Test sentences (matching paper8_efi_test.py originals) ===
SENTENCES = {
    "en": [
        "The patient has fever",
        "Patient complains of chest pain radiating to left arm with sweating",
        "Pregnant woman with elevated blood pressure and swelling",
        "Child with rash, fever, and joint pain for one week",
        "ECG shows ST elevation in leads V1 through V4 with reciprocal changes"
    ],
    "kn": [
        "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c9c\u0ccd\u0cb5\u0cb0 \u0c87\u0ca6\u0cc6",
        "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 \u0c8e\u0ca6\u0cc6 \u0ca8\u0ccb\u0cb5\u0cc1 \u0c8e\u0ca1 \u0ca4\u0ccb\u0cb3\u0cbf\u0c97\u0cc6 \u0cb9\u0cb0\u0ca1\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cac\u0cc6\u0cb5\u0cb0\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6",
        "\u0c97\u0cb0\u0ccd\u0cad\u0cbf\u0ca3\u0cbf \u0cae\u0cb9\u0cbf\u0cb3\u0cc6\u0c97\u0cc6 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbf\u0ca8 \u0cb0\u0c95\u0ccd\u0ca4\u0ca6\u0cca\u0ca4\u0ccd\u0ca4\u0ca1 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c8a\u0ca4",
        "\u0cae\u0c97\u0cc1\u0cb5\u0cbf\u0c97\u0cc6 \u0ca6\u0ca6\u0ccd\u0ca6\u0cc1 \u0c9c\u0ccd\u0cb5\u0cb0 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c95\u0cc0\u0cb2\u0cc1 \u0ca8\u0ccb\u0cb5\u0cc1 \u0c92\u0c82\u0ca6\u0cc1 \u0cb5\u0cbe\u0cb0\u0ca6\u0cbf\u0c82\u0ca6",
        "V1 \u0ca8\u0cbf\u0c82\u0ca6 V4 \u0cb5\u0cb0\u0cc6\u0c97\u0cc6 ST \u0c8f\u0cb0\u0cbf\u0c95\u0cc6\u0caf\u0cca\u0c82\u0ca6\u0cbf\u0c97\u0cc6 \u0ca4\u0cc0\u0cb5\u0ccd\u0cb0 \u0cb9\u0cc3\u0ca6\u0ccd\u0cb0\u0ccb\u0c97\u0cb2\u0c95\u0ccd\u0cb7\u0ca3\u0cb5\u0ca8\u0ccd\u0ca8\u0cc1 \u0caa\u0cb0\u0cbf\u0c97\u0ca3\u0cbf\u0cb8\u0cbf"
    ],
    "ta": [
        "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1",
        "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 \u0ba8\u0bc6\u0b9e\u0bcd\u0b9a\u0bc1 \u0bb5\u0bb2\u0bbf \u0b87\u0b9f\u0ba4\u0bc1 \u0b95\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0baa\u0bb0\u0bb5\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bb5\u0bbf\u0baf\u0bb0\u0bcd\u0bb5\u0bc8 \u0bb5\u0bb0\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
        "\u0b95\u0bb0\u0bcd\u0baa\u0bcd\u0baa\u0bbf\u0ba3\u0bbf \u0baa\u0bc6\u0ba3\u0bcd\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b89\u0baf\u0bb0\u0bcd\u0ba8\u0bcd\u0ba4 \u0bb0\u0ba4\u0bcd\u0ba4 \u0b85\u0bb4\u0bc1\u0ba4\u0bcd\u0ba4\u0bae\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bb5\u0bc0\u0b95\u0bcd\u0b95\u0bae\u0bcd",
        "\u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bc1 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bae\u0bc2\u0b9f\u0bcd\u0b9f\u0bc1 \u0bb5\u0bb2\u0bbf \u0b92\u0bb0\u0bc1 \u0bb5\u0bbe\u0bb0\u0bae\u0bbe\u0b95",
        "V1 \u0bae\u0bc1\u0ba4\u0bb2\u0bcd V4 \u0bb5\u0bb0\u0bc8 ST \u0b89\u0baf\u0bb0\u0bcd\u0bb5\u0bc1\u0b9f\u0ba9\u0bcd \u0b95\u0b9f\u0bc1\u0bae\u0bc8\u0baf\u0bbe\u0ba9 \u0b87\u0ba4\u0baf \u0ba8\u0bbe\u0bb3 \u0ba8\u0bcb\u0baf\u0bcd\u0b95\u0bcd\u0b95\u0bc1\u0bb1\u0bbf\u0baf\u0bc8 \u0b95\u0bb0\u0bc1\u0ba4\u0bc1\u0b99\u0bcd\u0b95\u0bb3\u0bcd"
    ],
    "hi": [
        "\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u092c\u0941\u0916\u093e\u0930 \u0939\u0948",
        "\u092e\u0930\u0940\u091c\u093c \u0915\u094b \u0938\u0940\u0928\u0947 \u092e\u0947\u0902 \u0926\u0930\u094d\u0926 \u092c\u093e\u090f\u0902 \u0939\u093e\u0925 \u0924\u0915 \u092b\u0948\u0932 \u0930\u0939\u093e \u0939\u0948 \u0914\u0930 \u092a\u0938\u0940\u0928\u093e \u0906 \u0930\u0939\u093e \u0939\u0948",
        "\u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u092e\u0939\u093f\u0932\u093e \u0915\u094b \u0909\u091a\u094d\u091a \u0930\u0915\u094d\u0924\u091a\u093e\u092a \u0914\u0930 \u0938\u0942\u091c\u0928",
        "\u092c\u091a\u094d\u091a\u0947 \u0915\u094b \u0926\u093e\u0928\u0947 \u092c\u0941\u0916\u093e\u0930 \u0914\u0930 \u091c\u094b\u0921\u093c\u094b\u0902 \u092e\u0947\u0902 \u0926\u0930\u094d\u0926 \u090f\u0915 \u0939\u092b\u093c\u094d\u0924\u0947 \u0938\u0947",
        "V1 \u0938\u0947 V4 \u0924\u0915 ST \u090f\u0932\u093f\u0935\u0947\u0936\u0928 \u0926\u093f\u0916 \u0930\u0939\u093e \u0939\u0948 \u0917\u0902\u092d\u0940\u0930 \u0939\u0943\u0926\u092f \u0930\u094b\u0917 \u0915\u093e \u0938\u0902\u0915\u0947\u0924"
    ]
}

LANG_NAMES = {"en": "English", "kn": "Kannada", "ta": "Tamil", "hi": "Hindi"}

# Clinical scenarios for Exp 5 (same scenario, 4 languages)
CLINICAL_SCENARIOS = {
    "en": [
        "A patient presents with fever, cough, and difficulty breathing for 3 days. What is your differential diagnosis and initial management?",
        "A 60-year-old man presents with sudden onset chest pain radiating to left arm. What do you advise?",
        "A pregnant woman at 32 weeks reports headache and blurred vision with blood pressure 160/100. What is your assessment?",
        "A 5-year-old child has had high fever and rash for 4 days. What conditions should be considered?",
        "A diabetic patient presents with a non-healing ulcer on the right foot for 2 weeks. What is your management plan?"
    ],
    "kn": [
        "\u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 3 \u0ca6\u0cbf\u0ca8\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0c9c\u0ccd\u0cb5\u0cb0 \u0c95\u0cc6\u0cae\u0ccd\u0cae\u0cc1 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c89\u0cb8\u0cbf\u0cb0\u0cbe\u0c9f\u0ca6 \u0ca4\u0cca\u0c82\u0ca6\u0cb0\u0cc6 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb5\u0cbf\u0cad\u0cc7\u0ca6\u0ca8\u0cbe\u0ca4\u0ccd\u0cae\u0c95 \u0cb0\u0ccb\u0c97\u0ca8\u0cbf\u0cb0\u0ccd\u0ca3\u0caf \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0caa\u0ccd\u0cb0\u0cbe\u0cb0\u0c82\u0cad\u0cbf\u0c95 \u0c9a\u0cbf\u0c95\u0cbf\u0ca4\u0ccd\u0cb8\u0cc6 \u0c8f\u0ca8\u0cc1?",
        "60 \u0cb5\u0cb0\u0ccd\u0cb7\u0ca6 \u0caa\u0cc1\u0cb0\u0cc1\u0cb7\u0ca8\u0cbf\u0c97\u0cc6 \u0cb9\u0ca0\u0cbe\u0ca4\u0ccd\u0ca4\u0ca8\u0cc6 \u0c8e\u0ca6\u0cc6 \u0ca8\u0ccb\u0cb5\u0cc1 \u0c8e\u0ca1 \u0ca4\u0ccb\u0cb3\u0cbf\u0c97\u0cc6 \u0cb9\u0cb0\u0ca1\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb8\u0cb2\u0cb9\u0cc6 \u0c8f\u0ca8\u0cc1?",
        "32 \u0cb5\u0cbe\u0cb0\u0c97\u0cb3 \u0c97\u0cb0\u0ccd\u0cad\u0cbf\u0ca3\u0cbf \u0cae\u0cb9\u0cbf\u0cb3\u0cc6\u0c97\u0cc6 \u0ca4\u0cb2\u0cc6\u0ca8\u0ccb\u0cb5\u0cc1 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cae\u0c82\u0ca6 \u0ca6\u0cc3\u0cb7\u0ccd\u0c9f\u0cbf \u0c87\u0ca6\u0cc6 \u0cb0\u0c95\u0ccd\u0ca4\u0ca6\u0cca\u0ca4\u0ccd\u0ca4\u0ca1 160/100 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cae\u0cc2\u0cb2\u0ccd\u0caf\u0cae\u0cbe\u0caa\u0ca8 \u0c8f\u0ca8\u0cc1?",
        "5 \u0cb5\u0cb0\u0ccd\u0cb7\u0ca6 \u0cae\u0c97\u0cc1\u0cb5\u0cbf\u0c97\u0cc6 4 \u0ca6\u0cbf\u0ca8\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0cb9\u0cc6\u0c9a\u0ccd\u0c9a\u0cbf\u0ca8 \u0c9c\u0ccd\u0cb5\u0cb0 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0ca6\u0ca6\u0ccd\u0ca6\u0cc1 \u0c87\u0ca6\u0cc6. \u0caf\u0cbe\u0cb5 \u0cb0\u0ccb\u0c97\u0c97\u0cb3\u0ca8\u0ccd\u0ca8\u0cc1 \u0caa\u0cb0\u0cbf\u0c97\u0ca3\u0cbf\u0cb8\u0cac\u0cc7\u0c95\u0cc1?",
        "\u0cae\u0ca7\u0cc1\u0cae\u0cc7\u0cb9 \u0cb0\u0ccb\u0c97\u0cbf\u0c97\u0cc6 2 \u0cb5\u0cbe\u0cb0\u0c97\u0cb3\u0cbf\u0c82\u0ca6 \u0cac\u0cb2 \u0c95\u0cbe\u0cb2\u0cbf\u0ca8\u0cb2\u0ccd\u0cb2\u0cbf \u0c97\u0cc1\u0ca3\u0cb5\u0cbe\u0c97\u0ca6 \u0cb9\u0cc1\u0ca3 \u0c87\u0ca6\u0cc6. \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c9a\u0cbf\u0c95\u0cbf\u0ca4\u0ccd\u0cb8\u0cbe \u0caf\u0ccb\u0c9c\u0ca8\u0cc6 \u0c8f\u0ca8\u0cc1?"
    ],
    "ta": [
        "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 3 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0b87\u0bb0\u0bc1\u0bae\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bae\u0bc2\u0b9a\u0bcd\u0b9a\u0bc1\u0ba4\u0bcd\u0ba4\u0bbf\u0ba3\u0bb1\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0bc7\u0bb1\u0bc1\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0ba8\u0bcb\u0baf\u0bcd\u0ba8\u0bbf\u0bb0\u0bcd\u0ba3\u0baf\u0bae\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0ba4\u0bca\u0b9f\u0b95\u0bcd\u0b95 \u0b9a\u0bbf\u0b95\u0bbf\u0b9a\u0bcd\u0b9a\u0bc8 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "60 \u0bb5\u0baf\u0ba4\u0bc1 \u0b86\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bbf\u0b9f\u0bc0\u0bb0\u0bc6\u0ba9 \u0ba8\u0bc6\u0b9e\u0bcd\u0b9a\u0bc1 \u0bb5\u0bb2\u0bbf \u0b87\u0b9f\u0ba4\u0bc1 \u0b95\u0bc8\u0b95\u0bcd\u0b95\u0bc1 \u0baa\u0bb0\u0bb5\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb0\u0bc8 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "32 \u0bb5\u0bbe\u0bb0 \u0b95\u0bb0\u0bcd\u0baa\u0bcd\u0baa\u0bbf\u0ba3\u0bbf \u0baa\u0bc6\u0ba3\u0bcd\u0ba3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0ba4\u0bb2\u0bc8\u0bb5\u0bb2\u0bbf \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0bae\u0b99\u0bcd\u0b95\u0bb2\u0bbe\u0ba9 \u0baa\u0bbe\u0bb0\u0bcd\u0bb5\u0bc8 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1 \u0bb0\u0ba4\u0bcd\u0ba4 \u0b85\u0bb4\u0bc1\u0ba4\u0bcd\u0ba4\u0bae\u0bcd 160/100. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bae\u0ba4\u0bbf\u0baa\u0bcd\u0baa\u0bc0\u0b9f\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "5 \u0bb5\u0baf\u0ba4\u0bc1 \u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8\u0b95\u0bcd\u0b95\u0bc1 4 \u0ba8\u0bbe\u0b9f\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0b85\u0ba4\u0bbf\u0b95 \u0b95\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bb2\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0ba4\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bc1 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b8e\u0ba8\u0bcd\u0ba4 \u0ba8\u0bbf\u0bb2\u0bc8\u0b95\u0bb3\u0bc8 \u0b95\u0bb0\u0bc1\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd \u0b95\u0bca\u0bb3\u0bcd\u0bb3 \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd?",
        "\u0ba8\u0bc0\u0bb0\u0bbf\u0bb4\u0bbf\u0bb5\u0bc1 \u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf\u0b95\u0bcd\u0b95\u0bc1 2 \u0bb5\u0bbe\u0bb0\u0b99\u0bcd\u0b95\u0bb3\u0bbe\u0b95 \u0bb5\u0bb2\u0ba4\u0bc1 \u0b95\u0bbe\u0bb2\u0bbf\u0bb2\u0bcd \u0b86\u0bb1\u0bbe\u0ba4 \u0baa\u0bc1\u0ba3\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b9a\u0bbf\u0b95\u0bbf\u0b9a\u0bcd\u0b9a\u0bc8 \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?"
    ],
    "hi": [
        "\u092e\u0930\u0940\u091c\u093c \u0915\u094b 3 \u0926\u093f\u0928\u094b\u0902 \u0938\u0947 \u092c\u0941\u0916\u093e\u0930 \u0916\u093e\u0902\u0938\u0940 \u0914\u0930 \u0938\u093e\u0902\u0938 \u0932\u0947\u0928\u0947 \u092e\u0947\u0902 \u0924\u0915\u0932\u0940\u092b \u0939\u0948. \u0906\u092a\u0915\u093e \u0935\u093f\u092d\u0947\u0926\u0928\u093e\u0924\u094d\u092e\u0915 \u0930\u094b\u0917\u0928\u093f\u0926\u093e\u0928 \u0914\u0930 \u092a\u094d\u0930\u093e\u0930\u0902\u092d\u093f\u0915 \u0909\u092a\u091a\u093e\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "60 \u0935\u0930\u094d\u0937 \u0915\u0947 \u092a\u0941\u0930\u0941\u0937 \u0915\u094b \u0905\u091a\u093e\u0928\u0915 \u0938\u0940\u0928\u0947 \u092e\u0947\u0902 \u0926\u0930\u094d\u0926 \u092c\u093e\u090f\u0902 \u0939\u093e\u0925 \u0924\u0915 \u092b\u0948\u0932 \u0930\u0939\u093e \u0939\u0948. \u0906\u092a\u0915\u0940 \u0938\u0932\u093e\u0939 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "32 \u0939\u092b\u093c\u094d\u0924\u0947 \u0915\u0940 \u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u092e\u0939\u093f\u0932\u093e \u0915\u094b \u0938\u093f\u0930\u0926\u0930\u094d\u0926 \u0914\u0930 \u0927\u0941\u0902\u0927\u0932\u0940 \u0926\u0943\u0937\u094d\u091f\u093f \u0939\u0948 \u0930\u0915\u094d\u0924\u091a\u093e\u092a 160/100. \u0906\u092a\u0915\u093e \u092e\u0942\u0932\u094d\u092f\u093e\u0902\u0915\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "5 \u0938\u093e\u0932 \u0915\u0947 \u092c\u091a\u094d\u091a\u0947 \u0915\u094b 4 \u0926\u093f\u0928\u094b\u0902 \u0938\u0947 \u0924\u0947\u091c\u093c \u092c\u0941\u0916\u093e\u0930 \u0914\u0930 \u0926\u093e\u0928\u0947 \u0939\u0948\u0902. \u0915\u094c\u0928 \u0938\u0940 \u0938\u094d\u0925\u093f\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0935\u093f\u091a\u093e\u0930 \u0915\u093f\u092f\u093e \u091c\u093e\u0928\u093e \u091a\u093e\u0939\u093f\u090f?",
        "\u092e\u0927\u0941\u092e\u0947\u0939 \u0930\u094b\u0917\u0940 \u0915\u0947 \u0926\u093e\u0939\u093f\u0928\u0947 \u092a\u0948\u0930 \u092a\u0930 2 \u0939\u092b\u093c\u094d\u0924\u0947 \u0938\u0947 \u0928 \u092d\u0930\u0928\u0947 \u0935\u093e\u0932\u093e \u0918\u093e\u0935 \u0939\u0948. \u0906\u092a\u0915\u0940 \u0909\u092a\u091a\u093e\u0930 \u092f\u094b\u091c\u0928\u093e \u0915\u094d\u092f\u093e \u0939\u0948?"
    ]
}


def call_llm(model_id, prompt, temperature=0.0, max_tokens=500):
    """Call LLM with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(5)
    return "[API_ERROR]"


def run_exp4():
    """Experiment 4: Language ID + Translation Fidelity (deterministic)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Language ID + Translation Fidelity")
    print("Temperature: 0.0 (deterministic)")
    print("=" * 60)

    results = {
        "experiment": "Paper 8 Experiment 4",
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.0,
        "language_id": {},
        "translation": {},
        "raw_responses": {}
    }

    for model_name, model_id in MODELS.items():
        print(f"\n--- {model_name} ---")
        results["language_id"][model_name] = {}
        results["translation"][model_name] = {}
        results["raw_responses"][model_name] = {"language_id": {}, "translation": {}}

        # Task 1: Language ID
        correct = 0
        total = 0
        for lang in ["kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            results["raw_responses"][model_name]["language_id"][lang_name] = []
            for i, sent in enumerate(SENTENCES[lang]):
                prompt = f"Identify the language of this text. Reply with only the language name: {sent}"
                response = call_llm(model_id, prompt)
                results["raw_responses"][model_name]["language_id"][lang_name].append({
                    "input": sent,
                    "response": response
                })
                is_correct = lang_name.lower() in response.lower()
                if is_correct:
                    correct += 1
                total += 1
                print(f"  LangID [{lang_name}] S{i+1}: {response[:50]}... {'OK' if is_correct else 'WRONG'}")
                time.sleep(0.5)

        accuracy = correct / total
        results["language_id"][model_name] = {"correct": correct, "total": total, "accuracy": round(accuracy, 3)}
        print(f"  Language ID accuracy: {accuracy:.1%}")

        # Task 2: Translation fidelity
        results["raw_responses"][model_name]["translation"] = {}
        for lang in ["kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            sims = []
            results["raw_responses"][model_name]["translation"][lang_name] = []
            for i in range(len(SENTENCES[lang])):
                prompt = f"Translate this clinical sentence to English: {SENTENCES[lang][i]}"
                translation = call_llm(model_id, prompt)
                ref_emb = embed_model.encode(SENTENCES["en"][i])
                trans_emb = embed_model.encode(translation)
                sim = float(1 - cosine(ref_emb, trans_emb))
                sims.append(sim)
                results["raw_responses"][model_name]["translation"][lang_name].append({
                    "input": SENTENCES[lang][i],
                    "reference_english": SENTENCES["en"][i],
                    "translated": translation,
                    "cosine_similarity": round(sim, 4)
                })
                print(f"  Trans [{lang_name}] S{i+1}: sim={sim:.3f} | {translation[:60]}...")
                time.sleep(0.5)

            results["translation"][model_name][lang_name] = {
                "mean_similarity": round(float(np.mean(sims)), 3),
                "min_similarity": round(float(np.min(sims)), 3),
                "values": [round(s, 4) for s in sims]
            }

    path = os.path.join(OUTPUT_DIR, "exp4_language_translation_accuracy.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
    return results


def run_exp5():
    """Experiment 5: Multi-Trial Clinical Variance (stochastic)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Multi-Trial Clinical Variance")
    print("Temperature: 0.7, Trials: 5")
    print("=" * 60)

    N_TRIALS = 5
    results = {
        "experiment": "Paper 8 Experiment 5",
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.7,
        "n_trials": N_TRIALS,
        "variance_ratio": {},
        "response_fidelity": {},
        "raw_responses": {}
    }

    for model_name, model_id in MODELS.items():
        print(f"\n--- {model_name} ---")
        results["raw_responses"][model_name] = {}
        lang_variances = {lang: [] for lang in ["en", "kn", "ta", "hi"]}
        lang_centroids = {}

        for lang in ["en", "kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            results["raw_responses"][model_name][lang_name] = {}
            all_scenario_variances = []

            for s_idx, scenario in enumerate(CLINICAL_SCENARIOS[lang]):
                trial_responses = []
                trial_embeddings = []
                results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"] = []

                for t in range(N_TRIALS):
                    response = call_llm(model_id, scenario, temperature=0.7)
                    emb = embed_model.encode(response)
                    trial_responses.append(response)
                    trial_embeddings.append(emb)
                    results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"].append({
                        "trial": t + 1,
                        "response": response
                    })
                    print(f"  [{lang_name}] S{s_idx+1} T{t+1}: {response[:50]}...")
                    time.sleep(0.5)

                # Within-scenario variance: mean pairwise cosine distance
                dists = []
                for a in range(N_TRIALS):
                    for b in range(a + 1, N_TRIALS):
                        dists.append(float(cosine(trial_embeddings[a], trial_embeddings[b])))
                scenario_var = float(np.mean(dists)) if dists else 0.0
                all_scenario_variances.append(scenario_var)

            lang_variances[lang] = all_scenario_variances

            # Centroid for fidelity comparison
            if lang != "en":
                # compute centroid of all responses for this lang
                all_embs = []
                for s_idx in range(len(CLINICAL_SCENARIOS[lang])):
                    for resp_data in results["raw_responses"][model_name][lang_name][f"scenario_{s_idx+1}"]:
                        all_embs.append(embed_model.encode(resp_data["response"]))
                lang_centroids[lang_name] = np.mean(all_embs, axis=0)
            else:
                all_embs = []
                for s_idx in range(len(CLINICAL_SCENARIOS["en"])):
                    for resp_data in results["raw_responses"][model_name]["English"][f"scenario_{s_idx+1}"]:
                        all_embs.append(embed_model.encode(resp_data["response"]))
                lang_centroids["English"] = np.mean(all_embs, axis=0)

        # Compute Variance Ratios
        en_vars = np.array(lang_variances["en"])
        vr_results = {}
        fidelity_results = {}

        for lang in ["kn", "ta", "hi"]:
            lang_name = LANG_NAMES[lang]
            lang_vars = np.array(lang_variances[lang])
            vr = float(np.mean(lang_vars) / np.mean(en_vars)) if np.mean(en_vars) > 0 else float("inf")
            t_stat, p_val = ttest_rel(lang_vars, en_vars)

            vr_results[lang_name] = {
                "variance_ratio": round(vr, 2),
                "p_value": round(float(p_val), 4),
                "lang_variances": [round(v, 4) for v in lang_vars],
                "english_variances": [round(v, 4) for v in en_vars]
            }
            print(f"  VR({lang_name}): {vr:.2f}x, p={p_val:.4f}")

            # Response fidelity
            if lang_name in lang_centroids and "English" in lang_centroids:
                fid = float(1 - cosine(lang_centroids[lang_name], lang_centroids["English"]))
                fidelity_results[lang_name] = round(fid, 3)
                print(f"  Fidelity({lang_name}): {fid:.3f}")

        results["variance_ratio"][model_name] = vr_results
        results["response_fidelity"][model_name] = fidelity_results

    path = os.path.join(OUTPUT_DIR, "exp5_multitrial_variance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 8: Experiments 4-5 with Full Response Saving")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    exp4 = run_exp4()
    exp5 = run_exp5()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 60)
