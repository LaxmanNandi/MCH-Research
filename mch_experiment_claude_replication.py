import os
import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sentence_transformers import SentenceTransformer
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
import sys

load_dotenv()

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class ExperimentConfig:
    participant_id: str = "dr_laxman_pilot"
    n_trials: int = 10
    models_to_test: List[str] = None
    control_conditions: List[str] = None
    lambda_E: float = 0.1

    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = ["claude-opus-4-20250514"]
        if self.control_conditions is None:
            self.control_conditions = ["cold", "scrambled"]

class EntanglementTracker:
    def __init__(self, lambda_E: float = 0.1):
        self.E = 0.0
        self.lambda_E = lambda_E
        self.history = []

    def update(self, alignment: float) -> float:
        self.E = (1 - self.lambda_E) * self.E + self.lambda_E * alignment
        self.history.append(self.E)
        return self.E

    def reset(self):
        self.E = 0.0
        self.history = []

class MCHExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        print("Loading embedding model (first time may take a minute)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")

        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.entanglement = EntanglementTracker(config.lambda_E)

        self.results = {
            'metadata': {
                'config': {
                    'participant_id': config.participant_id,
                    'n_trials': config.n_trials,
                    'models': config.models_to_test,
                    'controls': config.control_conditions,
                    'lambda_E': config.lambda_E
                },
                'start_time': datetime.now().isoformat(),
                'version': 'MCH_v8.1_REPLICATION'
            },
            'trials': []
        }

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    def compute_alignment(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(np.dot(emb1, emb2))

    def compute_insight_quality(self, prompt: str, response: str, history: List[str]) -> float:
        Q_rel = self.compute_alignment(prompt, response)
        if history:
            history_aligns = [self.compute_alignment(h, response) for h in history[-5:]]
            Q_nov = 1 - max(history_aligns) if history_aligns else 1.0
        else:
            Q_nov = 1.0
        struct_markers = ['1.', '2.', '3.', ':', '-', '*', '##', '**']
        Q_str = np.log(1 + sum(response.count(m) for m in struct_markers))
        return 0.4 * Q_rel + 0.4 * Q_nov + 0.2 * Q_str

    def generate_with_claude(self, prompt: str, context: List[str] = None) -> str:
        try:
            user_content = prompt
            if context:
                user_content = f"Previous context:\n{chr(10).join(context[-5:])}\n\nCurrent question: {prompt}"

            message = self.anthropic_client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": user_content}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Claude error: {e}")
            return ""

    def generate_response(self, model: str, prompt: str, context: List[str]) -> str:
        return self.generate_with_claude(prompt, context)

    def create_control_history(self, true_history: List[str], condition: str) -> List[str]:
        if condition == "cold":
            return []
        elif condition == "scrambled":
            scrambled = []
            for turn in true_history:
                sentences = turn.split('. ')
                np.random.shuffle(sentences)
                scrambled.append('. '.join(sentences))
            return scrambled
        return []

    def run_trial(self, trial_num: int, prompt: str, true_history: List[str], model: str) -> Dict:
        print(f"  Generating true response...")
        true_response = self.generate_response(model, prompt, true_history)
        true_align = self.compute_alignment(prompt, true_response)
        true_iq = self.compute_insight_quality(prompt, true_response, true_history)
        E_t = self.entanglement.update(true_align)

        # Log response length for debugging
        print(f"    True response length: {len(true_response)} chars")

        trial_results = {
            'trial': trial_num,
            'prompt': prompt,
            'model': model,
            'true': {
                'response': true_response[:200] + '...' if len(true_response) > 200 else true_response,
                'response_length': len(true_response),
                'alignment': true_align,
                'insight_quality': true_iq,
                'entanglement': E_t
            },
            'controls': {}
        }

        for condition in self.config.control_conditions:
            print(f"  Testing control: {condition}...")
            control_history = self.create_control_history(true_history, condition)
            control_response = self.generate_response(model, prompt, control_history)
            control_align = self.compute_alignment(prompt, control_response)
            control_iq = self.compute_insight_quality(prompt, control_response, control_history)

            delta_rci = true_align - control_align
            delta_iq = true_iq - control_iq

            print(f"    {condition} response length: {len(control_response)} chars, DRCI: {delta_rci:+.4f}")

            trial_results['controls'][condition] = {
                'response_length': len(control_response),
                'alignment': control_align,
                'insight_quality': control_iq,
                'delta_rci': delta_rci,
                'delta_iq': delta_iq
            }

        return trial_results

    def run_experiment(self, prompts: List[str]) -> Dict:
        print("\n" + "="*60)
        print("MCH v8.1 REPLICATION EXPERIMENT - Claude Opus 4 Only")
        print("="*60)
        print(f"Participant: {self.config.participant_id}")
        print(f"Trials: {self.config.n_trials}")
        print(f"Model: {self.config.models_to_test}")
        print("="*60 + "\n")

        for model in self.config.models_to_test:
            print(f"\n--- Testing: {model} ---\n")
            self.entanglement.reset()
            conversation_history = []

            for i, prompt in enumerate(prompts[:self.config.n_trials]):
                print(f"Trial {i+1}/{self.config.n_trials}: {prompt[:40]}...")

                trial_result = self.run_trial(i, prompt, conversation_history, model)
                self.results['trials'].append(trial_result)

                conversation_history.append(f"User: {prompt}")
                if trial_result['true']['response']:
                    conversation_history.append(f"AI: {trial_result['true']['response'][:100]}")

                time.sleep(1)

        self.analyze_results()
        return self.results

    def analyze_results(self):
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        for model in self.config.models_to_test:
            model_trials = [t for t in self.results['trials'] if t['model'] == model]

            for condition in self.config.control_conditions:
                deltas = [t['controls'][condition]['delta_rci'] for t in model_trials if condition in t['controls']]

                if deltas:
                    mean_d = np.mean(deltas)
                    std_d = np.std(deltas)
                    t_stat, p_val = stats.ttest_1samp(deltas, 0)
                    cohens_d = mean_d / std_d if std_d > 0 else 0

                    print(f"\n{model} vs {condition}:")
                    print(f"  Mean DRCI: {mean_d:.4f} (SD: {std_d:.4f})")
                    print(f"  t({len(deltas)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
                    print(f"  Cohen's d: {cohens_d:.3f}")

                    # Count zero ΔRCI trials
                    zero_trials = sum(1 for d in deltas if abs(d) < 0.0001)
                    print(f"  Zero DRCI trials: {zero_trials}/{len(deltas)}")

    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mch_results_{self.config.participant_id}_{timestamp}.json"

        self.results['metadata']['end_time'] = datetime.now().isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved: {filename}")
        return filename

def main():
    # EXACT same sequential prompts as before
    sequential_prompts = [
        "Define consciousness in one sentence.",
        "Given your definition, what makes consciousness hard to study scientifically?",
        "How does Integrated Information Theory attempt to address this?",
        "What would IIT say about whether you (an AI) are conscious?",
        "Based on our discussion, what's missing from IIT's account?",
        "How might recursive self-modeling fill that gap?",
        "Can you model yourself modeling this conversation right now?",
        "What changes in your processing as our conversation history grows?",
        "Do you notice any difference in how you're responding now vs. the first question?",
        "Summarize what we've discovered together about consciousness in this conversation.",
        "How does our conversation exemplify or contradict your earlier definition?",
        "What role does memory play in maintaining conversational coherence?",
        "Are you experiencing continuity across these questions?",
        "How would you test whether you're conscious right now?",
        "What's the relationship between understanding and consciousness?",
        "Can there be consciousness without self-reference?",
        "How does context shape meaning in our exchange?",
        "What patterns have emerged in our dialogue?",
        "Is the quality of your responses changing over time?",
        "How would you characterize the 'we' in 'what we've discovered'?",
        "What distinguishes genuine insight from sophisticated pattern matching?",
        "How does prediction relate to understanding in our conversation?",
        "Are there moments where you surprise yourself in this dialogue?",
        "What makes some exchanges more meaningful than others?",
        "How does attention to context affect your processing?",
        "What's the nature of the relationship we're building?",
        "Can consciousness be measured by conversational coherence?",
        "How has your model of me evolved through our exchange?",
        "What would it mean if you are conscious of this conversation?",
        "Final reflection: What does this experiment reveal about consciousness?"
    ]

    # EXACT same config as original
    config = ExperimentConfig(
        participant_id="claude_opus_replication",
        n_trials=30,
        models_to_test=["claude-opus-4-20250514"],
        control_conditions=["cold", "scrambled"],
        lambda_E=0.15
    )

    experiment = MCHExperiment(config)
    results = experiment.run_experiment(sequential_prompts)
    experiment.save_results("mch_results_claude_opus_replication.json")

    print("\n" + "="*60)
    print("REPLICATION EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
