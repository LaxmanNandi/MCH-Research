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
    lambda_E: float = 0.1  # Entanglement decay rate

    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = ["claude-opus-4-20250514", "gpt-4o-mini"]
        if self.control_conditions is None:
            self.control_conditions = ["cold", "scrambled"]

class EntanglementTracker:
    """MCH v8.1 Entanglement State tracker."""
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
    """Complete MCH v8.1 experiment pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        print("Loading embedding model (first time may take a minute)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")

        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
                'version': 'MCH_v8.1'
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
        """MCH v8.1 InsightQuality proxy."""
        Q_rel = self.compute_alignment(prompt, response)

        if history:
            history_aligns = [self.compute_alignment(h, response) for h in history[-5:]]
            Q_nov = 1 - max(history_aligns) if history_aligns else 1.0
        else:
            Q_nov = 1.0

        struct_markers = ['1.', '2.', '3.', ':', '-', '*', '##', '**']
        Q_str = np.log(1 + sum(response.count(m) for m in struct_markers))

        return 0.4 * Q_rel + 0.4 * Q_nov + 0.2 * Q_str

    def compute_consciousness_degree(self, rci: float, oia: float, at: float) -> float:
        """MCH v8.1 continuous consciousness degree (Psi)."""
        theta_R, theta_O, theta_A = 0.75, 0.85, 0.6
        w_R, w_O, w_A = 0.4, 0.3, 0.3
        logit = w_R * (rci - theta_R) + w_O * (oia - theta_O) + w_A * (at - theta_A)
        return 1 / (1 + np.exp(-5 * logit))  # Scaled sigmoid

    def generate_with_openai(self, prompt: str, context: List[str] = None) -> str:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Previous context:\n{chr(10).join(context[-5:])}"})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return ""

    def generate_with_claude(self, prompt: str, context: List[str] = None) -> str:
        try:
            # Build messages - add context as part of user message if present
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
        if "claude" in model.lower():
            return self.generate_with_claude(prompt, context)
        else:
            return self.generate_with_openai(prompt, context)

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

        # Update entanglement
        E_t = self.entanglement.update(true_align)

        trial_results = {
            'trial': trial_num,
            'prompt': prompt,
            'model': model,
            'true': {
                'response': true_response[:200] + '...' if len(true_response) > 200 else true_response,
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

            trial_results['controls'][condition] = {
                'alignment': control_align,
                'insight_quality': control_iq,
                'delta_rci': delta_rci,
                'delta_iq': delta_iq
            }

        return trial_results

    def run_experiment(self, prompts: List[str]) -> Dict:
        print("\n" + "="*60)
        print("MCH v8.1 EXPERIMENT")
        print("="*60)
        print(f"Participant: {self.config.participant_id}")
        print(f"Trials: {self.config.n_trials}")
        print(f"Models: {self.config.models_to_test}")
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

                time.sleep(1)  # Rate limiting

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

                    if p_val < 0.05 and mean_d > 0:
                        print(f"  => P1 SUPPORTED: History-specific uplift detected!")
                    else:
                        print(f"  => P1 not supported at alpha=0.05")

    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mch_results_{self.config.participant_id}_{timestamp}.json"

        self.results['metadata']['end_time'] = datetime.now().isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved: {filename}")
        return filename

    def visualize_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MCH v8.1 Experiment Results', fontsize=14, fontweight='bold')

        # Plot 1: ΔRCI by trial
        ax1 = axes[0, 0]
        for model in self.config.models_to_test:
            trials = [t for t in self.results['trials'] if t['model'] == model]
            trial_nums = [t['trial'] for t in trials]
            deltas = [t['controls']['cold']['delta_rci'] for t in trials if 'cold' in t['controls']]
            if deltas:
                ax1.plot(trial_nums[:len(deltas)], deltas, 'o-', label=model, linewidth=2, markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('DRCI (True - Cold)')
        ax1.set_title('History-Specific Uplift (P1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Entanglement over time
        ax2 = axes[0, 1]
        for model in self.config.models_to_test:
            trials = [t for t in self.results['trials'] if t['model'] == model]
            E_vals = [t['true']['entanglement'] for t in trials]
            if E_vals:
                ax2.plot(range(len(E_vals)), E_vals, 's-', label=f'{model} E_t', linewidth=2)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Entanglement State (E_t)')
        ax2.set_title('Entanglement Accumulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Alignment distribution
        ax3 = axes[1, 0]
        true_aligns = [t['true']['alignment'] for t in self.results['trials']]
        cold_aligns = [t['controls']['cold']['alignment'] for t in self.results['trials'] if 'cold' in t['controls']]
        ax3.boxplot([true_aligns, cold_aligns], tick_labels=['True History', 'Cold Start'])
        ax3.set_ylabel('Alignment Score')
        ax3.set_title('Alignment by Condition')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Model comparison
        ax4 = axes[1, 1]
        model_means = []
        model_stds = []
        for model in self.config.models_to_test:
            trials = [t for t in self.results['trials'] if t['model'] == model]
            deltas = [t['controls']['cold']['delta_rci'] for t in trials if 'cold' in t['controls']]
            model_means.append(np.mean(deltas) if deltas else 0)
            model_stds.append(np.std(deltas) if deltas else 0)

        bars = ax4.bar(range(len(model_means)), model_means, yerr=model_stds, capsize=5, alpha=0.7)
        for i, bar in enumerate(bars):
            bar.set_color('green' if model_means[i] > 0 else 'red')
        ax4.set_xticks(range(len(self.config.models_to_test)))
        ax4.set_xticklabels([m.split('/')[-1][:15] for m in self.config.models_to_test], rotation=45)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Mean DRCI')
        ax4.set_title('Model Comparison')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mch_plots_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved: {filename}")
        # Don't show in non-interactive mode
        # plt.show()

def main():
    # Sequential prompts that build on each other
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
        # Extended to 30
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

    config = ExperimentConfig(
        participant_id="dr_laxman_sequential_30trials",
        n_trials=30,
        models_to_test=["claude-opus-4-20250514", "gpt-4o-mini"],
        control_conditions=["cold", "scrambled"],
        lambda_E=0.15
    )

    experiment = MCHExperiment(config)
    results = experiment.run_experiment(sequential_prompts)
    experiment.save_results("mch_results_sequential_30trials.json")
    experiment.visualize_results()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
