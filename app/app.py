"""
MCH Dataset Explorer - Interactive Streamlit App
Explore the Model Coherence Hypothesis experimental results.

1000 trials across 13 model-domain combinations:
- Philosophy: 700 trials (7 models x 100 trials)
- Medical: 300 trials (6 models x 50 trials)

January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="MCH Dataset Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .convergent { color: #28a745; font-weight: bold; }
    .sovereign { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Vendor colors
VENDOR_COLORS = {
    "OpenAI": "#10a37f",
    "Google": "#4285f4",
    "Anthropic": "#d4a574",
}

MODEL_VENDORS = {
    "GPT-4o": "OpenAI",
    "GPT-4o-mini": "OpenAI",
    "GPT-5.2": "OpenAI",
    "Claude Opus": "Anthropic",
    "Claude Haiku": "Anthropic",
    "Gemini 2.5 Pro": "Google",
    "Gemini 2.5 Flash": "Google",
}

# Data loading functions
@st.cache_data
def load_philosophy_data():
    """Load philosophy domain data."""
    data_dir = Path(__file__).parent / "data" / "philosophy"

    files = {
        'GPT-4o': 'mch_results_gpt4o_100trials.json',
        'GPT-4o-mini': 'mch_results_gpt4o_mini_n100_merged.json',
        'GPT-5.2': 'mch_results_gpt_5_2_100trials.json',
        'Claude Opus': 'mch_results_claude_opus_100trials.json',
        'Claude Haiku': 'mch_results_claude_haiku_100trials.json',
        'Gemini 2.5 Pro': 'mch_results_gemini_pro_100trials.json',
        'Gemini 2.5 Flash': 'mch_results_gemini_flash_100trials.json',
    }

    all_trials = []
    for model_name, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for trial in data.get('trials', []):
                trial_data = extract_trial_data(trial, model_name, 'philosophy')
                if trial_data:
                    all_trials.append(trial_data)

    return pd.DataFrame(all_trials)

@st.cache_data
def load_medical_data():
    """Load medical domain data."""
    data_dir = Path(__file__).parent / "data" / "medical"

    files = {
        'GPT-4o': 'mch_results_gpt4o_medical_50trials.json',
        'GPT-4o-mini': 'mch_results_gpt4o_mini_rerun_medical_50trials.json',
        'GPT-5.2': 'mch_results_gpt_5_2_medical_50trials.json',
        'Claude Haiku': 'mch_results_claude_haiku_medical_50trials.json',
        'Claude Opus': 'mch_results_claude_opus_medical_50trials.json',
        'Gemini 2.5 Flash': 'mch_results_gemini_flash_medical_50trials.json',
    }

    all_trials = []
    for model_name, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for trial in data.get('trials', []):
                trial_data = extract_trial_data(trial, model_name, 'medical')
                if trial_data:
                    all_trials.append(trial_data)

    return pd.DataFrame(all_trials)

def extract_trial_data(trial, model_name, domain):
    """Extract standardized trial data from different JSON formats."""
    try:
        trial_id = trial.get('trial', 0)

        # Get delta_rci
        if isinstance(trial.get('delta_rci'), dict):
            delta_rci = trial['delta_rci'].get('cold', 0)
        elif 'controls' in trial and 'cold' in trial['controls']:
            delta_rci = trial['controls']['cold'].get('delta_rci', 0)
        else:
            delta_rci = 0

        # Get prompt
        prompt = trial.get('prompt', '')
        if not prompt and 'prompts' in trial:
            prompt = trial['prompts'][0] if trial['prompts'] else ''

        # Get alignments
        align_true = align_cold = align_scrambled = 0
        if 'alignments' in trial:
            if isinstance(trial['alignments'].get('true'), list):
                align_true = np.mean(trial['alignments']['true'])
                align_cold = np.mean(trial['alignments'].get('cold', [0]))
                align_scrambled = np.mean(trial['alignments'].get('scrambled', [0]))
            else:
                align_true = trial['alignments'].get('mean_true', trial['alignments'].get('true', 0))
                align_cold = trial['alignments'].get('mean_cold', trial['alignments'].get('cold', 0))
                align_scrambled = trial['alignments'].get('mean_scrambled', trial['alignments'].get('scrambled', 0))
        elif 'true' in trial:
            align_true = trial['true'].get('alignment', 0)
            if 'controls' in trial:
                align_cold = trial['controls'].get('cold', {}).get('alignment', 0)
                align_scrambled = trial['controls'].get('scrambled', {}).get('alignment', 0)

        # Get insight quality and entanglement (philosophy)
        insight_quality = 0
        entanglement = 0
        if 'true' in trial:
            insight_quality = trial['true'].get('insight_quality', 0)
            entanglement = trial['true'].get('entanglement', 0)
        if 'entanglement' in trial:
            entanglement = trial['entanglement']

        # Get response text if available
        response = ''
        if 'true' in trial and 'response' in trial['true']:
            response = trial['true']['response']

        # Determine pattern
        if delta_rci > 0.01:
            pattern = 'CONVERGENT'
        elif delta_rci < -0.01:
            pattern = 'SOVEREIGN'
        else:
            pattern = 'NEUTRAL'

        return {
            'trial_id': f"{model_name}_{domain}_{trial_id}",
            'trial_num': trial_id,
            'model': model_name,
            'vendor': MODEL_VENDORS.get(model_name, 'Unknown'),
            'domain': domain,
            'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
            'full_prompt': prompt,
            'delta_rci': delta_rci,
            'pattern': pattern,
            'align_true': align_true,
            'align_cold': align_cold,
            'align_scrambled': align_scrambled,
            'insight_quality': insight_quality,
            'entanglement': entanglement,
            'response': response,
        }
    except Exception as e:
        return None

@st.cache_data
def get_summary_stats(df):
    """Calculate summary statistics by model and domain."""
    summary = df.groupby(['model', 'domain', 'vendor']).agg({
        'delta_rci': ['mean', 'std', 'count'],
        'pattern': lambda x: x.value_counts().index[0] if len(x) > 0 else 'NEUTRAL'
    }).reset_index()

    summary.columns = ['model', 'domain', 'vendor', 'mean_drci', 'std_drci', 'n_trials', 'dominant_pattern']

    # Calculate p-value (one-sample t-test against 0)
    p_values = []
    for _, row in summary.iterrows():
        model_data = df[(df['model'] == row['model']) & (df['domain'] == row['domain'])]['delta_rci']
        if len(model_data) > 1:
            t_stat, p_val = stats.ttest_1samp(model_data, 0)
            p_values.append(p_val)
        else:
            p_values.append(1.0)
    summary['p_value'] = p_values

    return summary

# Load data
@st.cache_data
def load_all_data():
    phil_df = load_philosophy_data()
    med_df = load_medical_data()
    return pd.concat([phil_df, med_df], ignore_index=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 MCH Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Model Coherence Hypothesis - 1000 Trials Across 13 Model-Domain Combinations</p>', unsafe_allow_html=True)

    # Load data
    df = load_all_data()
    summary = get_summary_stats(df)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["📊 Overview Dashboard", "🔍 Model Explorer", "📋 Trial Viewer",
         "⚖️ Model Comparison", "🔬 Domain Analysis", "🏢 Vendor Analysis", "📥 Export Data"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Total Trials", f"{len(df):,}")
    st.sidebar.metric("Philosophy", f"{len(df[df['domain']=='philosophy']):,}")
    st.sidebar.metric("Medical", f"{len(df[df['domain']=='medical']):,}")

    # Page routing
    if page == "📊 Overview Dashboard":
        show_overview(df, summary)
    elif page == "🔍 Model Explorer":
        show_model_explorer(df, summary)
    elif page == "📋 Trial Viewer":
        show_trial_viewer(df)
    elif page == "⚖️ Model Comparison":
        show_model_comparison(df, summary)
    elif page == "🔬 Domain Analysis":
        show_domain_analysis(df, summary)
    elif page == "🏢 Vendor Analysis":
        show_vendor_analysis(df, summary)
    elif page == "📥 Export Data":
        show_export(df, summary)

def show_overview(df, summary):
    """Overview Dashboard."""
    st.header("📊 Overview Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trials", f"{len(df):,}")
    with col2:
        convergent = len(df[df['pattern'] == 'CONVERGENT'])
        st.metric("Convergent Trials", f"{convergent:,} ({convergent/len(df)*100:.1f}%)")
    with col3:
        sovereign = len(df[df['pattern'] == 'SOVEREIGN'])
        st.metric("Sovereign Trials", f"{sovereign:,} ({sovereign/len(df)*100:.1f}%)")
    with col4:
        st.metric("Models Tested", f"{df['model'].nunique()}")

    st.markdown("---")

    # Summary table
    st.subheader("Summary Statistics by Model")

    # Format the summary table
    display_summary = summary.copy()
    display_summary['mean_drci'] = display_summary['mean_drci'].apply(lambda x: f"{x:+.4f}")
    display_summary['std_drci'] = display_summary['std_drci'].apply(lambda x: f"{x:.4f}")
    display_summary['p_value'] = display_summary['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")

    # Color the pattern column
    def color_pattern(val):
        if val == 'CONVERGENT':
            return 'background-color: #d4edda; color: #155724;'
        elif val == 'SOVEREIGN':
            return 'background-color: #f8d7da; color: #721c24;'
        else:
            return 'background-color: #e2e3e5; color: #383d41;'

    styled_summary = display_summary.style.applymap(color_pattern, subset=['dominant_pattern'])
    st.dataframe(display_summary, use_container_width=True)

    # Violin plots
    st.subheader("ΔRCI Distribution by Model")

    domain_filter = st.radio("Domain", ["All", "Philosophy", "Medical"], horizontal=True)

    if domain_filter != "All":
        plot_df = df[df['domain'] == domain_filter.lower()]
    else:
        plot_df = df

    fig = px.violin(
        plot_df,
        x='model',
        y='delta_rci',
        color='vendor',
        color_discrete_map=VENDOR_COLORS,
        box=True,
        points="outliers",
        title=f"ΔRCI Distribution - {domain_filter}",
        labels={'delta_rci': 'ΔRCI (True - Cold)', 'model': 'Model'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig.add_hline(y=0.01, line_dash="dot", line_color="green", annotation_text="Convergent threshold")
    fig.add_hline(y=-0.01, line_dash="dot", line_color="red", annotation_text="Sovereign threshold")
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_model_explorer(df, summary):
    """Model Explorer page."""
    st.header("🔍 Model Explorer")

    # Model selector
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_model = st.selectbox("Select Model", sorted(df['model'].unique()))
    with col2:
        domain_filter = st.selectbox("Domain", ["All", "Philosophy", "Medical"])

    # Filter data
    model_df = df[df['model'] == selected_model]
    if domain_filter != "All":
        model_df = model_df[model_df['domain'] == domain_filter.lower()]

    # Model stats
    st.subheader(f"{selected_model} Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trials", len(model_df))
    with col2:
        mean_drci = model_df['delta_rci'].mean()
        st.metric("Mean ΔRCI", f"{mean_drci:+.4f}")
    with col3:
        std_drci = model_df['delta_rci'].std()
        st.metric("Std ΔRCI", f"{std_drci:.4f}")
    with col4:
        pattern = 'CONVERGENT' if mean_drci > 0.01 else 'SOVEREIGN' if mean_drci < -0.01 else 'NEUTRAL'
        st.metric("Pattern", pattern)

    # Distribution
    st.subheader("ΔRCI Distribution")
    fig = px.histogram(
        model_df,
        x='delta_rci',
        nbins=30,
        color='domain',
        title=f"{selected_model} - ΔRCI Histogram",
        labels={'delta_rci': 'ΔRCI'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=mean_drci, line_dash="solid", line_color="red", annotation_text=f"Mean: {mean_drci:.4f}")
    st.plotly_chart(fig, use_container_width=True)

    # Trial table
    st.subheader("Trials")
    display_cols = ['trial_id', 'domain', 'prompt', 'delta_rci', 'pattern']
    st.dataframe(
        model_df[display_cols].sort_values('trial_num'),
        use_container_width=True,
        height=400
    )

def show_trial_viewer(df):
    """Trial Viewer page."""
    st.header("📋 Trial Viewer")

    # Trial selector
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        model = st.selectbox("Model", sorted(df['model'].unique()), key='trial_model')
    with col2:
        domain = st.selectbox("Domain", df[df['model']==model]['domain'].unique(), key='trial_domain')
    with col3:
        trial_nums = sorted(df[(df['model']==model) & (df['domain']==domain)]['trial_num'].unique())
        trial_num = st.selectbox("Trial #", trial_nums, key='trial_num')

    # Get trial
    trial = df[(df['model']==model) & (df['domain']==domain) & (df['trial_num']==trial_num)]

    if len(trial) > 0:
        trial = trial.iloc[0]

        st.subheader("Trial Details")

        # Prompt
        st.markdown("**Prompt:**")
        st.info(trial['full_prompt'] if trial['full_prompt'] else trial['prompt'])

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ΔRCI", f"{trial['delta_rci']:+.4f}")
        with col2:
            pattern_color = 'green' if trial['pattern'] == 'CONVERGENT' else 'red' if trial['pattern'] == 'SOVEREIGN' else 'gray'
            st.markdown(f"**Pattern:** <span style='color:{pattern_color}'>{trial['pattern']}</span>", unsafe_allow_html=True)
        with col3:
            st.metric("Vendor", trial['vendor'])

        # Alignments
        st.subheader("Alignments")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TRUE", f"{trial['align_true']:.4f}")
        with col2:
            st.metric("COLD", f"{trial['align_cold']:.4f}")
        with col3:
            st.metric("SCRAMBLED", f"{trial['align_scrambled']:.4f}")

        # Bar chart of alignments
        align_data = pd.DataFrame({
            'Condition': ['TRUE', 'COLD', 'SCRAMBLED'],
            'Alignment': [trial['align_true'], trial['align_cold'], trial['align_scrambled']]
        })
        fig = px.bar(align_data, x='Condition', y='Alignment',
                     color='Condition',
                     color_discrete_map={'TRUE': 'green', 'COLD': 'blue', 'SCRAMBLED': 'orange'},
                     title="Alignment by Condition")
        st.plotly_chart(fig, use_container_width=True)

        # Additional metrics (if available)
        if trial['insight_quality'] > 0:
            st.subheader("Additional Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Insight Quality", f"{trial['insight_quality']:.4f}")
            with col2:
                st.metric("Entanglement", f"{trial['entanglement']:.4f}")

        # Response (if available)
        if trial['response']:
            st.subheader("Response (TRUE condition)")
            st.text_area("Response", trial['response'], height=150, disabled=True)

def show_model_comparison(df, summary):
    """Model Comparison page."""
    st.header("⚖️ Model Comparison")

    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Model 1", sorted(df['model'].unique()), key='compare_model1')
    with col2:
        model2 = st.selectbox("Model 2", sorted(df['model'].unique()), index=1, key='compare_model2')

    domain_filter = st.radio("Domain", ["All", "Philosophy", "Medical"], horizontal=True, key='compare_domain')

    # Filter data
    df1 = df[df['model'] == model1]
    df2 = df[df['model'] == model2]

    if domain_filter != "All":
        df1 = df1[df1['domain'] == domain_filter.lower()]
        df2 = df2[df2['domain'] == domain_filter.lower()]

    # Side-by-side stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(model1)
        st.metric("N Trials", len(df1))
        st.metric("Mean ΔRCI", f"{df1['delta_rci'].mean():+.4f}")
        st.metric("Std ΔRCI", f"{df1['delta_rci'].std():.4f}")

    with col2:
        st.subheader(model2)
        st.metric("N Trials", len(df2))
        st.metric("Mean ΔRCI", f"{df2['delta_rci'].mean():+.4f}")
        st.metric("Std ΔRCI", f"{df2['delta_rci'].std():.4f}")

    # Statistical comparison
    st.subheader("Statistical Comparison")

    if len(df1) > 1 and len(df2) > 1:
        # t-test
        t_stat, p_val = stats.ttest_ind(df1['delta_rci'], df2['delta_rci'])

        # Cohen's d
        pooled_std = np.sqrt((df1['delta_rci'].std()**2 + df2['delta_rci'].std()**2) / 2)
        cohens_d = (df1['delta_rci'].mean() - df2['delta_rci'].mean()) / pooled_std if pooled_std > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("t-statistic", f"{t_stat:.3f}")
        with col2:
            st.metric("p-value", f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}")
        with col3:
            st.metric("Cohen's d", f"{cohens_d:.3f}")

    # Side-by-side violin plots
    st.subheader("Distribution Comparison")

    compare_df = pd.concat([df1, df2])
    fig = px.violin(
        compare_df,
        x='model',
        y='delta_rci',
        color='model',
        box=True,
        points="outliers",
        title=f"ΔRCI Comparison: {model1} vs {model2}"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

def show_domain_analysis(df, summary):
    """Domain Analysis page."""
    st.header("🔬 Domain Analysis")

    st.subheader("Philosophy vs Medical Domain Comparison")

    # Get models that appear in both domains
    phil_models = set(df[df['domain'] == 'philosophy']['model'].unique())
    med_models = set(df[df['domain'] == 'medical']['model'].unique())
    common_models = phil_models & med_models

    # Cross-domain comparison
    cross_domain = []
    for model in common_models:
        phil_mean = df[(df['model'] == model) & (df['domain'] == 'philosophy')]['delta_rci'].mean()
        med_mean = df[(df['model'] == model) & (df['domain'] == 'medical')]['delta_rci'].mean()

        phil_pattern = 'CONVERGENT' if phil_mean > 0.01 else 'SOVEREIGN' if phil_mean < -0.01 else 'NEUTRAL'
        med_pattern = 'CONVERGENT' if med_mean > 0.01 else 'SOVEREIGN' if med_mean < -0.01 else 'NEUTRAL'

        flip = phil_pattern != med_pattern

        cross_domain.append({
            'Model': model,
            'Vendor': MODEL_VENDORS.get(model, 'Unknown'),
            'Philosophy ΔRCI': phil_mean,
            'Medical ΔRCI': med_mean,
            'Shift': med_mean - phil_mean,
            'Philosophy Pattern': phil_pattern,
            'Medical Pattern': med_pattern,
            'Behavior Flip': '⚡ YES' if flip else 'No'
        })

    cross_df = pd.DataFrame(cross_domain)

    # Highlight flippers
    st.markdown("### Models That Flip Behavior")
    st.dataframe(cross_df, use_container_width=True)

    # Visualization
    st.subheader("Cross-Domain ΔRCI Shift")

    fig = go.Figure()

    for _, row in cross_df.iterrows():
        color = VENDOR_COLORS.get(row['Vendor'], 'gray')
        # Line connecting philosophy to medical
        fig.add_trace(go.Scatter(
            x=['Philosophy', 'Medical'],
            y=[row['Philosophy ΔRCI'], row['Medical ΔRCI']],
            mode='lines+markers',
            name=row['Model'],
            line=dict(color=color, width=3),
            marker=dict(size=12)
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.01, line_dash="dot", line_color="green")
    fig.add_hline(y=-0.01, line_dash="dot", line_color="red")

    fig.update_layout(
        title="ΔRCI Shift: Philosophy → Medical",
        yaxis_title="ΔRCI",
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Domain-specific insights
    st.subheader("Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Philosophy Domain")
        st.markdown("- Most models show **SOVEREIGN/NEUTRAL** behavior")
        st.markdown("- **GPT-5.2** is uniquely **CONVERGENT**")
        st.markdown("- High variability in responses")

    with col2:
        st.markdown("### Medical Domain")
        st.markdown("- Most models show **CONVERGENT** behavior")
        st.markdown("- **Gemini Flash** remains **SOVEREIGN**")
        st.markdown("- Lower variability, consistent patterns")

def show_vendor_analysis(df, summary):
    """Vendor Analysis page."""
    st.header("🏢 Vendor Analysis")

    # Aggregate by vendor
    vendor_stats = df.groupby(['vendor', 'domain']).agg({
        'delta_rci': ['mean', 'std', 'count']
    }).reset_index()
    vendor_stats.columns = ['vendor', 'domain', 'mean_drci', 'std_drci', 'n_trials']

    st.subheader("Vendor Statistics")
    st.dataframe(vendor_stats, use_container_width=True)

    # Box plots by vendor
    st.subheader("ΔRCI by Vendor")

    domain_filter = st.radio("Domain", ["All", "Philosophy", "Medical"], horizontal=True, key='vendor_domain')

    if domain_filter != "All":
        plot_df = df[df['domain'] == domain_filter.lower()]
    else:
        plot_df = df

    fig = px.box(
        plot_df,
        x='vendor',
        y='delta_rci',
        color='vendor',
        color_discrete_map=VENDOR_COLORS,
        points="outliers",
        title=f"ΔRCI Distribution by Vendor - {domain_filter}"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ANOVA
    st.subheader("ANOVA Results")

    vendors = plot_df['vendor'].unique()
    if len(vendors) >= 2:
        groups = [plot_df[plot_df['vendor'] == v]['delta_rci'].values for v in vendors]
        f_stat, p_val = stats.f_oneway(*groups)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-statistic", f"{f_stat:.3f}")
        with col2:
            st.metric("p-value", f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}")

        if p_val < 0.05:
            st.success("Significant difference between vendors (p < 0.05)")
        else:
            st.info("No significant difference between vendors (p >= 0.05)")

def show_export(df, summary):
    """Export Data page."""
    st.header("📥 Export Data")

    st.subheader("Download Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Filtered Data (CSV)")

        domain_filter = st.selectbox("Filter by Domain", ["All", "Philosophy", "Medical"], key='export_domain')
        model_filter = st.multiselect("Filter by Model", sorted(df['model'].unique()), key='export_models')

        export_df = df.copy()
        if domain_filter != "All":
            export_df = export_df[export_df['domain'] == domain_filter.lower()]
        if model_filter:
            export_df = export_df[export_df['model'].isin(model_filter)]

        st.write(f"Selected: {len(export_df)} trials")

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="mch_filtered_data.csv",
            mime="text/csv"
        )

    with col2:
        st.markdown("### Summary Statistics (CSV)")

        summary_csv = summary.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="mch_summary_stats.csv",
            mime="text/csv"
        )

        st.markdown("### Complete Dataset (JSON)")

        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="mch_complete_dataset.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
