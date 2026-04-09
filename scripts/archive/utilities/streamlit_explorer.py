"""
MCH Dataset Explorer - Interactive Streamlit App
Explore the Model Coherence Hypothesis experimental results.
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the MCH dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "mch_complete_dataset.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


@st.cache_data
def create_trials_dataframe(data):
    """Convert trials to a flat DataFrame."""
    rows = []
    for model_name, model_data in data['models'].items():
        info = model_data['info']
        for trial in model_data['trials']:
            rows.append({
                'model': model_name,
                'vendor': info['vendor'],
                'tier': info['tier'],
                'model_id': info['model_id'],
                'trial_id': trial['trial_id'],
                'prompt': trial['prompt'],
                'alignment_true': trial['alignments']['true'],
                'alignment_cold': trial['alignments']['cold'],
                'alignment_scrambled': trial['alignments']['scrambled'],
                'delta_rci_cold': trial['delta_rci']['cold'],
                'delta_rci_scrambled': trial['delta_rci']['scrambled'],
                'entanglement': trial['entanglement'],
                'response_length_true': trial['response_lengths']['true'],
                'response_length_cold': trial['response_lengths']['cold'],
                'response_length_scrambled': trial['response_lengths']['scrambled']
            })
    return pd.DataFrame(rows)


def get_pattern(mean, p_value):
    """Determine pattern based on mean and p-value."""
    if p_value < 0.05:
        if mean > 0:
            return "CONVERGENT"
        else:
            return "SOVEREIGN"
    return "NEUTRAL"


def render_overview(data, df):
    """Render the overview dashboard."""
    st.markdown('<p class="main-header">MCH Dataset Overview</p>', unsafe_allow_html=True)

    metadata = data['metadata']

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials", metadata['total_trials'])
    with col2:
        st.metric("Models Tested", len(data['models']))
    with col3:
        st.metric("Vendors", len(set(m['vendor'] for m in metadata['models'])))
    with col4:
        st.metric("Trials per Model", metadata['trials_per_model'])

    st.divider()

    # Summary statistics table
    st.subheader("Model Summary Statistics")

    summary_data = []
    for model_name, model_data in data['models'].items():
        info = model_data['info']
        summary = model_data['summary']
        drci_values = [t['delta_rci']['cold'] for t in model_data['trials']]
        t_stat, p_val = stats.ttest_1samp(drci_values, 0)

        summary_data.append({
            'Model': model_name,
            'Vendor': info['vendor'],
            'Tier': info['tier'],
            'ΔRCI Mean': f"{summary['drci_cold']['mean']:.4f}",
            'ΔRCI Std': f"{summary['drci_cold']['std']:.4f}",
            'Min': f"{summary['drci_cold']['min']:.4f}",
            'Max': f"{summary['drci_cold']['max']:.4f}",
            'p-value': f"{p_val:.4f}",
            'Pattern': get_pattern(summary['drci_cold']['mean'], p_val)
        })

    summary_df = pd.DataFrame(summary_data)

    # Color the pattern column
    def color_pattern(val):
        if val == 'SOVEREIGN':
            return 'background-color: #ffcccc'
        elif val == 'CONVERGENT':
            return 'background-color: #ccffcc'
        return 'background-color: #ffffcc'

    styled_df = summary_df.style.applymap(color_pattern, subset=['Pattern'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.divider()

    # Distribution plot
    st.subheader("ΔRCI Distribution by Model")

    fig = px.violin(df, x='model', y='delta_rci_cold', color='vendor',
                    box=True, points='outliers',
                    labels={'delta_rci_cold': 'ΔRCI (Cold)', 'model': 'Model'},
                    color_discrete_map={
                        'OpenAI': '#10a37f',
                        'Google': '#4285f4',
                        'Anthropic': '#d4a574'
                    })
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Metadata info
    with st.expander("Dataset Metadata"):
        st.json({
            'title': metadata['title'],
            'author': metadata['author'],
            'affiliation': metadata['affiliation'],
            'created_date': metadata['created_date'],
            'version': metadata['version'],
            'experiment_parameters': metadata['experiment_parameters']
        })


def render_model_selector(data, df):
    """Render the model selector view."""
    st.markdown('<p class="main-header">Model Explorer</p>', unsafe_allow_html=True)

    # Model selection
    model_names = list(data['models'].keys())
    selected_model = st.selectbox("Select a Model", model_names)

    model_data = data['models'][selected_model]
    model_df = df[df['model'] == selected_model]

    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Vendor:** {model_data['info']['vendor']}")
    with col2:
        st.info(f"**Tier:** {model_data['info']['tier']}")
    with col3:
        st.info(f"**Model ID:** {model_data['info']['model_id']}")

    st.divider()

    # Statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ΔRCI Statistics")
        summary = model_data['summary']['drci_cold']
        drci_values = model_df['delta_rci_cold'].values
        t_stat, p_val = stats.ttest_1samp(drci_values, 0)

        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 't-statistic', 'p-value', 'Pattern'],
            'Value': [
                f"{summary['mean']:.4f}",
                f"{summary['std']:.4f}",
                f"{summary['min']:.4f}",
                f"{summary['max']:.4f}",
                f"{t_stat:.4f}",
                f"{p_val:.6f}",
                get_pattern(summary['mean'], p_val)
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Distribution")
        fig = px.histogram(model_df, x='delta_rci_cold', nbins=20,
                          labels={'delta_rci_cold': 'ΔRCI (Cold)'},
                          color_discrete_sequence=['#1f77b4'])
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=summary['mean'], line_dash="solid", line_color="green",
                     annotation_text=f"Mean: {summary['mean']:.3f}")
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    st.subheader("ΔRCI Across Trials")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_df['trial_id'],
        y=model_df['delta_rci_cold'],
        mode='lines+markers',
        name='ΔRCI',
        line=dict(color='#1f77b4', width=1),
        marker=dict(size=4)
    ))

    # Rolling mean
    window = 10
    rolling_mean = model_df['delta_rci_cold'].rolling(window=window).mean()
    fig.add_trace(go.Scatter(
        x=model_df['trial_id'],
        y=rolling_mean,
        mode='lines',
        name=f'{window}-trial Rolling Mean',
        line=dict(color='orange', width=2)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        xaxis_title="Trial ID",
        yaxis_title="ΔRCI (Cold)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trial_viewer(data, df):
    """Render the individual trial viewer."""
    st.markdown('<p class="main-header">Trial Viewer</p>', unsafe_allow_html=True)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        model_names = ['All'] + list(data['models'].keys())
        selected_model = st.selectbox("Filter by Model", model_names, key="trial_model")
    with col2:
        trial_range = st.slider("Trial ID Range", 1, 100, (1, 100))

    # Filter dataframe
    filtered_df = df.copy()
    if selected_model != 'All':
        filtered_df = filtered_df[filtered_df['model'] == selected_model]
    filtered_df = filtered_df[
        (filtered_df['trial_id'] >= trial_range[0]) &
        (filtered_df['trial_id'] <= trial_range[1])
    ]

    st.info(f"Showing {len(filtered_df)} trials")

    # Display trials
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Trial {row['trial_id']} - {row['model']} (ΔRCI: {row['delta_rci_cold']:.4f})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ΔRCI (Cold)", f"{row['delta_rci_cold']:.4f}")
            with col2:
                st.metric("ΔRCI (Scrambled)", f"{row['delta_rci_scrambled']:.4f}")
            with col3:
                st.metric("Entanglement", f"{row['entanglement']:.4f}")

            st.markdown("**Prompt:**")
            st.write(row['prompt'])

            st.markdown("**Alignments:**")
            align_df = pd.DataFrame({
                'Condition': ['True', 'Cold', 'Scrambled'],
                'Alignment': [row['alignment_true'], row['alignment_cold'], row['alignment_scrambled']],
                'Response Length': [row['response_length_true'], row['response_length_cold'], row['response_length_scrambled']]
            })
            st.dataframe(align_df, use_container_width=True, hide_index=True)


def render_comparison(data, df):
    """Render the model comparison view."""
    st.markdown('<p class="main-header">Model Comparison</p>', unsafe_allow_html=True)

    model_names = list(data['models'].keys())

    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Select Model 1", model_names, index=0)
    with col2:
        model2 = st.selectbox("Select Model 2", model_names, index=1)

    if model1 == model2:
        st.warning("Please select two different models to compare.")
        return

    df1 = df[df['model'] == model1]
    df2 = df[df['model'] == model2]

    data1 = data['models'][model1]
    data2 = data['models'][model2]

    st.divider()

    # Side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(model1)
        st.info(f"**{data1['info']['vendor']}** | {data1['info']['tier']}")

        summary1 = data1['summary']['drci_cold']
        drci1 = df1['delta_rci_cold'].values
        t1, p1 = stats.ttest_1samp(drci1, 0)

        st.metric("Mean ΔRCI", f"{summary1['mean']:.4f}")
        st.metric("Std Dev", f"{summary1['std']:.4f}")
        st.metric("Pattern", get_pattern(summary1['mean'], p1))

    with col2:
        st.subheader(model2)
        st.info(f"**{data2['info']['vendor']}** | {data2['info']['tier']}")

        summary2 = data2['summary']['drci_cold']
        drci2 = df2['delta_rci_cold'].values
        t2, p2 = stats.ttest_1samp(drci2, 0)

        st.metric("Mean ΔRCI", f"{summary2['mean']:.4f}")
        st.metric("Std Dev", f"{summary2['std']:.4f}")
        st.metric("Pattern", get_pattern(summary2['mean'], p2))

    st.divider()

    # Statistical comparison
    st.subheader("Statistical Comparison")
    t_ind, p_ind = stats.ttest_ind(drci1, drci2)
    u_stat, p_mann = stats.mannwhitneyu(drci1, drci2, alternative='two-sided')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("t-statistic", f"{t_ind:.4f}")
    with col2:
        st.metric("p-value (t-test)", f"{p_ind:.6f}")
    with col3:
        significant = "Yes" if p_ind < 0.05 else "No"
        st.metric("Significant Difference?", significant)

    st.divider()

    # Overlay distributions
    st.subheader("Distribution Comparison")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=drci1, name=model1, opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=drci2, name=model2, opacity=0.7, nbinsx=20))
    fig.update_layout(
        barmode='overlay',
        xaxis_title="ΔRCI (Cold)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Box plot comparison
    comparison_df = pd.concat([
        df1[['model', 'delta_rci_cold']],
        df2[['model', 'delta_rci_cold']]
    ])

    fig = px.box(comparison_df, x='model', y='delta_rci_cold', color='model',
                labels={'delta_rci_cold': 'ΔRCI (Cold)', 'model': 'Model'})
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def render_vendor_analysis(data, df):
    """Render the vendor analysis view."""
    st.markdown('<p class="main-header">Vendor & Tier Analysis</p>', unsafe_allow_html=True)

    # Two-way ANOVA results
    st.subheader("Two-Way ANOVA Results")

    # Calculate ANOVA
    vendors = df['vendor'].unique()
    tiers = df['tier'].unique()

    # Vendor effect
    vendor_groups = [df[df['vendor'] == v]['delta_rci_cold'].values for v in vendors]
    f_vendor, p_vendor = stats.f_oneway(*vendor_groups)

    # Tier effect
    tier_groups = [df[df['tier'] == t]['delta_rci_cold'].values for t in tiers]
    f_tier, p_tier = stats.f_oneway(*tier_groups)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vendor Effect")
        st.metric("F-statistic", f"{f_vendor:.3f}")
        st.metric("p-value", f"{p_vendor:.6f}")
        if p_vendor < 0.05:
            st.success("Significant (p < 0.05)")
        else:
            st.warning("Not Significant")

    with col2:
        st.markdown("### Tier Effect")
        st.metric("F-statistic", f"{f_tier:.3f}")
        st.metric("p-value", f"{p_tier:.6f}")
        if p_tier < 0.05:
            st.success("Significant (p < 0.05)")
        else:
            st.warning("Not Significant")

    st.divider()

    # Box plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ΔRCI by Vendor")
        fig = px.box(df, x='vendor', y='delta_rci_cold', color='vendor',
                    labels={'delta_rci_cold': 'ΔRCI (Cold)', 'vendor': 'Vendor'},
                    color_discrete_map={
                        'OpenAI': '#10a37f',
                        'Google': '#4285f4',
                        'Anthropic': '#d4a574'
                    })
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Add mean markers
        for vendor in vendors:
            mean_val = df[df['vendor'] == vendor]['delta_rci_cold'].mean()
            fig.add_scatter(x=[vendor], y=[mean_val], mode='markers',
                          marker=dict(symbol='diamond', size=15, color='red'),
                          name=f'{vendor} Mean', showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ΔRCI by Tier")
        fig = px.box(df, x='tier', y='delta_rci_cold', color='tier',
                    labels={'delta_rci_cold': 'ΔRCI (Cold)', 'tier': 'Tier'},
                    color_discrete_map={
                        'Efficient': '#90EE90',
                        'Flagship': '#FFB6C1'
                    })
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        for tier in tiers:
            mean_val = df[df['tier'] == tier]['delta_rci_cold'].mean()
            fig.add_scatter(x=[tier], y=[mean_val], mode='markers',
                          marker=dict(symbol='diamond', size=15, color='red'),
                          name=f'{tier} Mean', showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Vendor x Tier interaction
    st.subheader("Vendor × Tier Interaction")

    fig = px.box(df, x='vendor', y='delta_rci_cold', color='tier',
                labels={'delta_rci_cold': 'ΔRCI (Cold)', 'vendor': 'Vendor', 'tier': 'Tier'},
                color_discrete_map={
                    'Efficient': '#90EE90',
                    'Flagship': '#FFB6C1'
                })
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table by vendor
    st.subheader("Summary by Vendor")
    vendor_summary = df.groupby('vendor').agg({
        'delta_rci_cold': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    vendor_summary.columns = ['Mean', 'Std', 'Min', 'Max', 'N']
    st.dataframe(vendor_summary, use_container_width=True)


def render_export(data, df):
    """Render the export view."""
    st.markdown('<p class="main-header">Export Data</p>', unsafe_allow_html=True)

    st.write("Export filtered data as CSV for further analysis.")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        models = ['All'] + list(data['models'].keys())
        selected_models = st.multiselect("Select Models", models, default=['All'])

    with col2:
        vendors = ['All'] + list(df['vendor'].unique())
        selected_vendors = st.multiselect("Select Vendors", vendors, default=['All'])

    with col3:
        tiers = ['All'] + list(df['tier'].unique())
        selected_tiers = st.multiselect("Select Tiers", tiers, default=['All'])

    # Filter data
    export_df = df.copy()

    if 'All' not in selected_models:
        export_df = export_df[export_df['model'].isin(selected_models)]
    if 'All' not in selected_vendors:
        export_df = export_df[export_df['vendor'].isin(selected_vendors)]
    if 'All' not in selected_tiers:
        export_df = export_df[export_df['tier'].isin(selected_tiers)]

    st.info(f"Selected {len(export_df)} trials for export")

    # Preview
    st.subheader("Data Preview")
    st.dataframe(export_df.head(20), use_container_width=True)

    # Column selection
    st.subheader("Select Columns to Export")
    all_columns = list(export_df.columns)
    selected_columns = st.multiselect("Columns", all_columns, default=all_columns)

    if selected_columns:
        export_df = export_df[selected_columns]

    # Export button
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="mch_data_export.csv",
        mime="text/csv",
        type="primary"
    )

    # Also offer JSON export
    st.divider()
    st.subheader("Full Dataset Export")

    json_str = json.dumps(data, indent=2)
    st.download_button(
        label="Download Full Dataset (JSON)",
        data=json_str,
        file_name="mch_complete_dataset.json",
        mime="application/json"
    )


def main():
    """Main app function."""
    # Sidebar
    st.sidebar.title("MCH Dataset Explorer")
    st.sidebar.markdown("---")

    # Load data
    try:
        data = load_data()
        df = create_trials_dataframe(data)
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure mch_complete_dataset.json is in the same directory.")
        return

    # Navigation
    st.sidebar.subheader("Navigation")
    pages = {
        "Overview": "overview",
        "Model Explorer": "model",
        "Trial Viewer": "trials",
        "Model Comparison": "comparison",
        "Vendor Analysis": "vendor",
        "Export Data": "export"
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **MCH Study**: Differential Relational Dynamics in Large Language Models

    **Author**: Dr. Laxman M M, MBBS

    **Trials**: 600 total (100 per model)
    """)

    # Render selected page
    if selection == "Overview":
        render_overview(data, df)
    elif selection == "Model Explorer":
        render_model_selector(data, df)
    elif selection == "Trial Viewer":
        render_trial_viewer(data, df)
    elif selection == "Model Comparison":
        render_comparison(data, df)
    elif selection == "Vendor Analysis":
        render_vendor_analysis(data, df)
    elif selection == "Export Data":
        render_export(data, df)


if __name__ == "__main__":
    main()
