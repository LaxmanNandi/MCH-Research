# MCH Dataset Explorer - Streamlit App

An interactive web application to explore the Model Coherence Hypothesis (MCH) experimental results.

## Features

### 1. Overview Dashboard
- Total trials, models, and vendors at a glance
- Summary statistics table with pattern classification
- Interactive violin plot showing ΔRCI distribution by model
- Dataset metadata viewer

### 2. Model Explorer
- Dropdown to select any model
- Detailed statistics (mean, std, min, max, p-value)
- Distribution histogram
- Time series plot of ΔRCI across trials with rolling mean

### 3. Trial Viewer
- Filter by model and trial ID range
- Expandable cards for each trial
- View prompt text, alignments, and response lengths
- See individual ΔRCI values

### 4. Model Comparison
- Side-by-side comparison of any two models
- Statistical tests (t-test, Mann-Whitney U)
- Overlaid distribution histograms
- Box plot comparison

### 5. Vendor Analysis
- Two-way ANOVA results (Vendor × Tier)
- Box plots by vendor and tier
- Interaction plot (Vendor × Tier)
- Summary statistics by vendor

### 6. Export Data
- Filter by model, vendor, and tier
- Select columns to export
- Download as CSV
- Download full dataset as JSON

## Installation

### Option 1: Using pip
```bash
cd C:\Users\barla\mch_experiments\publication_analysis
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda create -n mch-explorer python=3.10
conda activate mch-explorer
pip install -r requirements.txt
```

## Running the App

```bash
cd C:\Users\barla\mch_experiments\publication_analysis
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.18.0
- scipy >= 1.11.0

## File Structure

```
publication_analysis/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── mch_complete_dataset.json   # Dataset (required)
└── STREAMLIT_README.md         # This file
```

## Data Source

The app uses `mch_complete_dataset.json` which contains:
- 600 total trials (100 per model)
- 6 models across 3 vendors
- Alignment scores, ΔRCI values, and entanglement metrics

## Troubleshooting

### App won't start
- Ensure `mch_complete_dataset.json` is in the same directory as `app.py`
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Slow loading
- First load may take a few seconds to initialize cache
- Subsequent loads will be faster due to `@st.cache_data`

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## Author

Dr. Laxman M M, MBBS
Government Duty Medical Officer, Primary Health Centre Manchi
Bantwal Taluk, Dakshina Kannada, Karnataka, India

## License

Research use only. Contact author for commercial applications.
