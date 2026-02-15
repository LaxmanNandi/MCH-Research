# Scripts Directory

## Organization

### `experiments/` -- Running Experiments
Scripts that execute the MCH experimental protocol (API calls, embedding computation, data collection).

| Script | Purpose |
|--------|---------|
| `run_medical_experiments.py` | Run medical domain experiments for open models |
| `rerun_philosophy_open_models.py` | Re-run philosophy experiments with response text saving |

### `analysis/` -- Data Analysis
Scripts for analyzing experimental data and generating results for Papers 3-6.

| Script | Paper | Purpose |
|--------|-------|---------|
| `generate_paper3_cross_domain.py` | 3 | Cross-domain temporal analysis |
| `regenerate_entanglement_figures.py` | 4 | Entanglement validation figures |
| `regenerate_paper4_fig5_fig7.py` | 4 | Paper 4 specific figures |
| `paper6_conservation_law.py` | 6 | MI-based conservation test |
| `paper6_conservation_product.py` | 6 | Direct product conservation test |
| `paper6_figures.py` | 6 | Paper 6 publication figures |
| `paper6_verify.py` | 6 | Full Paper 6 data verification |
| `validate_entanglement.py` | 4 | Entanglement theory validation |

### `validate/` -- Verification Scripts
Scripts for validating data integrity and paper claims.

| Script | Purpose |
|--------|---------|
| `paper5_accuracy_analysis.py` | Paper 5 accuracy verification |
| `paper6_ushape_analysis.py` | Var_Ratio U-shape analysis |

### Root Scripts (Legacy)
Scripts at the `scripts/` root level are from earlier phases of the project. Key ones:

| Script | Purpose |
|--------|---------|
| `generate_paper1_figures.py` | Paper 1 figure generation |
| `generate_paper2_figures.py` | Paper 2 figure generation |
| `generate_paper5_figures.py` | Paper 5 figure generation |
| `paper2_analysis.py` | Paper 2 statistical analysis |
| `mch_*.py` | Individual model experiment runners |

## Running Scripts

All scripts use absolute paths to `C:/Users/barla/mch_experiments` as the base directory. Run from any location:

```bash
python scripts/analysis/paper6_conservation_product.py
python scripts/generate_paper5_figures.py
```

## Dependencies

See `/requirements.txt` for package dependencies. Key libraries:
- `sentence-transformers` (all-MiniLM-L6-v2 embeddings)
- `scipy` (statistical tests)
- `matplotlib` (figure generation)
- `pandas`, `numpy` (data processing)
