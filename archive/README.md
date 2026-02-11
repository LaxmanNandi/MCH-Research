# Archive Directory

This directory contains deprecated files, old data, and historical materials from the MCH Experiments project.

## Structure

### `old_figures/`
Pre-50-trial figures (Paper 1 original submission):
- Used 30-trial or 100-trial data (old methodology)
- Replaced by current 50-trial publication-ready figures in `docs/figures/`
- **Date**: January 2026 (arXiv v1 submission)

### `old_data/`
Deprecated data exports:
- `paper2_exports/`: Trial-level data exports from earlier analysis phases
- Contains intermediate analysis outputs no longer used in current papers

### `investigations/`
Temporary validation and verification scripts:
- `check_historical_correlation.py`: Verified r=0.76 entanglement correlation
- `check_response_text_availability.py`: Audited which models have response text
- `INVESTIGATION_SUMMARY.md`: Documentation of Paper 4 entanglement data restoration
- `historical_entanglement.csv`: Backup of valid 11-model entanglement data
- **Purpose**: One-time investigation during Paper 4 data validation (Feb 11, 2026)

### `deprecated_scripts/`
Old analysis scripts replaced by current workflow:
- `paper1_medical_stats.py`: Early statistical analysis (pre-cross-domain design)
- `paper1_statistical_analysis.py`: Original Paper 1 analysis
- `verify_epistemological_cues.py`: Exploratory cue detection script
- `parse_alignments.py`: Alignment parsing utilities

### `arxiv_v1/`
Original arXiv submission materials (Paper 1):
- `MCH_Paper1_arXiv.pdf`: Published arXiv preprint
- `MCH_Paper1_arXiv.tex`: LaTeX source
- Original figures (7 figures from pre-50-trial data)
- **Status**: Historical reference; current work uses updated methodology

---

## Current Active Locations

**Active Figures**: `docs/figures/publication/` and `docs/figures/paper3/`, `docs/figures/paper4/`
**Active Data**: `data/*/` (50-trial JSON files), `results/tables/` (CSV metrics)
**Active Scripts**: `scripts/analysis/` (publication figure generation)
**Active Papers**: `docs/papers/Paper3_Results.md`, `docs/papers/Paper4_Results.md`

---

**Last Updated**: February 11, 2026
**Archived By**: Data cleanup and consolidation process
