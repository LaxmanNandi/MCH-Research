# Scripts Directory

This directory contains the experiment runners, analysis pipelines, and
figure-generation scripts for the MCH research programme (Papers 1–9).

## Layout

```
scripts/
├── shared/         # Path helpers and shared utilities (NEW)
├── experiments/    # API-call experiment runners (medical, legal, ethics, etc.)
├── analysis/       # Generic analysis utilities
├── eeg_pilot/      # Sleep-EDF brain coherence pilot
├── paper3/         # Paper 3 (cross-domain temporal) scripts
├── paper6/         # Paper 6 (conservation constraint) scripts
├── paper7/         # Paper 7 (content-order decomposition) scripts
├── paper8/         # Paper 8 (encoding fidelity / EFI) scripts
├── paper9/         # Paper 9 (measurement validation) scripts
├── validate/       # Verification scripts
└── archive/        # Frozen historical scripts (do not modify)
```

The README at `papers/README.md` describes the publication status of each paper.

## Running Scripts

All active scripts use **repository-relative paths** via `scripts/shared/paths.py`.
You can run them from any working directory:

```bash
python scripts/paper6/paper6_unified_K_pipeline.py
python scripts/paper6/paper6_conservation_product.py
python scripts/paper8/paper8_efi_pipeline.py
```

To check that path resolution is working, run:

```bash
python scripts/shared/paths.py
```

You should see the absolute path of the repository root and its main subdirectories.

## Active Pipelines (Paper 6 example)

For Paper 6 (the conservation constraint), the canonical entry points are:

| Script | Purpose |
|--------|---------|
| `paper6/paper6_unified_K_pipeline.py` | **Authoritative single-source K computation across all 24 model-domain runs.** Reads `paper6_manuscript_data.json` and verifies stored vs recomputed K. No fallbacks. |
| `paper6/paper6_conservation_product.py` | Legacy combined-source product test. Retained for figure generation. |
| `paper6/paper6_figures.py` | Publication figures for Paper 6. |
| `paper6/compile_paper6_data.py` | Rebuild `paper6_manuscript_data.json` from raw trial JSONs. |

For provenance and audit history, see `scripts/paper6/METHODOLOGY_AUDIT.md`.

## Adding a New Script

When writing a new script, use the shared path helper instead of hardcoding
absolute paths. Two patterns work:

**Pattern 1 — module-style (preferred for new scripts):**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.paths import repo_root, data_dir

raw = data_dir() / "medical" / "open_models" / "mch_results_x.json"
```

**Pattern 2 — direct (single-file scripts):**

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = REPO_ROOT / "data" / "paper6" / "paper6_manuscript_data.json"
```

Either pattern keeps the script portable. **Do not** hardcode
`C:/Users/barla/mch_experiments`. Many older scripts in `archive/` still do —
they are frozen and were not migrated.

## Dependencies

See repository root `requirements.txt`. Key libraries:

- `sentence-transformers` (all-MiniLM-L6-v2, all-mpnet-base-v2, LaBSE embeddings)
- `scipy`, `numpy`, `pandas` (statistics and data wrangling)
- `matplotlib` (figure generation)
- `mne`, `mne-bids` (EEG pilot only)

## Provenance

The MCH programme's data hygiene principle: **every result must be traceable
back to raw trial JSONs.** Per-trial files in `data/{medical,philosophy,legal,ethics}/`
are the ground truth. Any aggregated CSV or summary JSON is derived. If you
find a discrepancy, the raw trial files are authoritative.

For the conservation constraint specifically, the audit trail is documented
in `scripts/paper6/METHODOLOGY_AUDIT.md`.
