"""
Repository Reorganization Script
Consolidates folders and creates clean public-facing structure

PROPOSED STRUCTURE:
mch_experiments/
├── data/                  # All experiment data (consolidated)
│   ├── medical/          # Medical domain results
│   └── philosophy/       # Philosophy domain results
├── results/              # Analysis outputs and figures
│   ├── figures/         # Publication-ready figures
│   ├── tables/          # CSV analysis tables
│   └── metrics/         # Computed metrics
├── docs/                 # Documentation only
│   ├── papers/          # Paper drafts (Results, Methods)
│   └── guides/          # Usage guides
├── scripts/             # Experiment and analysis scripts
│   ├── experiments/    # Run experiments
│   └── analysis/       # Analyze data
├── .dev/               # Development files (gitignored)
│   ├── logs/
│   ├── drafts/
│   └── archive/
└── [root files]        # README, LICENSE, requirements.txt only
"""

import shutil
from pathlib import Path
import sys

BASE = Path('c:/Users/barla/mch_experiments')

# ============================================================
# REORGANIZATION PLAN
# ============================================================

FOLDER_STRUCTURE = {
    'data/medical': 'Consolidated medical domain experiments',
    'data/philosophy': 'Consolidated philosophy domain experiments',
    'results/figures': 'Publication-ready figures',
    'results/tables': 'CSV analysis tables',
    'results/metrics': 'Computed metrics',
    'docs/papers': 'Paper manuscripts and results',
    'docs/guides': 'Usage documentation',
    'scripts/experiments': 'Experiment runners',
    'scripts/analysis': 'Analysis and validation',
    '.dev/logs': 'Development logs',
    '.dev/drafts': 'Work in progress',
    '.dev/archive': 'Old versions',
}

# Data consolidation
DATA_MOVES = [
    # Medical data
    ('data/medical_results', 'data/medical/closed_models', 'Original closed medical'),
    ('data/gemini_flash_medical_rerun', 'data/medical/gemini_flash', 'Gemini Flash rerun'),
    ('data/open_medical_rerun', 'data/medical/open_models', 'Open model medical rerun'),

    # Philosophy data
    ('data/philosophy_results', 'data/philosophy/original', 'Original philosophy'),
    ('data/open_model_results', 'data/philosophy/open_models', 'Open philosophy'),
    ('data/closed_model_philosophy_rerun', 'data/philosophy/closed_models', 'Closed philosophy rerun'),

    # Export data can be regenerated
    ('data/paper2_exports', '.dev/archive/paper2_exports', 'Paper 2 exports'),
]

# Analysis results
ANALYSIS_MOVES = [
    ('analysis/trial_level_drci.csv', 'results/tables/trial_level_drci.csv', 'Trial-level DRCI'),
    ('analysis/entanglement_position_data.csv', 'results/tables/entanglement_position_data.csv', 'Entanglement data'),
    ('analysis/entanglement_correlations.csv', 'results/tables/entanglement_correlations.csv', 'Entanglement correlations'),
    ('analysis/entanglement_variance_summary.csv', 'results/tables/entanglement_variance_summary.csv', 'Variance summary'),
    ('analysis/position30_analysis', 'results/metrics/position30_analysis', 'P30 metrics'),
]

# Figures consolidation
FIGURE_MOVES = [
    ('docs/figures/publication/*', 'results/figures/', 'Publication figures'),
    ('docs/figures/paper4/*', 'results/figures/paper4/', 'Paper 4 figures'),
    ('docs/figures/legacy/analysis/entanglement_theory_validation.png', 'results/figures/entanglement_validation.png', 'Featured figure'),
    ('figures/*', '.dev/archive/old_figures/', 'Old figure folder'),
]

# Documentation
DOC_MOVES = [
    ('docs/Results_Temporal_Dynamics.md', 'docs/papers/Paper3_Results.md', 'Paper 3 results'),
    ('docs/Results_Entanglement_Analysis.md', 'docs/papers/Paper4_Results.md', 'Paper 4 results'),
    ('docs/Methods_Entanglement.md', 'docs/papers/Paper4_Methods.md', 'Paper 4 methods'),
    ('docs/Claims_Evidence_Entanglement.md', 'docs/papers/Paper4_Claims.md', 'Paper 4 claims'),
    ('docs/Llama_Safety_Anomaly.md', 'docs/papers/Llama_Safety_Note.md', 'Llama anomaly note'),
]

# Scripts
SCRIPT_MOVES = [
    ('scripts/mch_open_models_medical_rerun.py', 'scripts/experiments/run_medical_experiments.py', 'Medical experiment runner'),
    ('scripts/validate/extract_and_analyze_trial_level.py', 'scripts/analysis/compute_trial_drci.py', 'Trial-level analysis'),
    ('scripts/validate/test_entanglement_theory.py', 'scripts/analysis/validate_entanglement.py', 'Entanglement validation'),
    ('scripts/paper3_generate_figures.py', 'scripts/analysis/generate_paper3_figures.py', 'Paper 3 figures'),
    ('scripts/validate/*.py', 'scripts/analysis/', 'Other validation scripts'),
]

# Development/internal moves
DEV_MOVES = [
    ('conversation_logs', '.dev/logs/conversations', 'Conversation logs'),
    ('.internal', '.dev/internal', 'Internal docs'),
    ('app', '.dev/archive/app', 'Old app folder'),
    ('arxiv_submission', '.dev/archive/arxiv_v1', 'ArXiv v1 submission'),
    ('paper1', '.dev/archive/paper1', 'Paper 1 old'),
    ('papers', '.dev/archive/papers', 'Papers old'),
    ('visualizations', '.dev/archive/visualizations', 'Old visualizations'),
]

# Root-level cleanup
ROOT_MOVES = [
    ('MCH_Paper1_arXiv.pdf', '.dev/archive/MCH_Paper1_arXiv.pdf', 'Paper 1 PDF'),
    ('MCH_Paper1_arXiv.tex', '.dev/archive/MCH_Paper1_arXiv.tex', 'Paper 1 LaTeX'),
    ('parse_alignments.py', '.dev/archive/parse_alignments.py', 'Old script'),
    ('CLEANED_README.md', '.dev/drafts/CLEANED_README.md', 'Draft README'),
    ('_config.yml', '.dev/archive/_config.yml', 'Jekyll config'),
]

# ============================================================
# GITIGNORE UPDATE
# ============================================================

GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Development
.dev/

# OS
.DS_Store
Thumbs.db

# Data (optional - uncomment if data shouldn't be in repo)
# data/

# Results (generated)
results/
analysis/

# Logs
*.log
"""

# ============================================================
# FUNCTIONS
# ============================================================

def create_folders(execute=False):
    """Create new folder structure"""
    print("=" * 80)
    print("PHASE 1: CREATE NEW FOLDER STRUCTURE")
    print("=" * 80)

    for folder, desc in FOLDER_STRUCTURE.items():
        folder_path = BASE / folder
        if execute:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] {folder:<30s} # {desc}")
        else:
            print(f"[PREVIEW] {folder:<30s} # {desc}")
    print()

def move_files(moves, phase_name, execute=False):
    """Move files according to plan"""
    print("=" * 80)
    print(f"PHASE: {phase_name}")
    print("=" * 80)

    for source, dest, reason in moves:
        source_path = BASE / source
        dest_path = BASE / dest

        # Handle wildcards
        if '*' in source:
            source_dir = BASE / Path(source).parent
            pattern = Path(source).name
            if source_dir.exists():
                files = list(source_dir.glob(pattern))
                if execute:
                    dest_path.mkdir(parents=True, exist_ok=True)
                    for file in files:
                        shutil.move(str(file), str(dest_path / file.name))
                    print(f"[OK] {source} -> {dest} ({len(files)} files)")
                else:
                    print(f"[PREVIEW] {source} -> {dest} ({len(files)} files)")
            continue

        if source_path.exists():
            if execute:
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # If destination exists and is a directory, move into it
                if dest_path.exists() and dest_path.is_dir():
                    shutil.move(str(source_path), str(dest_path / source_path.name))
                else:
                    shutil.move(str(source_path), str(dest_path))

                print(f"[OK] {source}")
                print(f"    -> {dest}")
            else:
                print(f"[PREVIEW] {source} -> {dest}")
                print(f"  Reason: {reason}")
        else:
            if not execute:
                print(f"[SKIP] Not found: {source}")
    print()

def update_gitignore(execute=False):
    """Update .gitignore"""
    print("=" * 80)
    print("PHASE: UPDATE .gitignore")
    print("=" * 80)

    gitignore_path = BASE / '.gitignore'

    if execute:
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(GITIGNORE_CONTENT)
        print("[OK] Updated .gitignore")
    else:
        print("[PREVIEW] Would write new .gitignore")
    print()

def update_readme(execute=False):
    """Update README structure section"""
    print("=" * 80)
    print("PHASE: UPDATE README STRUCTURE")
    print("=" * 80)

    new_structure = """## Repository Structure

```
mch_experiments/
├── data/                    # Experiment data
│   ├── medical/            # Medical domain (STEMI case)
│   └── philosophy/         # Philosophy domain (consciousness)
├── results/                # Analysis outputs
│   ├── figures/           # Publication figures
│   ├── tables/            # Data tables (CSV)
│   └── metrics/           # Computed metrics
├── docs/                   # Documentation
│   ├── papers/            # Paper manuscripts
│   └── guides/            # Usage guides
├── scripts/                # Code
│   ├── experiments/       # Run experiments
│   └── analysis/          # Analyze data
├── README.md
├── LICENSE
├── requirements.txt
└── CONTRIBUTORS.md
```
"""

    if execute:
        readme_path = BASE / 'README.md'
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace structure section
        import re
        pattern = r'## Repository Structure.*?```\n\n---'
        replacement = new_structure + '\n---'
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("[OK] Updated README.md")
    else:
        print("[PREVIEW] Would update README structure section")
    print()

def cleanup_empty_dirs(execute=False):
    """Remove empty directories"""
    print("=" * 80)
    print("PHASE: CLEANUP EMPTY DIRECTORIES")
    print("=" * 80)

    if execute:
        for path in sorted(BASE.rglob('*'), reverse=True):
            if path.is_dir() and not any(path.iterdir()) and path.name not in ['.git', '.dev', 'venv']:
                try:
                    path.rmdir()
                    print(f"[OK] Removed empty: {path.relative_to(BASE)}")
                except:
                    pass
    else:
        print("[PREVIEW] Would remove empty directories")
    print()

def generate_summary():
    """Generate summary report"""
    print("\n" + "=" * 80)
    print("REORGANIZATION SUMMARY")
    print("=" * 80)
    print(f"New folders:     {len(FOLDER_STRUCTURE)}")
    print(f"Data moves:      {len(DATA_MOVES)}")
    print(f"Analysis moves:  {len(ANALYSIS_MOVES)}")
    print(f"Figure moves:    {len(FIGURE_MOVES)}")
    print(f"Doc moves:       {len(DOC_MOVES)}")
    print(f"Script moves:    {len(SCRIPT_MOVES)}")
    print(f"Dev moves:       {len(DEV_MOVES)}")
    print(f"Root cleanup:    {len(ROOT_MOVES)}")
    print("=" * 80)

# ============================================================
# MAIN
# ============================================================

def main():
    execute = '--execute' in sys.argv

    print("\n" + "=" * 80)
    print("MCH EXPERIMENTS - REPOSITORY REORGANIZATION")
    print("=" * 80)
    print(f"Mode: {'EXECUTE' if execute else 'PREVIEW (dry run)'}")
    print("=" * 80)
    print()

    if not execute:
        print("WARNING: PREVIEW MODE - No changes will be made")
        print("   Run with --execute to apply changes")
        print()

    create_folders(execute)
    move_files(DATA_MOVES, "DATA CONSOLIDATION", execute)
    move_files(ANALYSIS_MOVES, "ANALYSIS RESULTS", execute)
    move_files(FIGURE_MOVES, "FIGURES", execute)
    move_files(DOC_MOVES, "DOCUMENTATION", execute)
    move_files(SCRIPT_MOVES, "SCRIPTS", execute)
    move_files(DEV_MOVES, "DEVELOPMENT FILES", execute)
    move_files(ROOT_MOVES, "ROOT CLEANUP", execute)
    update_gitignore(execute)
    update_readme(execute)
    cleanup_empty_dirs(execute)

    generate_summary()

    if execute:
        print("\n[OK] Reorganization complete!")
        print("\nNEXT STEPS:")
        print("1. Review changes:  git status")
        print("2. Update imports in scripts if needed")
        print("3. Test scripts still work")
        print("4. Commit: git add . && git commit -m 'refactor: clean repository structure'")
    else:
        print("\nTo apply changes, run:")
        print("  python scripts/validate/reorganize_repository.py --execute")

if __name__ == '__main__':
    main()
