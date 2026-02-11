"""
Repository Cleanup Script - Professional Publication Prep
==========================================================
Systematically removes internal collaboration language and moves
internal documents to .internal/ folder (git-ignored).

WHAT THIS SCRIPT DOES:
1. Creates .internal/ folder structure
2. Moves reviewer prep docs, drafts, and audit logs
3. Rewrites script headers to remove "WRONG", outlier drama
4. Renames public docs to remove "Paper 3/4" labels
5. Updates .gitignore
6. Generates cleanup report

RUN WITH: python cleanup_repo.py --preview (dry run)
          python cleanup_repo.py --execute (actual cleanup)
"""

import os
import shutil
import re
from pathlib import Path

# Base directory
BASE = Path(r'C:\Users\barla\mch_experiments')

# ============================================================
# PHASE 1: CREATE .internal/ FOLDER STRUCTURE
# ============================================================

INTERNAL_STRUCTURE = {
    '.internal': [
        'reviewer_prep',
        'summaries',
        'audit',
        'methods',
        'drafts',
        'backups'
    ]
}

def create_internal_folders(execute=False):
    """Create .internal/ folder structure"""
    print("=" * 70)
    print("PHASE 1: CREATE .internal/ FOLDER STRUCTURE")
    print("=" * 70)

    for parent, subdirs in INTERNAL_STRUCTURE.items():
        parent_path = BASE / parent
        if execute:
            parent_path.mkdir(exist_ok=True)
            print(f"OK Created: {parent_path}")
        else:
            print(f"[PREVIEW] Would create: {parent_path}")

        for subdir in subdirs:
            subdir_path = parent_path / subdir
            if execute:
                subdir_path.mkdir(exist_ok=True)
                print(f"  OK Created: {subdir_path}")
            else:
                print(f"  [PREVIEW] Would create: {subdir_path}")

    print()

# ============================================================
# PHASE 2: FILE MOVES (INTERNAL DOCS)
# ============================================================

FILE_MOVES = [
    # (source, destination, reason)
    ('docs/Paper3_Critical_Revision_TypesOfContextSensitivity.md',
     '.internal/reviewer_prep/Paper3_Critical_Revision.md',
     'Contains reviewer concern anticipation'),

    ('docs/Paper3_Executive_Summary.md',
     '.internal/summaries/Paper3_Executive_Summary.md',
     'Internal summary with reviewer concerns'),

    ('docs/audit_error_log.md',
     '.internal/audit/audit_error_log.md',
     'Exposes AI collaboration workflow'),

    ('docs/audit_error_log.csv',
     '.internal/audit/audit_error_log.csv',
     'Audit data with Codex references'),

    ('docs/DIA_Methods_Appendix.md',
     '.internal/methods/DIA_Methods_Appendix.md',
     'Details AI collaboration architecture'),

    ('docs/Paper3_Results_Section_DRAFT.md',
     '.internal/drafts/Paper3_Results_Section_DRAFT.md',
     'Draft file'),
]

def move_files(execute=False):
    """Move internal documents to .internal/"""
    print("=" * 70)
    print("PHASE 2: MOVE INTERNAL DOCUMENTS")
    print("=" * 70)

    for source, dest, reason in FILE_MOVES:
        source_path = BASE / source
        dest_path = BASE / dest

        if source_path.exists():
            if execute:
                # Create backup first
                backup_path = BASE / '.internal' / 'backups' / source_path.name
                shutil.copy2(source_path, backup_path)

                # Move file
                shutil.move(str(source_path), str(dest_path))
                print(f"OK Moved: {source} -> {dest}")
                print(f"  Reason: {reason}")
            else:
                print(f"[PREVIEW] Would move: {source} -> {dest}")
                print(f"  Reason: {reason}")
        else:
            print(f"WARNING: Not found: {source}")

    print()

# ============================================================
# PHASE 3: SCRIPT HEADER REWRITES
# ============================================================

SCRIPT_REWRITES = {
    'scripts/validate/extract_and_analyze_trial_level.py': {
        'old_header': '''"""
Bayesian Convergence Re-Analysis on CORRECT Trial-Level Data
=============================================================
Previous analysis used position-level data (entanglement_position_data.csv) — WRONG.
This script extracts trial-level dRCI from JSON files and runs convergence analysis.
"""''',
        'new_header': '''"""
Trial-Level ΔRCI Convergence Analysis
======================================
Extracts trial-level ΔRCI values from experiment JSON files and performs
Bayesian convergence analysis stratified by domain and model.

Data sources:
- Medical reasoning: Type 2 (closed-goal) tasks
- Philosophy: Type 1 (open-goal) tasks

Output:
- trial_level_drci.csv: Consolidated trial-level data
- Bayesian convergence statistics (overall, by domain, by model)
- Comparison table and visualization
"""'''
    },

    'scripts/validate/clean_figures_8_9.py': {
        'old_header': '''"""
Clean Figures 8 and 9: Remove OLD outliers, add RERUN data
- Exclude: gemini_flash medical original (dRCI=-0.13)
- Exclude: gpt4o_mini medical original (dRCI=0.03)
- Include: gemini_flash medical RERUN (dRCI~0.43) — missing from CSV
- Keep: gpt4o_mini_rerun (dRCI~0.32) — already in CSV
"""''',
        'new_header': '''"""
Publication-Quality Figures: Trial Convergence and Model Comparison
====================================================================
Generates Figures 8 and 9 using cleaned 50-trial dataset.

Dataset methodology:
- Uses 50-trial reruns with corrected prompt set
- Excludes early runs with uncorrected prompts
- Includes updated data for models with multiple runs

Output:
- Figure 8: Trial-level convergence (scatter + rolling mean)
- Figure 9: Model comparison (mean ΔRCI with 95% CI)

Both figures saved at 300 DPI for publication.
"""'''
    }
}

COMMENT_REPLACEMENTS = [
    # (pattern, replacement)
    (r'# TASK (\d+):', r'# STEP \1:'),
    (r'WRONG', 'updated'),
    (r'OLD outlier', 'early run'),
    (r'RERUN data', 'updated data'),
    (r'NOTE: Did NOT push to git \(Codex will handle\)\.', ''),
    (r'outlier variance', 'variance'),
]

def rewrite_scripts(execute=False):
    """Rewrite script headers and comments"""
    print("=" * 70)
    print("PHASE 3: REWRITE SCRIPT HEADERS")
    print("=" * 70)

    for script_path, rewrites in SCRIPT_REWRITES.items():
        full_path = BASE / script_path

        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace header
            old_header = rewrites['old_header']
            new_header = rewrites['new_header']

            if old_header in content:
                new_content = content.replace(old_header, new_header)

                # Apply comment replacements
                for pattern, replacement in COMMENT_REPLACEMENTS:
                    new_content = re.sub(pattern, replacement, new_content)

                if execute:
                    # Backup
                    backup_path = BASE / '.internal' / 'backups' / full_path.name
                    shutil.copy2(full_path, backup_path)

                    # Write cleaned version
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"OK Rewrote: {script_path}")
                else:
                    print(f"[PREVIEW] Would rewrite: {script_path}")
            else:
                print(f"WARNING: Header not found in: {script_path}")
        else:
            print(f"WARNING: Not found: {script_path}")

    print()

# ============================================================
# PHASE 4: RENAME PUBLIC DOCS
# ============================================================

DOC_RENAMES = [
    # (old_name, new_name)
    ('docs/Paper3_Results_Discussion.md', 'docs/Results_Temporal_Dynamics.md'),
    ('docs/Paper4_Results_Discussion.md', 'docs/Results_Entanglement_Analysis.md'),
    ('docs/Paper4_Claims_Evidence.md', 'docs/Claims_Evidence_Entanglement.md'),
    ('docs/Paper4_Entanglement_Methods_Appendix.md', 'docs/Methods_Entanglement.md'),
]

def rename_docs(execute=False):
    """Rename public docs to remove Paper 3/4 labels"""
    print("=" * 70)
    print("PHASE 4: RENAME PUBLIC DOCUMENTS")
    print("=" * 70)

    for old_name, new_name in DOC_RENAMES:
        old_path = BASE / old_name
        new_path = BASE / new_name

        if old_path.exists():
            if execute:
                # Backup
                backup_path = BASE / '.internal' / 'backups' / old_path.name
                shutil.copy2(old_path, backup_path)

                # Rename
                shutil.move(str(old_path), str(new_path))
                print(f"OK Renamed: {old_name} -> {new_name}")
            else:
                print(f"[PREVIEW] Would rename: {old_name} -> {new_name}")
        else:
            print(f"WARNING: Not found: {old_name}")

    print()

# ============================================================
# PHASE 5: UPDATE .gitignore
# ============================================================

GITIGNORE_ADDITIONS = """
# Internal documents (not for public release)
.internal/
CLEANED_*.md
*_HEADER.py
cleanup_plan.md
"""

def update_gitignore(execute=False):
    """Add .internal/ to .gitignore"""
    print("=" * 70)
    print("PHASE 5: UPDATE .gitignore")
    print("=" * 70)

    gitignore_path = BASE / '.gitignore'

    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if '.internal/' not in content:
            if execute:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n' + GITIGNORE_ADDITIONS)
                print("OK Updated .gitignore")
            else:
                print("[PREVIEW] Would add to .gitignore:")
                print(GITIGNORE_ADDITIONS)
        else:
            print("OK .gitignore already contains .internal/")
    else:
        if execute:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(GITIGNORE_ADDITIONS)
            print("OK Created .gitignore")
        else:
            print("[PREVIEW] Would create .gitignore")

    print()

# ============================================================
# PHASE 6: UPDATE README
# ============================================================

def update_readme(execute=False):
    """Replace README with cleaned version"""
    print("=" * 70)
    print("PHASE 6: UPDATE README")
    print("=" * 70)

    old_readme = BASE / 'README.md'
    new_readme = BASE / 'CLEANED_README.md'

    if new_readme.exists():
        if execute:
            # Backup
            backup_path = BASE / '.internal' / 'backups' / 'README_original.md'
            shutil.copy2(old_readme, backup_path)

            # Replace
            shutil.copy2(new_readme, old_readme)
            print("OK Updated README.md")
            print("  Original backed up to .internal/backups/README_original.md")
        else:
            print("[PREVIEW] Would replace README.md with CLEANED_README.md")
    else:
        print("WARNING: CLEANED_README.md not found")

    print()

# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(execute=False):
    """Generate cleanup summary report"""
    print("=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)

    print(f"\nMode: {'EXECUTE' if execute else 'PREVIEW (dry run)'}")
    print(f"\nFolders created: {len(INTERNAL_STRUCTURE['.internal'])}")
    print(f"Files moved: {len(FILE_MOVES)}")
    print(f"Scripts rewritten: {len(SCRIPT_REWRITES)}")
    print(f"Docs renamed: {len(DOC_RENAMES)}")

    if execute:
        print("\nOK Cleanup complete!")
        print("\nNEXT STEPS:")
        print("1. Review changes with: git status")
        print("2. Check renamed files work correctly")
        print("3. Commit changes: git add . && git commit -m 'Prepare repository for public release'")
        print("4. Push to origin: git push")
    else:
        print("\n[PREVIEW MODE] No changes made.")
        print("Run with --execute to apply changes.")

    print()

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import sys

    execute = '--execute' in sys.argv

    if not execute:
        print("\n" + "!" * 70)
        print("RUNNING IN PREVIEW MODE (dry run)")
        print("No files will be modified. Use --execute to apply changes.")
        print("!" * 70 + "\n")

    create_internal_folders(execute)
    move_files(execute)
    rewrite_scripts(execute)
    rename_docs(execute)
    update_gitignore(execute)
    update_readme(execute)
    generate_report(execute)
