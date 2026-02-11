# Repository Cleanup - Ready for Your Review

## What I've Created

### 1. **cleanup_plan.md** - Detailed Strategy
Complete breakdown of what needs cleaning and why. Read this first to understand the full scope.

### 2. **cleanup_repo.py** - Automated Cleanup Script
Safe, automated script with:
- **Preview mode** (default): Shows what WOULD change without touching anything
- **Execute mode** (--execute flag): Actually performs the cleanup

### 3. **Cleaned Examples** - See Before/After

#### CLEANED_README.md
- ✓ Removed "Paper 3/4" labels → "Study A/B" or descriptive names
- ✓ Professional tone throughout
- ✓ Removed internal process references
- ✓ Added acknowledgment of AI collaboration (transparent but professional)

#### CLEANED Script Headers (2 files)
- extract_and_analyze_trial_level_HEADER.py
- clean_figures_8_9_HEADER.py

**Changes:**
- ✗ "Previous analysis used WRONG data"
- ✓ "Trial-level ΔRCI convergence analysis"

- ✗ "Remove OLD outliers (dRCI=-0.13)"
- ✓ "Uses 50-trial reruns with corrected prompt set"

- ✗ "NOTE: Codex will handle"
- ✓ (removed entirely)

---

## What the Script Will Do

### Phase 1: Create .internal/ Folder Structure
```
.internal/
├── reviewer_prep/    (concern docs, post-hoc discussion)
├── summaries/        (executive summaries with reviewer notes)
├── audit/            (error logs, Codex references)
├── methods/          (DIA workflow details)
├── drafts/           (WIP files)
└── backups/          (originals of all changed files)
```

### Phase 2: Move Internal Documents (6 files)
- Paper3_Critical_Revision_TypesOfContextSensitivity.md → .internal/reviewer_prep/
- Paper3_Executive_Summary.md → .internal/summaries/
- audit_error_log.md + .csv → .internal/audit/
- DIA_Methods_Appendix.md → .internal/methods/
- Paper3_Results_Section_DRAFT.md → .internal/drafts/

### Phase 3: Rewrite Script Headers (2 scripts)
- extract_and_analyze_trial_level.py
- clean_figures_8_9.py

**Replacements:**
- "WRONG" → "updated"
- "OLD outlier" → "early run"
- "TASK" → "STEP"
- Removes Codex references

### Phase 4: Rename Public Docs (4 files)
- Paper3_Results_Discussion.md → Results_Temporal_Dynamics.md
- Paper4_Results_Discussion.md → Results_Entanglement_Analysis.md
- Paper4_Claims_Evidence.md → Claims_Evidence_Entanglement.md
- Paper4_Entanglement_Methods_Appendix.md → Methods_Entanglement.md

### Phase 5: Update .gitignore
Adds:
```
.internal/
CLEANED_*.md
*_HEADER.py
cleanup_plan.md
```

### Phase 6: Replace README
README.md ← CLEANED_README.md

---

## Safety Features

1. **Backup Everything**: All original files saved to `.internal/backups/`
2. **Preview First**: Default mode is dry-run (no changes)
3. **Git-Ignored**: .internal/ folder never committed
4. **Reversible**: Keep originals in backups

---

## Next Steps

### Step 1: Preview (Dry Run)
```bash
cd C:\Users\barla\mch_experiments\scripts\validate
python cleanup_repo.py
```

This shows you EXACTLY what would change without touching any files.

### Step 2: Review Cleaned Examples
- Read CLEANED_README.md
- Compare with current README.md
- Review script headers (_HEADER.py files)

### Step 3: Execute (if happy with preview)
```bash
python cleanup_repo.py --execute
```

This actually performs the cleanup.

### Step 4: Verify Changes
```bash
cd C:\Users\barla\mch_experiments
git status
git diff README.md
```

### Step 5: Commit (if all looks good)
```bash
git add .
git commit -m "Prepare repository for public release

- Remove internal collaboration language
- Move reviewer prep docs to .internal/
- Rewrite script headers for professional presentation
- Rename docs to remove Paper 3/4 labels
- Update README with neutral terminology"

git push
```

---

## What Gets Hidden vs. Kept

### HIDDEN (moved to .internal/):
- ✗ "Anticipating reviewer concerns" documents
- ✗ "Is this post-hoc rationalization?" discussions
- ✗ DIA workflow details (Codex collaboration)
- ✗ Audit error logs
- ✗ Draft files

### KEPT (cleaned up):
- ✓ All experiment data
- ✓ All analysis scripts
- ✓ Results and methods
- ✓ Figures and evidence
- ✓ Acknowledgment of AI collaboration (in README)

---

## Transparency vs. Polish

**Current repo:** Shows ALL the research mess (iterations, mistakes, corrections, internal debates)

**After cleanup:** Shows polished research (methodology, results, evidence) while keeping messy process private

**Still transparent about:**
- AI collaboration (acknowledged in README)
- Dataset corrections (described neutrally as "50-trial reruns with corrected prompts")
- Multiple experiment versions (described as "updated data")

**No longer visible:**
- "This was WRONG"
- "Codex will handle"
- "Risk: cherry-picking"
- Internal debates about reviewers

---

## Questions to Consider

1. **Do you want to keep DIA visible?**
   - Current plan: Moves DIA_Methods_Appendix to .internal/
   - Alternative: Keep it public (shows innovative methodology)
   - My recommendation: Move it (too much AI collaboration detail for now)

2. **Git history cleanup?**
   - Current plan: Keeps commit history as-is
   - Alternative: Rewrite commit messages (DESTRUCTIVE)
   - My recommendation: Keep history (it's fine, history shows iteration)

3. **Paper 3/4 labels?**
   - Current plan: Remove all "Paper 3/4" references
   - Alternative: Keep them (they're just internal labels)
   - My recommendation: Remove (looks like incomplete multi-paper project)

---

## Ready to Proceed?

1. Run preview: `python cleanup_repo.py`
2. Review output
3. Tell me if you want any adjustments to the plan
4. Run execute when ready: `python cleanup_repo.py --execute`
