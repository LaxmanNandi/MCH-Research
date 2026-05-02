"""
Repository path helpers
=======================

Use this module instead of hardcoding `C:/Users/barla/mch_experiments` in scripts.

The repo root is resolved from this file's location:
    repo_root() == <repo>/

Convenience accessors:
    data_dir()      == <repo>/data
    docs_dir()      == <repo>/docs
    papers_dir()    == <repo>/papers
    results_dir()   == <repo>/results
    scripts_dir()   == <repo>/scripts

Usage:
    from scripts.shared.paths import repo_root, data_dir
    raw = data_dir() / "medical" / "open_models" / "mch_results_x.json"

Or, for scripts run directly (not as modules), import like this:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.paths import repo_root
"""

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory (parent of scripts/)."""
    # This file lives at <repo>/scripts/shared/paths.py
    return Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    return repo_root() / "data"


def docs_dir() -> Path:
    return repo_root() / "docs"


def papers_dir() -> Path:
    return repo_root() / "papers"


def results_dir() -> Path:
    return repo_root() / "results"


def scripts_dir() -> Path:
    return repo_root() / "scripts"


if __name__ == "__main__":
    print(f"repo_root:    {repo_root()}")
    print(f"data_dir:     {data_dir()}")
    print(f"docs_dir:     {docs_dir()}")
    print(f"papers_dir:   {papers_dir()}")
    print(f"results_dir: {results_dir()}")
    print(f"scripts_dir: {scripts_dir()}")
