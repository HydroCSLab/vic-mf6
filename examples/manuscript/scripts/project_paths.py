"""Project-relative paths for the manuscript workflow."""

from pathlib import Path
import os
import sys


def get_project_dir():
    """Return the parent directory containing both repository checkouts."""
    repository_dir = Path(__file__).resolve().parents[3]
    if not (repository_dir / "examples" / "manuscript").is_dir():
        sys.exit("This script must be located in a vic-mf6 checkout.")
    return repository_dir.parent


project_dir = get_project_dir()
analysis_dir = project_dir / "analysis" / "vic-mf6-manuscript"


def find_paper_dir():
    """Find the sibling paper checkout in the project directory."""
    configured = os.environ.get("VICMF6_PAPER_DIR")
    candidates = [Path(configured)] if configured else []
    candidates += [project_dir / "vic-mf6-paper", project_dir / "vic-mf6-paper-submit"]
    for candidate in candidates:
        if candidate and (candidate / "manuscript").is_dir():
            return candidate.resolve()
    sys.exit(
        "No sibling paper checkout found. Clone vic-mf6-paper into the "
        "project directory or set VICMF6_PAPER_DIR."
    )


def project_path(path):
    """Resolve a path relative to the project directory when needed."""
    path = Path(path)
    return path if path.is_absolute() else (project_dir / path).resolve()
