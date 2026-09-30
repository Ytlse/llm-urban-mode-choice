"""Repository root for the reference files that the package does not copy.

The tests can be run from `mobility_core/` or from the root: the repository root is
designated explicitly instead of depending on the current directory.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MOBILITY_CORE_REPO_ROOT", str(REPO_ROOT))
