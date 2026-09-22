"""Puts the extension root on sys.path.

The modules under test live in the `merge_studio` package beside this suite,
not in an installed distribution, so `python -m pytest` from the root cannot
find the package on its own. This is the same insert `scripts/merge_studio_ui`
does when Forge loads the extension.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
