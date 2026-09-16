"""Puts the repository root on sys.path.

The modules under test sit at the repository root rather than in an installed
package, so `python -m pytest` from the root cannot import them on its own.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
