from __future__ import annotations

import sys
from pathlib import Path


# Ensure vendored graphix_zx (in src/) is importable for all tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GZX = SRC / "graphix_zx"
for p in (SRC, SRC_GZX):
    s = str(p)
    if s not in sys.path:
        sys.path.append(s)

