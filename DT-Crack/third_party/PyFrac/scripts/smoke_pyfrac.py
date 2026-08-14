from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))
    for name in ("mesh", "properties", "fracture", "controller", "fracture_initialization"):
        __import__(name)
    print(json.dumps({"status": "PASS", "src": str(src), "numpy": np.__version__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
