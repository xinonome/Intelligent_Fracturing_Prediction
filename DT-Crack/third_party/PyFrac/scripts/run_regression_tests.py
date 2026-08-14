from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    # Compatibility aliases are process-local test harness shims. The vendored
    # upstream source remains unchanged.
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    return int(pytest.main([str(root / "regression_tests"), "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
