from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "PyFrac" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from front_stability import (  # noqa: E402
    audit_front_state,
    canonicalize_front_state,
    front_cfl_time_step,
)


def test_front_cfl_ignores_nan_and_caps_fastest_tip():
    limited, changed = front_cfl_time_step(10.0, [np.nan, 2.0, 4.0], 1.0, 2.0, 0.5)
    assert changed
    assert limited == 0.125


def test_front_cfl_does_not_change_stagnant_front():
    limited, changed = front_cfl_time_step(10.0, [0.0, np.nan, -1.0], 1.0, 1.0)
    assert limited == 10.0
    assert not changed


def test_duplicate_tip_metadata_is_aligned_deterministically():
    fracture = SimpleNamespace(
        EltChannel=np.array([0, 1]),
        EltTip=np.array([2, 2, 3]),
        EltCrack=np.array([0, 1, 2, 3]),
        EltRibbon=np.array([4, 5]),
        fully_traversed=np.array([0]),
        l=np.array([0.1, 0.2, 0.3]),
        alpha=np.array([1.0, 2.0, 3.0]),
        v=np.array([4.0, 5.0, 6.0]),
        FillF=np.array([0.1, 0.2, 0.3]),
        ZeroVertex=np.array([0, 1, 2]),
        Ffront=np.arange(12.0).reshape(3, 4),
    )
    repairs = canonicalize_front_state(fracture)
    assert fracture.EltTip.tolist() == [2, 3]
    assert fracture.l.tolist() == [0.1, 0.3]
    assert fracture.Ffront.shape == (2, 4)
    assert repairs
    assert audit_front_state(fracture) == []


def test_semantic_overlap_is_rejected_instead_of_silently_repaired():
    fracture = SimpleNamespace(
        EltChannel=np.array([0, 1]),
        EltTip=np.array([1, 2]),
        EltCrack=np.array([0, 1, 2]),
        EltRibbon=np.array([2, 3]),
        fully_traversed=np.array([0]),
        l=np.array([0.1, 0.2]),
        alpha=np.array([1.0, 2.0]),
        v=np.array([4.0, 5.0]),
        FillF=np.array([0.1, 0.2]),
        ZeroVertex=np.array([0, 1]),
        Ffront=np.zeros((2, 4)),
    )
    assert any("overlap" in error for error in audit_front_state(fracture))
