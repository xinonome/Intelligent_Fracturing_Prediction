from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_engine import DecisionEngine  # noqa: E402
from decision_engine.risk_rules import ABNORMAL_STATES, NORMAL_STATE  # noqa: E402
from human_machine.confirmation_flow import confirmation_status  # noqa: E402


ABNORMAL_STATE = next(iter(ABNORMAL_STATES))


def test_normal_state_is_low_risk() -> None:
    result = DecisionEngine().decide(
        current_state=NORMAL_STATE,
        transition_probabilities={NORMAL_STATE: 0.82, ABNORMAL_STATE: 0.08},
        enkf_abs_residual=1.0,
    )
    assert result["risk_level"] == "low"
    assert result["requires_confirmation"] is False


def test_abnormal_state_requires_confirmation() -> None:
    result = DecisionEngine().decide(
        current_state=NORMAL_STATE,
        transition_probabilities={NORMAL_STATE: 0.18, ABNORMAL_STATE: 0.36},
        enkf_abs_residual=9.2,
    )
    assert result["risk_level"] == "high"
    assert result["requires_confirmation"] is True


def test_permission_blocks_viewer_confirmation() -> None:
    status = confirmation_status(role="viewer", requires_confirmation=True)
    assert status["status"] == "blocked"
    assert status["allowed"] is False
