from __future__ import annotations

from dataclasses import asdict, dataclass

from .policy_recommender import recommend
from .risk_rules import NORMAL_STATE, requires_confirmation, risk_level
from .uncertainty import combine_uncertainty, probability_uncertainty, residual_uncertainty


@dataclass
class DecisionResult:
    current_state: str
    predicted_next_state: str
    transition_probability: float
    uncertainty: str
    risk_level: str
    recommendation: str
    requires_confirmation: bool


class DecisionEngine:
    def decide(
        self,
        current_state: str,
        transition_probabilities: dict[str, float],
        enkf_abs_residual: float = 0.0,
        action_type: str = "advisory",
    ) -> dict:
        if not transition_probabilities:
            transition_probabilities = {NORMAL_STATE: 1.0}
        next_state, prob = max(transition_probabilities.items(), key=lambda kv: kv[1])
        uncertainty = combine_uncertainty(
            probability_uncertainty(float(prob)),
            residual_uncertainty(float(enkf_abs_residual)),
        )
        risk = risk_level(next_state, float(prob), uncertainty)
        result = DecisionResult(
            current_state=current_state,
            predicted_next_state=next_state,
            transition_probability=float(prob),
            uncertainty=uncertainty,
            risk_level=risk,
            recommendation=recommend(next_state, risk),
            requires_confirmation=requires_confirmation(risk, action_type),
        )
        return asdict(result)
