from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inversion.balance_causal_evaluation import evaluate_balance_causality
from inversion.cluster_control import ClusterActuatorCapabilities, ClusterAllocationController, ClusterControlUnavailable
from inversion.closed_loop_controller import ClosedLoopConfig, ClosedLoopController
from inversion.pump_control import DryRunPumpAdapter, PumpSafetyGate, PumpSafetyLimits, PumpState
from inversion.stage_liquid_ledger import StageLedgerConfig, replay_stage_sequence


def test_multistage_ledger_releases_and_draws_with_conservation() -> None:
    stages = pd.DataFrame(
        {
            "stage_order": [1, 2, 3],
            "stage_id": ["s1", "s2", "s3"],
            "response_class": ["fast", "slow", "neutral"],
            "planned_liquid_m3": [100.0, 100.0, 100.0],
            "requested_delta_fraction": [-0.10, 0.10, 0.0],
            "actual_liquid_m3": [90.0, 110.0, 100.0],
        }
    )
    decisions, summary = replay_stage_sequence(stages, config=StageLedgerConfig(max_reserve_m3=20.0))
    assert decisions["committed"].all()
    assert decisions.loc[0, "release_m3"] == 10.0
    assert decisions.loc[1, "draw_m3"] == 10.0
    assert summary["bank_balance_m3"] == 0.0
    assert abs(summary["conservation_residual_m3"]) < 1.0e-8
    assert summary["validation_status"] == "pass"


def test_field_mode_requires_actual_volume() -> None:
    stages = pd.DataFrame({"stage_id": ["s1"], "response_class": ["fast"], "planned_liquid_m3": [100.0]})
    with pytest.raises(ValueError, match="actual_liquid_m3"):
        replay_stage_sequence(stages, execution_mode="field_observed")


def test_shadow_controller_does_not_write_or_mutate_ledger() -> None:
    adapter = DryRunPumpAdapter(gate=PumpSafetyGate(PumpSafetyLimits(require_manual_approval=False)))
    controller = ClosedLoopController(adapter, config=ClosedLoopConfig(mode="shadow"))
    row = controller.process_stage(
        stage_id="s1",
        stage_order=1,
        response_class="fast",
        planned_volume_m3=100.0,
        requested_delta_fraction=-0.10,
        current_state=PumpState("s1", 1.0, 80.0, 5.0),
    )
    assert row["field_control_written"] is False
    assert controller.ledger.bank_balance_m3 == 0.0


def test_direct_cluster_control_is_rejected_without_actuator() -> None:
    controller = ClusterAllocationController(ClusterActuatorCapabilities(direct_cluster_flow=False))
    plan = controller.plan_cluster_allocation("s1", 1.0, [1.0, 2.0, 1.0])
    assert plan.mode == "model_only"
    with pytest.raises(ClusterControlUnavailable):
        controller.execute_cluster_allocation(plan, approved=True)


def test_causal_evaluation_does_not_overclaim_observational_replay() -> None:
    stages = pd.DataFrame(
        {
            "policy": ["piggy_bank", "baseline"],
            "balance_before": [0.5, 0.5],
            "balance_after": [0.6, 0.55],
        }
    )
    _, summary = evaluate_balance_causality(stages, design_type="observational")
    assert summary["status"] == "not_estimable"
    assert summary["causal_effect_proven"] is False


def test_causal_evaluation_can_be_estimable_with_declared_matched_design() -> None:
    stages = pd.DataFrame(
        {
            "policy": ["piggy_bank", "piggy_bank", "baseline", "baseline"],
            "balance_before": [0.5, 0.5, 0.5, 0.5],
            "balance_after": [0.7, 0.65, 0.55, 0.54],
        }
    )
    _, summary = evaluate_balance_causality(stages, design_type="matched_control", bootstrap_samples=200)
    assert summary["status"] == "causal_estimable"
    assert summary["causal_effect_proven"] is False
    assert summary["causal_effect_estimate_available"] is True
    assert summary["estimated_effect_treatment_minus_control"] > 0.0
