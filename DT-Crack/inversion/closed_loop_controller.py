"""Stage-level closed-loop orchestration with explicit operating modes.

Modes:

``open_loop``
    Compute and log recommendations only.
``shadow``
    Produce a command-shaped recommendation and send it only to a dry-run or
    shadow adapter; neither the pump nor the ledger is mutated.
``manual``
    A human approval is required before a command is sent and the stage can be
    committed after actual volume is received.
``closed_loop``
    Requires an armed write-capable adapter and a safety gate configured for
    autonomous operation.  Even then, the ledger is committed only with the
    actual volume reported by the pump system.

This is a real control framework, but it cannot create a field connection or
field authorization that the project does not possess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from inversion.pump_control import PumpCommand, PumpCommandResult, PumpControlAdapter, PumpSafetyGate, PumpState
from inversion.stage_liquid_ledger import StageLedgerConfig, StageLedgerDecision, StageLiquidLedger


VALID_MODES = {"open_loop", "shadow", "manual", "closed_loop"}


@dataclass(frozen=True)
class ClosedLoopConfig:
    mode: str = "shadow"
    stage_duration_s: float = 60.0
    require_actual_volume_for_commit: bool = True
    max_stage_adjustment_fraction: float = 0.10
    max_reserve_m3: float = 1.0e9


class ClosedLoopController:
    """Coordinate one stage recommendation, command audit and ledger commit."""

    def __init__(self, adapter: PumpControlAdapter, *, config: ClosedLoopConfig | None = None, gate: PumpSafetyGate | None = None) -> None:
        self.config = config or ClosedLoopConfig()
        mode = str(self.config.mode).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported closed-loop mode: {self.config.mode}")
        self.adapter = adapter
        self.gate = gate or getattr(adapter, "gate", None) or PumpSafetyGate()
        self.ledger = StageLiquidLedger(StageLedgerConfig(max_stage_adjustment_fraction=self.config.max_stage_adjustment_fraction, max_reserve_m3=self.config.max_reserve_m3))
        self.audit: list[dict[str, Any]] = []
        if mode == "closed_loop" and not bool(getattr(adapter, "write_enabled", False)):
            raise RuntimeError("closed_loop mode requires an armed write-capable pump adapter")

    @property
    def mode(self) -> str:
        return str(self.config.mode).lower()

    def process_stage(
        self,
        *,
        stage_id: str,
        stage_order: int,
        response_class: str,
        planned_volume_m3: float,
        current_state: PumpState,
        requested_delta_fraction: float | None = None,
        requested_delta_m3: float | None = None,
        data_quality_valid: bool = True,
        uncertainty_high: bool = False,
        safety_ok: bool = True,
        approved: bool = False,
        actual_volume_m3: float | None = None,
    ) -> dict[str, Any]:
        decision = self.ledger.propose(
            stage_id,
            response_class,
            planned_volume_m3,
            requested_delta_m3=requested_delta_m3,
            requested_delta_fraction=requested_delta_fraction,
            stage_order=stage_order,
            data_quality_valid=data_quality_valid,
            uncertainty_high=uncertainty_high,
            safety_ok=safety_ok,
        )
        if decision.status.endswith("proposed") and (approved or self.mode == "closed_loop"):
            self.ledger.approve(decision, approved=True)
        duration = max(float(self.config.stage_duration_s), 1.0e-9)
        command = PumpCommand.create(
            stage_id,
            decision.recommended_volume_m3 / duration,
            target_volume_m3=decision.recommended_volume_m3,
            reason=decision.reason,
        )
        command_result: PumpCommandResult
        if self.mode == "open_loop":
            command_result = PumpCommandResult(False, False, command.command_id, "not_written_open_loop", "recommendation_only", "none")
        else:
            command_result = self.adapter.send(current_state, command, approved=approved or self.mode == "closed_loop")

        committed = decision
        if self.mode == "closed_loop" and command_result.accepted and actual_volume_m3 is not None:
            committed = self.ledger.commit(decision, actual_volume_m3=actual_volume_m3, execution_mode="field_observed")
        elif self.mode == "manual" and command_result.accepted and actual_volume_m3 is not None:
            committed = self.ledger.commit(decision, actual_volume_m3=actual_volume_m3, execution_mode="field_observed")
        elif self.mode == "shadow":
            committed = self.ledger.commit(decision, execution_mode="shadow")
        elif self.mode == "open_loop":
            committed.status = "recommendation_not_committed"
            committed.reason = committed.reason or "open_loop_no_commit"
        row = {
            "stage_id": str(stage_id),
            "stage_order": int(stage_order),
            "mode": self.mode,
            "response_class": str(response_class),
            "planned_volume_m3": float(planned_volume_m3),
            "recommended_volume_m3": float(decision.recommended_volume_m3),
            "recommended_rate_m3_s": float(command.target_rate_m3_s),
            "command_id": command.command_id,
            "command_accepted": bool(command_result.accepted),
            "command_written": bool(command_result.written),
            "command_status": command_result.status,
            "command_reason": command_result.reason,
            "ledger_status": committed.status,
            "actual_volume_m3": committed.actual_volume_m3,
            "bank_before_m3": float(committed.bank_before_m3),
            "bank_after_m3": float(committed.bank_after_m3),
            "release_m3": float(committed.release_m3),
            "draw_m3": float(committed.draw_m3),
            "field_control_written": bool(command_result.written),
        }
        self.audit.append(row)
        return row

    def write_audit(self, output_dir: str | Path) -> dict[str, str]:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.audit)
        csv_path = output / "closed_loop_stage_audit.csv"
        json_path = output / "closed_loop_summary.json"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary = {
            "mode": self.mode,
            "stage_count": int(len(frame)),
            "field_control_written": bool(frame["field_control_written"].any()) if not frame.empty else False,
            "ledger": self.ledger.summary(),
            "automatic_execution_verified": self.mode == "closed_loop" and bool(frame["field_control_written"].all()) if not frame.empty else False,
            "limitations": [
                "A field closed-loop claim requires vendor acknowledgement and independent safety acceptance logs.",
                "Cluster-level control remains unavailable unless an actuator advertises direct_cluster_flow capability.",
            ],
        }
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return {"audit_csv": str(csv_path), "summary_json": str(json_path)}
