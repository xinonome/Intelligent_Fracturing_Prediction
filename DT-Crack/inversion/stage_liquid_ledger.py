"""Auditable multi-stage Piggy-Bank liquid ledger.

This module models the only liquid that can be redistributed with a
surface-total-rate actuator: liquid that has *not yet been pumped*.  It never
creates liquid and it never changes the allocation inside a completed stage.

The ledger deliberately separates three events:

``propose``
    Create a recommendation without changing the balance.
``approve``
    Record an operator/system approval without changing the balance.
``commit``
    Enter the actual executed stage volume and update the balance.

For a real field run, ``actual_volume_m3`` must come from the pump system.  A
simulation may explicitly use ``execution_mode="simulated"`` to commit the
recommended volume; the audit record keeps that provenance visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd


RESPONSE_CLASSES = {"fast", "slow", "neutral", "unknown"}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


@dataclass(frozen=True)
class StageLedgerConfig:
    """Hard bounds for stage-level liquid redistribution."""

    max_stage_adjustment_fraction: float = 0.10
    max_reserve_m3: float = 1.0e9
    minimum_stage_volume_m3: float = 0.0
    conservation_tolerance_m3: float = 1.0e-8
    require_approval_for_commit: bool = True


@dataclass
class StageLedgerDecision:
    """One proposed or committed stage-level Piggy-Bank action."""

    stage_id: str
    stage_order: int
    response_class: str
    planned_volume_m3: float
    requested_delta_m3: float
    recommended_volume_m3: float
    release_candidate_m3: float
    draw_candidate_m3: float
    approved: bool = False
    committed: bool = False
    actual_volume_m3: float | None = None
    release_m3: float = 0.0
    draw_m3: float = 0.0
    bank_before_m3: float = 0.0
    bank_after_m3: float = 0.0
    status: str = "proposed"
    reason: str = ""
    execution_mode: str = "uncommitted"
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageLiquidLedger:
    """Maintain a conservation-aware cross-stage liquid reserve."""

    def __init__(self, config: StageLedgerConfig | None = None) -> None:
        self.config = config or StageLedgerConfig()
        self.bank_balance_m3 = 0.0
        self._next_stage_order = 0
        self._decisions: list[StageLedgerDecision] = []

    @property
    def decisions(self) -> tuple[StageLedgerDecision, ...]:
        return tuple(self._decisions)

    def snapshot(self) -> dict[str, Any]:
        return {
            "bank_balance_m3": float(self.bank_balance_m3),
            "next_stage_order": int(self._next_stage_order),
            "decisions": [entry.to_dict() for entry in self._decisions],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        balance = _finite(snapshot.get("bank_balance_m3"), 0.0)
        if balance < -self.config.conservation_tolerance_m3:
            raise ValueError("cannot restore a negative Piggy-Bank balance")
        self.bank_balance_m3 = max(balance, 0.0)
        self._next_stage_order = int(snapshot.get("next_stage_order", 0))
        restored: list[StageLedgerDecision] = []
        for raw in snapshot.get("decisions", []):
            if isinstance(raw, dict):
                restored.append(StageLedgerDecision(**raw))
        self._decisions = restored

    def propose(
        self,
        stage_id: str,
        response_class: str,
        planned_volume_m3: float,
        *,
        requested_delta_m3: float | None = None,
        requested_delta_fraction: float | None = None,
        stage_order: int | None = None,
        safety_ok: bool = True,
        data_quality_valid: bool = True,
        uncertainty_high: bool = False,
        manual_hold: bool = False,
        reason: str = "",
    ) -> StageLedgerDecision:
        """Create a stage recommendation without mutating the ledger.

        A fast stage may release volume.  A slow stage may draw only from the
        current balance.  Neutral or unknown stages hold.  Safety/data gates
        always override the requested direction.
        """

        response = str(response_class).strip().lower()
        if response not in RESPONSE_CLASSES:
            raise ValueError(f"unsupported response_class: {response_class}")
        planned = max(_finite(planned_volume_m3), self.config.minimum_stage_volume_m3)
        max_fraction = max(_finite(self.config.max_stage_adjustment_fraction), 0.0)
        if requested_delta_m3 is None:
            fraction = np.clip(_finite(requested_delta_fraction), -max_fraction, max_fraction)
            requested = planned * float(fraction)
        else:
            requested = _finite(requested_delta_m3)
        requested = float(np.clip(requested, -planned, planned * max_fraction))
        order = self._next_stage_order if stage_order is None else int(stage_order)
        base = dict(
            stage_id=str(stage_id),
            stage_order=order,
            response_class=response,
            planned_volume_m3=planned,
            requested_delta_m3=requested,
            recommended_volume_m3=planned,
            release_candidate_m3=0.0,
            draw_candidate_m3=0.0,
            bank_before_m3=float(self.bank_balance_m3),
            reason=reason,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        if manual_hold:
            return StageLedgerDecision(**base, status="hold_manual", reason=reason or "manual_hold")
        if not safety_ok:
            return StageLedgerDecision(**base, status="hold_safety", reason=reason or "safety_gate")
        if not data_quality_valid:
            return StageLedgerDecision(**base, status="hold_data_quality", reason=reason or "data_quality_gate")
        if uncertainty_high:
            return StageLedgerDecision(**base, status="hold_uncertainty", reason=reason or "uncertainty_gate")
        if response == "fast" and requested < 0.0:
            release = min(-requested, planned)
            base.update(
                recommended_volume_m3=planned - release,
                release_candidate_m3=release,
                status="release_proposed",
                reason=reason or "fast_stage_release",
            )
        elif response == "slow" and requested > 0.0:
            draw = min(requested, self.bank_balance_m3, max(planned * max_fraction, 0.0))
            base.update(
                recommended_volume_m3=planned + draw,
                draw_candidate_m3=draw,
                status="draw_proposed" if draw > 0.0 else "hold_no_reserve",
                reason=reason or ("slow_stage_draw" if draw > 0.0 else "slow_stage_no_reserve"),
            )
        else:
            base.update(status="hold_no_direction", reason=reason or "response_class_or_direction_mismatch")
        return StageLedgerDecision(**base)

    def approve(self, decision: StageLedgerDecision, *, approved: bool = True, reason: str = "") -> StageLedgerDecision:
        """Approve or reject a proposal without changing the reserve."""

        decision.approved = bool(approved)
        if approved:
            decision.status = "approved"
            decision.reason = reason or decision.reason or "approved"
        else:
            decision.status = "rejected"
            decision.reason = reason or "rejected_by_operator_or_policy"
        return decision

    def commit(
        self,
        decision: StageLedgerDecision,
        *,
        actual_volume_m3: float | None = None,
        execution_mode: str = "field_observed",
        force: bool = False,
    ) -> StageLedgerDecision:
        """Commit actual stage volume and update the reserve.

        ``field_observed`` requires a measured actual volume.  ``simulated``
        may use the recommendation explicitly.  ``shadow`` is never allowed
        to mutate the ledger unless ``force=True`` is supplied by a test or a
        controlled replay.
        """

        mode = str(execution_mode).strip().lower()
        if mode not in {"field_observed", "simulated", "shadow", "manual"}:
            raise ValueError(f"unsupported execution_mode: {execution_mode}")
        if self.config.require_approval_for_commit and not decision.approved and not force:
            decision.status = "commit_rejected_not_approved"
            decision.reason = "approval_required_before_commit"
            return decision
        if mode == "shadow" and not force:
            decision.status = "shadow_not_committed"
            decision.execution_mode = mode
            decision.reason = "shadow_mode_does_not_mutate_ledger"
            return decision
        if actual_volume_m3 is None:
            if mode == "simulated":
                actual = float(decision.recommended_volume_m3)
            else:
                decision.status = "commit_rejected_missing_actual_volume"
                decision.reason = "actual_volume_required_for_non_simulated_commit"
                return decision
        else:
            actual = max(_finite(actual_volume_m3), self.config.minimum_stage_volume_m3)
        if decision.committed:
            raise ValueError(f"stage {decision.stage_id} is already committed")

        bank_before = float(self.bank_balance_m3)
        release = 0.0
        draw = 0.0
        delta_from_plan = actual - float(decision.planned_volume_m3)
        if decision.response_class == "fast" and delta_from_plan < 0.0:
            release = min(-delta_from_plan, decision.planned_volume_m3)
        elif decision.response_class == "slow" and delta_from_plan > 0.0:
            draw = delta_from_plan
        elif abs(delta_from_plan) > self.config.conservation_tolerance_m3:
            decision.status = "commit_rejected_response_mismatch"
            decision.reason = "actual_volume_direction_does_not_match_response_class"
            return decision

        if draw > bank_before + self.config.conservation_tolerance_m3:
            decision.status = "commit_rejected_insufficient_reserve"
            decision.reason = "actual_slow_stage_exceeds_bank_balance"
            return decision
        new_balance = bank_before + release - draw
        if new_balance > self.config.max_reserve_m3 + self.config.conservation_tolerance_m3:
            decision.status = "commit_rejected_reserve_cap"
            decision.reason = "reserve_cap_exceeded"
            return decision
        if new_balance < -self.config.conservation_tolerance_m3:
            decision.status = "commit_rejected_negative_reserve"
            decision.reason = "negative_reserve_would_result"
            return decision

        self.bank_balance_m3 = max(new_balance, 0.0)
        self._next_stage_order = max(self._next_stage_order, decision.stage_order + 1)
        decision.actual_volume_m3 = actual
        decision.release_m3 = float(release)
        decision.draw_m3 = float(draw)
        decision.bank_before_m3 = bank_before
        decision.bank_after_m3 = float(self.bank_balance_m3)
        decision.approved = True
        decision.committed = True
        decision.status = "committed"
        decision.execution_mode = mode
        decision.reason = decision.reason or "committed"
        self._decisions.append(decision)
        return decision

    def rollback(self, stage_id: str) -> StageLedgerDecision:
        """Undo the latest committed stage and restore its prior balance."""

        matches = [entry for entry in self._decisions if entry.stage_id == str(stage_id) and entry.committed]
        if not matches:
            raise KeyError(f"no committed stage found for rollback: {stage_id}")
        target = matches[-1]
        if abs(self.bank_balance_m3 - target.bank_after_m3) > self.config.conservation_tolerance_m3:
            raise RuntimeError("rollback is only allowed for the latest ledger balance")
        self.bank_balance_m3 = float(target.bank_before_m3)
        target.committed = False
        target.status = "rolled_back"
        target.reason = "explicit_rollback"
        target.execution_mode = "rollback"
        self._decisions.remove(target)
        self._next_stage_order = max((entry.stage_order + 1 for entry in self._decisions), default=0)
        return target

    def audit_frame(self) -> pd.DataFrame:
        rows = [entry.to_dict() for entry in self._decisions]
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        frame = self.audit_frame()
        committed = frame[frame.get("committed", pd.Series(dtype=bool)) == True] if not frame.empty else frame
        total_release = float(pd.to_numeric(committed.get("release_m3"), errors="coerce").sum()) if not committed.empty else 0.0
        total_draw = float(pd.to_numeric(committed.get("draw_m3"), errors="coerce").sum()) if not committed.empty else 0.0
        return {
            "ledger_status": "committed" if not committed.empty else "empty",
            "committed_stage_count": int(len(committed)),
            "bank_balance_m3": float(self.bank_balance_m3),
            "total_release_m3": total_release,
            "total_draw_m3": total_draw,
            "conservation_residual_m3": total_release - total_draw - float(self.bank_balance_m3),
            "max_reserve_m3": float(self.config.max_reserve_m3),
        }


def replay_stage_sequence(
    stages: pd.DataFrame,
    *,
    config: StageLedgerConfig | None = None,
    execution_mode: str = "field_observed",
    approve_all: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay a stage CSV through the real ledger.

    Required columns: ``stage_id``, ``response_class``,
    ``planned_liquid_m3``.  ``actual_liquid_m3`` is required for field replay;
    it may be omitted in simulated mode.  Optional ``requested_delta_m3`` or
    ``requested_delta_fraction`` controls the recommendation.
    """

    required = {"stage_id", "response_class", "planned_liquid_m3"}
    missing = required - set(stages.columns)
    if missing:
        raise ValueError(f"stage sequence missing columns: {sorted(missing)}")
    if execution_mode == "field_observed" and "actual_liquid_m3" not in stages.columns:
        raise ValueError("field_observed mode requires actual_liquid_m3")
    ledger = StageLiquidLedger(config)
    decisions: list[dict[str, Any]] = []
    frame = stages.copy()
    if "stage_order" not in frame.columns:
        frame.insert(0, "stage_order", np.arange(len(frame), dtype=int))
    for _, row in frame.sort_values("stage_order").iterrows():
        decision = ledger.propose(
            row["stage_id"],
            row["response_class"],
            row["planned_liquid_m3"],
            requested_delta_m3=row.get("requested_delta_m3") if "requested_delta_m3" in row else None,
            requested_delta_fraction=row.get("requested_delta_fraction") if "requested_delta_fraction" in row else None,
            stage_order=int(row["stage_order"]),
            safety_ok=bool(row.get("safety_ok", True)),
            data_quality_valid=bool(row.get("data_quality_valid", True)),
            uncertainty_high=bool(row.get("uncertainty_high", False)),
            manual_hold=bool(row.get("manual_hold", False)),
            reason=str(row.get("reason", "")),
        )
        # A neutral stage is still a real ledger event with zero release/draw.
        # Safety/data/uncertainty holds remain uncommitted even when the
        # replay is configured to approve ordinary recommendations.
        non_committable = {"hold_manual", "hold_safety", "hold_data_quality", "hold_uncertainty"}
        if approve_all and decision.status not in non_committable:
            ledger.approve(decision, approved=True)
        actual = row.get("actual_liquid_m3") if "actual_liquid_m3" in row else None
        committed = ledger.commit(decision, actual_volume_m3=actual, execution_mode=execution_mode)
        decisions.append(committed.to_dict())
    result = pd.DataFrame(decisions)
    summary = ledger.summary()
    summary.update(
        {
            "execution_mode": execution_mode,
            "input_stage_count": int(len(frame)),
            "committed_stage_count": int(result["committed"].sum()) if "committed" in result else 0,
            "validation_status": "pass" if abs(float(summary["conservation_residual_m3"])) <= (config or StageLedgerConfig()).conservation_tolerance_m3 else "fail",
        }
    )
    return result, summary
