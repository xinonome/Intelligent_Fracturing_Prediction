"""Safety-gated pump-control adapters.

The project does not know a field vendor's control protocol.  This module
therefore provides a stable command contract, a dry-run adapter, and an
opt-in HTTP adapter.  No write is allowed unless the adapter is explicitly
armed and the safety gate accepts the command.
"""

from __future__ import annotations

import json
import secrets
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


@dataclass(frozen=True)
class PumpSafetyLimits:
    max_pressure_mpa: float = 150.0
    max_sand_ratio_pct: float = 14.0
    max_rate_change_fraction: float = 0.10
    require_data_quality_valid: bool = True
    require_manual_approval: bool = True


@dataclass(frozen=True)
class PumpState:
    stage_id: str
    flow_rate_m3_s: float
    pressure_mpa: float
    sand_ratio_pct: float
    data_quality_valid: bool = True
    emergency_stop: bool = False


@dataclass(frozen=True)
class PumpCommand:
    stage_id: str
    target_rate_m3_s: float
    target_volume_m3: float | None = None
    reason: str = ""
    command_id: str = ""
    created_at_utc: str = ""

    @classmethod
    def create(cls, stage_id: str, target_rate_m3_s: float, *, target_volume_m3: float | None = None, reason: str = "") -> "PumpCommand":
        return cls(
            stage_id=str(stage_id),
            target_rate_m3_s=max(_finite(target_rate_m3_s), 0.0),
            target_volume_m3=None if target_volume_m3 is None else max(_finite(target_volume_m3), 0.0),
            reason=str(reason),
            command_id=secrets.token_hex(12),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class PumpCommandResult:
    accepted: bool
    written: bool
    command_id: str
    status: str
    reason: str
    adapter: str


class PumpSafetyGate:
    """Validate a proposed total-rate command before any adapter write."""

    def __init__(self, limits: PumpSafetyLimits | None = None) -> None:
        self.limits = limits or PumpSafetyLimits()

    def validate(self, state: PumpState, command: PumpCommand, *, approved: bool = False) -> tuple[bool, str]:
        if state.emergency_stop:
            return False, "emergency_stop_active"
        if self.limits.require_data_quality_valid and not state.data_quality_valid:
            return False, "pump_state_data_quality_invalid"
        if state.pressure_mpa >= self.limits.max_pressure_mpa:
            return False, "pressure_at_or_above_limit"
        if state.sand_ratio_pct > self.limits.max_sand_ratio_pct:
            return False, "current_sand_ratio_above_limit"
        if command.target_rate_m3_s < 0.0:
            return False, "negative_target_rate"
        current = max(_finite(state.flow_rate_m3_s), 0.0)
        change_fraction = abs(command.target_rate_m3_s - current) / max(abs(current), 1.0e-12)
        if change_fraction > max(self.limits.max_rate_change_fraction, 0.0) + 1.0e-12:
            return False, "rate_change_limit_exceeded"
        if self.limits.require_manual_approval and not approved:
            return False, "manual_approval_required"
        return True, "safety_gate_passed"


class PumpControlAdapter:
    """Minimal adapter contract used by the controller."""

    name = "abstract"
    write_enabled = False

    def send(self, state: PumpState, command: PumpCommand, *, approved: bool = False) -> PumpCommandResult:
        raise NotImplementedError


class DryRunPumpAdapter(PumpControlAdapter):
    """Persist accepted commands locally without touching a pump system."""

    name = "dry_run"
    write_enabled = False

    def __init__(self, audit_path: str | Path | None = None, gate: PumpSafetyGate | None = None) -> None:
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.gate = gate or PumpSafetyGate()

    def send(self, state: PumpState, command: PumpCommand, *, approved: bool = False) -> PumpCommandResult:
        accepted, reason = self.gate.validate(state, command, approved=approved)
        result = PumpCommandResult(accepted, False, command.command_id, "dry_run_accepted" if accepted else "dry_run_rejected", reason, self.name)
        self._record(state, command, result)
        return result

    def _record(self, state: PumpState, command: PumpCommand, result: PumpCommandResult) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"state": asdict(state), "command": asdict(command), "result": asdict(result)}, ensure_ascii=False) + "\n")


class HttpPumpAdapter(PumpControlAdapter):
    """Opt-in vendor-neutral HTTP adapter.

    The endpoint and payload are deliberately generic.  It must be replaced
    or configured against the vendor's documented interface.  ``write_enabled``
    is false by default and the constructor refuses to arm without an explicit
    opt-in.
    """

    name = "http"

    def __init__(self, endpoint: str, *, gate: PumpSafetyGate | None = None, write_enabled: bool = False, timeout_s: float = 5.0) -> None:
        self.endpoint = str(endpoint)
        self.gate = gate or PumpSafetyGate()
        self.write_enabled = bool(write_enabled)
        self.timeout_s = max(_finite(timeout_s, 5.0), 0.1)

    def send(self, state: PumpState, command: PumpCommand, *, approved: bool = False) -> PumpCommandResult:
        accepted, reason = self.gate.validate(state, command, approved=approved)
        if not accepted:
            return PumpCommandResult(False, False, command.command_id, "rejected", reason, self.name)
        if not self.write_enabled:
            return PumpCommandResult(False, False, command.command_id, "adapter_not_armed", "http_write_disabled", self.name)
        payload = json.dumps({"stage_id": command.stage_id, "target_rate_m3_s": command.target_rate_m3_s, "target_volume_m3": command.target_volume_m3, "command_id": command.command_id, "reason": command.reason}).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if not 200 <= int(response.status) < 300:
                    return PumpCommandResult(False, False, command.command_id, "remote_rejected", f"http_status_{response.status}", self.name)
        except Exception as exc:  # pragma: no cover - vendor/network dependent
            return PumpCommandResult(False, False, command.command_id, "remote_error", f"{type(exc).__name__}: {exc}", self.name)
        return PumpCommandResult(True, True, command.command_id, "written", "remote_accepted", self.name)
