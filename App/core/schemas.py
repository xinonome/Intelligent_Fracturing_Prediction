"""Stable schemas shared by the three contract pages and the replay engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataQuality:
    valid: bool = True
    source: str = "frozen_artifact"
    interpolation: str = "none"


@dataclass
class FSLState:
    working_type: str = "unknown"
    normal_probability: float | None = None
    abnormal_probability: float | None = None
    abnormal_type: str = "none"
    rule_hits: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class DTState:
    surface_pressure_mpa: float | None = None
    prior_bottomhole_pressure_mpa: float | None = None
    observed_bottomhole_pressure_mpa: float | None = None
    bottomhole_pressure_mpa: float | None = None
    net_pressure_mpa: float | None = None
    cumulative_liquid_m3: float | None = None
    cumulative_sand_t: float | None = None
    prior_parameters: dict[str, Any] = field(default_factory=dict)
    posterior_parameters: dict[str, Any] = field(default_factory=dict)
    prior_half_lengths_m: list[float] = field(default_factory=list)
    posterior_half_lengths_m: list[float] = field(default_factory=list)
    prior_error: float | None = None
    posterior_error: float | None = None
    prior_pressure_error: float | None = None
    posterior_pressure_error: float | None = None
    runtime_ms: float | None = None
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class HMIState:
    current_flow_m3_min: float | None = None
    current_sand_ratio_percent: float | None = None
    recommended_flow_m3_min: float | None = None
    recommended_sand_ratio_percent: float | None = None
    high_level_action: str = "unknown"
    risk_level: str = "unknown"
    uncertainty: str = "unknown"
    requires_confirmation: bool = True
    reward_components: dict[str, Any] = field(default_factory=dict)
    warning_5min: dict[str, Any] = field(default_factory=dict)
    validation_180s: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayFrame:
    """One aligned state of the integrated demonstration.

    The legacy flat fields are intentionally not part of this schema.  The
    replay service adds them when adapting to the old UI/test API, while new
    pages consume the typed ``fsl``, ``dt`` and ``hmi`` sections.
    """

    frame_id: int
    time_s: float
    stage: str = "08"
    fsl: FSLState = field(default_factory=FSLState)
    dt: DTState = field(default_factory=DTState)
    hmi: HMIState = field(default_factory=HMIState)
    alignment: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayFrame":
        def make(item: str, target: type[Any]) -> Any:
            raw = payload.get(item, {})
            return target(**raw) if isinstance(raw, dict) else target()

        return cls(
            frame_id=int(payload.get("frame_id", payload.get("replay_index", 1))),
            time_s=float(payload.get("time_s", 0.0)),
            stage=str(payload.get("stage", "08")),
            fsl=make("fsl", FSLState),
            dt=make("dt", DTState),
            hmi=make("hmi", HMIState),
            alignment=dict(payload.get("alignment", {})),
            events=list(payload.get("events", [])),
        )


def frame_payload(frame: ReplayFrame | dict[str, Any]) -> dict[str, Any]:
    return frame.to_dict() if isinstance(frame, ReplayFrame) else dict(frame)
