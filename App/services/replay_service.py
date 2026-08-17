"""Replay orchestration independent of the visual widgets."""

from __future__ import annotations

from typing import Any, Callable

from ..core.replay import build_replay_frames
from ..core.timeline import TimelineController
from ..data.registry_loader import RegistryLoader


class ReplayService:
    def __init__(self, registry: RegistryLoader | None = None) -> None:
        self.registry = registry or RegistryLoader()
        self.timeline = TimelineController(build_replay_frames(self.registry))

    @property
    def frames(self):
        return self.timeline.frames

    @property
    def index(self):
        return self.timeline.index

    @property
    def current(self):
        return self.timeline.current

    @property
    def frameChanged(self):
        return self.timeline.frameChanged

    def set_index(self, index: int, emit: bool = True):
        return self.timeline.set_index(index, emit=emit)

    def step(self, delta: int):
        return self.timeline.step(delta)

    def set_time(self, time_s: float):
        return self.timeline.set_time(time_s)

    def set_scenario(self, scenario_id: str) -> None:
        self.registry.set_scenario(scenario_id)
        self.timeline.set_frames(build_replay_frames(self.registry))

    def frame(self, index: int | None = None) -> dict[str, Any] | None:
        if index is None:
            return self.timeline.current
        return self.timeline.set_index(index)

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.timeline.frameChanged.connect(callback)

    def event_indices(self) -> dict[str, int]:
        events: dict[str, int] = {}
        for index, frame in enumerate(self.frames):
            option = frame.get("hmi_option", "")
            risk = frame.get("decision", {}).get("risk_level", "")
            if option and option not in events:
                events[str(option)] = index
            if risk == "high" and "high_risk" not in events:
                events["high_risk"] = index
        return events
