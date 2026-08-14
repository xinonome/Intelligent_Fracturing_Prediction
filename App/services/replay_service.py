"""Replay orchestration independent of the visual widgets."""

from __future__ import annotations

from typing import Any, Callable

from ..core.replay import build_replay_frames
from ..core.timeline import TimelineController
from ..data.registry_loader import RegistryLoader


class ReplayService:
    def __init__(self, registry: RegistryLoader | None = None) -> None:
        self.registry = registry or RegistryLoader()
        self.frames = build_replay_frames(self.registry)
        self.timeline = TimelineController(self.frames)

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
