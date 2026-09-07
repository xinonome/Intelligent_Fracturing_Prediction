"""Replay orchestration independent of the visual widgets."""

from __future__ import annotations

from typing import Any, Callable

from ..core.replay import build_replay_frames
from ..core.timeline import TimelineController
from ..core.model_runtime import resolve_runtime_selection
from ..data.registry_loader import RegistryLoader


class ReplayService:
    def __init__(self, registry: RegistryLoader | None = None) -> None:
        self.registry = registry or RegistryLoader()
        # Replay data is large and is only needed by the integrated/DT/HMI
        # pages.  Do not build it while the QApplication is being created;
        # the window can render first and the selected page will request it.
        self.timeline = TimelineController([])
        self._loaded = False
        self.agent_model_id = str(resolve_runtime_selection().agent_policy or "sac").lower()
        self._frame_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def _frame_cache_key(
        self,
        scenario_id: str | None = None,
        dataset_id: str | None = None,
        agent_model: str | None = None,
    ) -> tuple[str, str, str]:
        return (
            str(dataset_id or getattr(self.registry, "dataset_id", "default")),
            str(scenario_id or getattr(self.registry, "scenario_id", "default")),
            str(agent_model or self.agent_model_id or "default"),
        )

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        key = self._frame_cache_key(agent_model=self.agent_model_id)
        frames = self._frame_cache.get(key)
        if frames is None:
            frames = build_replay_frames(self.registry, agent_model=self.agent_model_id)
            self._frame_cache[key] = frames
        # ``TimelineController.set_frames`` emits frameChanged immediately.
        # UI callbacks read ``controller.frames`` during that emission.  Mark
        # the service loaded before replacing the timeline, otherwise the
        # callback re-enters ensure_loaded and recursively rebuilds the same
        # dataset until the APP raises RecursionError.
        self._loaded = True
        try:
            self.timeline.set_frames(frames)
        except Exception:
            self._loaded = False
            raise

    @property
    def frames(self):
        self.ensure_loaded()
        return self.timeline.frames

    @property
    def index(self):
        return self.timeline.index

    @property
    def current(self):
        self.ensure_loaded()
        return self.timeline.current

    @property
    def frameChanged(self):
        return self.timeline.frameChanged

    def set_index(self, index: int, emit: bool = True):
        self.ensure_loaded()
        return self.timeline.set_index(index, emit=emit)

    def step(self, delta: int):
        self.ensure_loaded()
        return self.timeline.step(delta)

    def set_time(self, time_s: float):
        self.ensure_loaded()
        return self.timeline.set_time(time_s)

    def set_scenario(self, scenario_id: str) -> None:
        frames = self.prepare_scenario_frames(scenario_id)
        self.apply_scenario_frames(scenario_id, frames)

    def prepare_scenario_frames(
        self,
        scenario_id: str,
        dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare a scenario without changing the active visible scenario."""

        selected_dataset = str(dataset_id or getattr(self.registry, "dataset_id", ""))
        key = self._frame_cache_key(scenario_id, selected_dataset)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        worker_registry = RegistryLoader(
            registry=self.registry.registry,
            scenario_id=str(scenario_id),
            dataset_id=selected_dataset,
            snapshot=self.registry.snapshot,
        )
        return build_replay_frames(worker_registry, agent_model=self.agent_model_id)

    def apply_scenario_frames(
        self,
        scenario_id: str,
        frames: list[dict[str, Any]],
        dataset_id: str | None = None,
        *,
        emit: bool = True,
    ) -> None:
        """Commit a prepared scenario on the UI thread."""

        self.registry.set_scenario(str(scenario_id))
        if dataset_id and str(dataset_id) != str(getattr(self.registry, "dataset_id", "")):
            self.registry.set_dataset(str(dataset_id))
            # set_dataset preserves the selected scenario when the dataset
            # declares support for it; restore it explicitly for old entries.
            self.registry.set_scenario(str(scenario_id))
        self._frame_cache[self._frame_cache_key(scenario_id, dataset_id, self.agent_model_id)] = frames
        self._loaded = True
        self.timeline.set_frames(frames, emit=emit)

    def prepare_agent_model_frames(self, agent_model: str) -> list[dict[str, Any]]:
        """Prepare a different frozen agent replay for the active scenario."""

        model_id = str(agent_model or "").strip().lower()
        if not model_id:
            raise ValueError("智能体模型不能为空")
        key = self._frame_cache_key(agent_model=model_id)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        worker_registry = RegistryLoader(
            registry=self.registry.registry,
            scenario_id=str(self.registry.scenario_id),
            dataset_id=str(self.registry.dataset_id),
            snapshot=self.registry.snapshot,
        )
        frames = build_replay_frames(worker_registry, agent_model=model_id)
        self._frame_cache[key] = frames
        return frames

    def set_agent_model(self, agent_model: str) -> None:
        """Switch the HMI page to a real model-specific replay source."""

        model_id = str(agent_model or "").strip().lower()
        if not model_id:
            raise ValueError("智能体模型不能为空")
        frames = self.prepare_agent_model_frames(model_id)
        if not frames:
            raise ValueError(f"{model_id.upper()} 没有可用的回放数据")
        self.agent_model_id = model_id
        self._loaded = True
        self.timeline.set_frames(frames)

    def set_dataset(self, dataset_id: str) -> None:
        self.registry.set_dataset(dataset_id)
        self._loaded = False
        self.ensure_loaded()

    def frame(self, index: int | None = None) -> dict[str, Any] | None:
        self.ensure_loaded()
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
