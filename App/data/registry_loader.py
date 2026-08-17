"""Registry-first access to frozen summaries, tables, caches and HTML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactRegistry
from ..core.paths import PATHS, resolve


class RegistryLoader:
    def __init__(self, registry: ArtifactRegistry | None = None, scenario_id: str | None = None) -> None:
        self.registry = registry or ArtifactRegistry()
        self.snapshot = self.registry.snapshot()
        self.scenario_id = scenario_id or self.module("dt").get("default_scenario", "das_cluster_observation")

    def set_scenario(self, scenario_id: str) -> None:
        scenarios = self.module("dt").get("scenarios", {})
        if scenario_id not in scenarios:
            raise ValueError(f"unknown DT scenario: {scenario_id}")
        self.scenario_id = scenario_id

    def module(self, name: str) -> dict[str, Any]:
        return self.snapshot.get("modules", {}).get(name, {})

    def summary(self, name: str) -> dict[str, Any]:
        value = self.module(name).get("summary", {})
        return value if isinstance(value, dict) else {}

    def _candidate_values(self, module: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for item in module.get("figures", []) or []:
            values.append(str(item))
        for item in module.get("tables", []) or []:
            values.append(str(item.get("path")) if isinstance(item, dict) else str(item))
        files = module.get("files", {}) or {}
        for key in ("figures", "tables"):
            for item in files.get(key, []) or []:
                values.append(str(item.get("path")) if isinstance(item, dict) else str(item))
        outputs = module.get("summary", {}).get("outputs", {})
        if isinstance(outputs, dict):
            values.extend(str(value) for value in outputs.values())
        return values

    def path(self, value: str | Path | None) -> Path | None:
        return resolve(value)

    def table(self, name: str, filename: str) -> Path | None:
        module = self.module(name)
        for value in self._candidate_values(module):
            path = resolve(value)
            if path and path.name == filename and path.exists():
                return path
        return None

    def output(self, name: str, key: str) -> Path | None:
        outputs = self.summary(name).get("outputs", {})
        if isinstance(outputs, dict):
            path = resolve(outputs.get(key))
            if path and path.exists():
                return path
        return None

    def frame_source(self) -> Path | None:
        value = self.module("dt").get("frame_source")
        path = resolve(value)
        return path if path and path.exists() else PATHS.dt_cache if PATHS.dt_cache.exists() else None

    def scenario(self, scenario_id: str | None = None) -> dict[str, Any]:
        selected = scenario_id or self.scenario_id
        return dict(self.module("dt").get("scenarios", {}).get(selected, {}))

    def html(self) -> Path | None:
        value = self.module("dt").get("html")
        path = resolve(value)
        return path if path else PATHS.dt_html

    def source_status(self, name: str) -> dict[str, Any]:
        module = self.module(name)
        return {
            "status": module.get("status", "not_available"),
            "reason": module.get("status_reason", ""),
            "summary": self.summary(name),
            "limitations": module.get("limitations", []),
        }
