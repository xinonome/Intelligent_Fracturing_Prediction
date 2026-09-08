"""Registry-first access to frozen summaries, tables, caches and HTML."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ..core.artifacts import ArtifactRegistry
from ..core.paths import PATHS, resolve
from ..core.model_runtime import resolve_runtime_selection
from .dt_dataset_registry import clear_dataset_catalog_cache, get_dataset, load_dataset_catalog


class RegistryLoader:
    def __init__(
        self,
        registry: ArtifactRegistry | None = None,
        scenario_id: str | None = None,
        dataset_id: str | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or ArtifactRegistry()
        # A scenario worker only needs a read-only copy of the already loaded
        # registry.  Rebuilding the snapshot includes parsing registered JSON
        # summaries and inspecting the embedded HTML files, which needlessly
        # delays every scenario switch.
        self.snapshot = snapshot if isinstance(snapshot, dict) else self.registry.snapshot()
        self._scenario_registry = self._load_scenario_registry()
        self.scenario_id = scenario_id or self._scenario_registry.get(
            "default_scenario",
            self.module("dt").get("default_scenario", "das_cluster_observation"),
        )
        catalog = load_dataset_catalog()
        self.dataset_id = dataset_id or catalog.get("default_dataset_id", "jy84_z1_stage08")
        self._dataset_cache: dict[str, dict[str, Any]] = {}

    def set_scenario(self, scenario_id: str) -> None:
        scenarios = self._scenario_registry.get("scenarios", {}) or self.module("dt").get("scenarios", {})
        if scenario_id not in scenarios:
            raise ValueError(f"unknown DT scenario: {scenario_id}")
        self.scenario_id = scenario_id

    def refresh_catalog(self) -> None:
        """Forget cached dataset metadata after an APP import operation."""

        clear_dataset_catalog_cache()
        self._dataset_cache.clear()

    def _load_scenario_registry(self) -> dict[str, Any]:
        path = Path(__file__).resolve().parents[1] / "config" / "dt_scenario_registry.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def dataset_catalog(self) -> dict[str, Any]:
        return load_dataset_catalog()

    def dataset(self, dataset_id: str | None = None) -> dict[str, Any]:
        selected_id = str(dataset_id or self.dataset_id)
        cached = self._dataset_cache.get(selected_id)
        if cached is None:
            cached = get_dataset(selected_id)
            self._dataset_cache[selected_id] = cached
        return cached

    def dataset_ready(self, dataset_id: str | None = None) -> bool:
        selected = self.dataset(dataset_id)
        if selected.get("data_scope") == "composite" or selected.get("evolution_supported") is False:
            return False
        cache = self.path(selected.get("cache_source"))
        html_values = selected.get("html_by_scenario", {}) or {}
        html_ready = any(bool(self.path(value) and self.path(value).exists()) for value in html_values.values())
        return bool(cache and cache.exists() and html_ready)

    def dataset_source_ready(self, dataset_id: str | None = None) -> bool:
        """Raw data availability is independent of a finished DT animation."""
        selected = self.dataset(dataset_id)
        if selected.get("data_scope") == "composite":
            return False
        source = self.path(selected.get("pressure_source"))
        return bool(source and source.is_file())

    def dataset_supports_scenario(self, dataset_id: str, scenario_id: str | None = None) -> bool:
        """Return whether a dataset belongs in the selected DT scenario.

        The APP must not present a pressure-only construction export in the
        DAS selector (or a DAS stage in the pressure-only selector).  New
        registry entries may declare ``supported_scenarios`` explicitly; the
        adapter/source fallback keeps older release packages compatible.
        """

        selected = self.dataset(dataset_id)
        if selected.get("data_scope") == "composite" or selected.get("evolution_supported") is False:
            return False
        selected_scenario = str(scenario_id or self.scenario_id)
        declared = selected.get("supported_scenarios")
        if isinstance(declared, list):
            return selected_scenario in {str(value) for value in declared}
        has_fiber = bool(selected.get("fiber_source"))
        if selected_scenario == "das_cluster_observation":
            return has_fiber
        if selected_scenario == "no_das_pressure_only":
            return not has_fiber and bool(selected.get("pressure_source"))
        return False

    def dataset_ready_for_scenario(self, dataset_id: str, scenario_id: str | None = None) -> bool:
        """Check readiness for one scenario instead of any available view."""

        selected_scenario = str(scenario_id or self.scenario_id)
        if not self.dataset_supports_scenario(dataset_id, selected_scenario):
            return False
        selected = self.dataset(dataset_id)
        cache = self.path(selected.get("cache_source"))
        html_values = selected.get("html_by_scenario", {}) or {}
        html_path = self.path(html_values.get(selected_scenario))
        return bool(cache and cache.exists() and html_path and html_path.exists())

    def datasets_for_scenario(self, scenario_id: str | None = None) -> list[dict[str, Any]]:
        """List only single-stage datasets compatible with a DT scenario."""

        selected_scenario = str(scenario_id or self.scenario_id)
        result = []
        for dataset_id, value in sorted((self.dataset_catalog().get("datasets", {}) or {}).items()):
            if self.dataset_ready_for_scenario(str(dataset_id), selected_scenario):
                result.append(dict(value, dataset_id=str(dataset_id)))
        return result

    def set_dataset(self, dataset_id: str) -> None:
        selected = self.dataset(dataset_id)
        self.dataset_id = str(selected["dataset_id"])
        declared = selected.get("supported_scenarios")
        if isinstance(declared, list):
            supported = [str(value) for value in declared if str(value)]
        else:
            supported = [str(value) for value in (selected.get("html_by_scenario", {}) or {}) if str(value)]
        if supported and self.scenario_id not in supported:
            # Dataset selection is global.  If the newly selected stage does
            # not belong to the previous observation scene, move to that
            # stage's first declared scene instead of silently forcing every
            # stage into the no-DAS branch.
            self.scenario_id = supported[0]

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
        selected = self.dataset()
        dataset_path = resolve(selected.get("cache_source"))
        if selected.get("cache_source"):
            # Never fall back to a different well when an imported stage has
            # not been computed yet. FSL can still use its original table.
            return dataset_path if dataset_path and dataset_path.exists() else None
        value = self.module("dt").get("frame_source")
        path = resolve(value)
        return path if path and path.exists() else PATHS.dt_cache if PATHS.dt_cache.exists() else None

    def scenario(self, scenario_id: str | None = None) -> dict[str, Any]:
        selected = scenario_id or self.scenario_id
        # The dedicated scenario registry is the source of truth for the
        # observation contract.  Keep legacy demo_registry fields as a
        # compatibility fallback so old frozen caches remain readable.
        base = dict(self.module("dt").get("scenarios", {}).get(selected, {}))
        base.update(dict((self._scenario_registry.get("scenarios", {}) or {}).get(selected, {})))
        dataset = self.dataset()
        if dataset.get("adapter") == "raw_frac_construction":
            base.update({
                "display_name": "无 DAS：压力在线校正",
                "observation_mode": "pressure_only",
                "pressure_source": dataset.get("pressure_source"),
                "fiber_source": None,
                "trajectory_source": None,
                "cluster_geometry_source": None,
                "well_id": dataset.get("well_id") or dataset.get("display_name"),
                "stage_id": dataset.get("stage_id") or "unknown",
                "source_start_s": dataset.get("source_start_s", 1),
                "source_end_s": dataset.get("source_end_s"),
                "assumed_cluster_count": dataset.get("assumed_cluster_count", 6),
                "calibration_status": dataset.get("calibration_status", "待校准"),
                "hmi_available": dataset.get("hmi_available", False),
            })
        return base

    def html(self, scenario_id: str | None = None) -> Path | None:
        selected = self.dataset()
        dataset_values = selected.get("html_by_scenario", {}) or {}
        if scenario_id or self.scenario_id in dataset_values:
            value = dataset_values.get(scenario_id or self.scenario_id)
            path = resolve(value)
            if path:
                return path
        module = self.module("dt")
        values = module.get("html_by_scenario", {}) or {}
        value = values.get(scenario_id or self.scenario_id, module.get("html"))
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

    def runtime_selection(self) -> dict[str, Any]:
        """Return the resolved KG-EnKF/agent selection used by the APP."""

        return resolve_runtime_selection().to_dict()
