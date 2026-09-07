"""Adapter for frame-level HMI action rows and acceptance validation."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from .dt_loader import number, read_csv
from ..core.paths import PATHS


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def enrich_control_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    last_by_episode: dict[str, tuple[str, str]] = {}
    output = []
    for original in rows:
        row = dict(original)
        episode = str(row.get("episode", "0"))
        flow = row.get("flow_m3_min", "")
        sand = row.get("sand_ratio_percent", "")
        current_flow = row.get("pre_action_flow_m3_min", "") or last_by_episode.get(episode, (flow, sand))[0]
        current_sand = (
            row.get("observed_current_sand_ratio_percent", "")
            or row.get("pre_action_sand_ratio_percent", "")
            or last_by_episode.get(episode, (flow, sand))[1]
        )
        recommendation = (
            row.get("recommended_sand_ratio_percent", "")
            or row.get("sand_ratio_percent", "")
        )
        row["current_flow_m3_min"] = current_flow
        row["current_sand_ratio_percent"] = current_sand
        row["recommended_sand_ratio_percent"] = recommendation
        row["current_control_source"] = "environment_pre_action_state" if row.get("pre_action_flow_m3_min") else "previous_action_fallback"
        last_by_episode[episode] = (row.get("flow_m3_min", flow), recommendation or sand)
        output.append(row)
    return output


def _model_root(registry_loader) -> Path | None:
    """Locate the frozen comparison run that owns the registered HMI replay.

    The APP is registry-first, but the HMI comparison contains several real
    policy artifacts.  Deriving the sibling model directories from the
    registered replay path keeps the selector tied to the delivered run
    instead of inventing a second list of model files in the UI.
    """

    module = registry_loader.module("hmi")
    candidates = [
        module.get("frame_source"),
        module.get("policy_model"),
        (module.get("summary", {}) or {}).get("frame_source"),
    ]
    for value in candidates:
        path = registry_loader.path(value)
        if not path:
            continue
        # .../<comparison>/<timestamp>/<algorithm>/seed_<n>/<run>/file
        for parent in path.parents:
            if parent.exists() and all((parent / algorithm).exists() for algorithm in ("sac", "td3", "ppo")):
                return parent
    return None


def discover_agent_models(registry_loader) -> list[dict[str, Any]]:
    """Return real agent replay artifacts available to the HMI page.

    A model is selectable for the APP only when its independent evaluation
    CSV exists.  The policy archive is reported separately because this
    release is a frozen-replay application: changing the selector changes
    the replay source, it does not claim to run online inference.
    """

    module = registry_loader.module("hmi")
    runtime = registry_loader.runtime_selection() if hasattr(registry_loader, "runtime_selection") else {}
    default_id = str(module.get("active_policy") or runtime.get("agent_policy") or "sac").strip().lower()
    root = _model_root(registry_loader)
    discovered: list[dict[str, Any]] = []
    if root and root.exists():
        for algorithm in ("sac", "td3", "ppo"):
            algorithm_root = root / algorithm
            if not algorithm_root.exists():
                continue
            runs = sorted(algorithm_root.glob("seed_2026/*"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
            for run in runs:
                evaluation = run / "rl_evaluation.csv"
                if not evaluation.exists():
                    continue
                summary = run / "summary.json"
                policy_candidates = sorted(run.glob("*_fracturing_policy.zip"))
                decisions = run / "human_machine_decisions.json"
                record = {
                    "model_id": algorithm,
                    "display_name": algorithm.upper(),
                    "label": f"{algorithm.upper()} · 真实离线回放",
                    "seed": "2026",
                    "run_dir": str(run),
                    "evaluation_path": str(evaluation),
                    "decisions_path": str(decisions) if decisions.exists() else None,
                    "policy_path": str(policy_candidates[0]) if policy_candidates else None,
                    "summary_path": str(summary) if summary.exists() else None,
                    "ready": True,
                    "status": "可切换回放",
                    "is_default": algorithm == default_id,
                }
                discovered.append(record)
                break
    order = {"sac": 0, "td3": 1, "ppo": 2}
    discovered.sort(key=lambda item: (not item["is_default"], order.get(item["model_id"], 99)))
    return discovered


def dataset_agent_evaluation_path(registry_loader, dataset_id: str | None, model_id: str | None) -> Path | None:
    """Return a dataset-specific prediction cache for one model, if present."""

    if not dataset_id or not model_id:
        return None
    dataset = registry_loader.dataset(dataset_id) if hasattr(registry_loader, "dataset") else {}
    cache_path = registry_loader.path(dataset.get("cache_source")) if hasattr(registry_loader, "path") else None
    if not cache_path or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if not isinstance(meta, dict):
        return None
    wanted = str(model_id).strip().lower()
    by_model = meta.get("hmi_recommendations_by_model", {})
    value = by_model.get(wanted) if isinstance(by_model, dict) else None
    if not value:
        value = by_model.get(wanted.upper()) if isinstance(by_model, dict) else None
    active_model = str(meta.get("hmi_policy", "")).strip().lower()
    if not value and active_model == wanted:
        value = meta.get("hmi_recommendations_source")
    path = registry_loader.path(value) if hasattr(registry_loader, "path") else None
    return path if path and path.exists() else None


class HMILoader:
    def __init__(self, registry_loader, model_id: str | None = None) -> None:
        self.registry = registry_loader
        self.module = registry_loader.module("hmi")
        self.summary = registry_loader.summary("hmi")
        self.models = discover_agent_models(registry_loader)
        requested = str(model_id or "").strip().lower()
        self.model_info = next(
            (item for item in self.models if item.get("model_id") == requested),
            None,
        )
        if self.model_info is None and self.models:
            self.model_info = next((item for item in self.models if item.get("is_default")), self.models[0])
        runtime = registry_loader.runtime_selection() if hasattr(registry_loader, "runtime_selection") else {}
        # A no-DAS segment has no valid reason to reuse the global HMI replay
        # from another well.  Prefer a recommendation cache generated from
        # this selected segment; fall back to the registered global replay
        # only for the DAS demo dataset that owns it.
        selected_dataset = registry_loader.dataset() if hasattr(registry_loader, "dataset") else {}
        dataset_eval = registry_loader.path(selected_dataset.get("hmi_recommendations_source"))
        dataset_decisions = registry_loader.path(selected_dataset.get("hmi_decisions_source"))
        configured_eval = registry_loader.path(runtime.get("replay_evaluation_path"))
        configured_decisions = registry_loader.path(runtime.get("replay_decisions_path"))
        dataset_model_eval = dataset_agent_evaluation_path(
            registry_loader,
            selected_dataset.get("dataset_id"),
            self.model_info.get("model_id") if self.model_info else None,
        )
        if dataset_model_eval:
            configured_eval = dataset_model_eval
        elif selected_dataset.get("adapter") == "raw_frac_construction":
            # Never fall back to the global JY84 replay for another raw well.
            configured_eval = None
        elif self.model_info and self.model_info.get("evaluation_path"):
            selected_eval = registry_loader.path(self.model_info.get("evaluation_path"))
            if selected_eval and selected_eval.exists():
                configured_eval = selected_eval
        elif dataset_eval and dataset_eval.exists():
            configured_eval = dataset_eval
        if selected_dataset.get("adapter") == "raw_frac_construction":
            # The raw-segment prediction cache contains actions, not human
            # confirmation records; do not reuse another well's audit log.
            configured_decisions = None
        elif self.model_info and self.model_info.get("decisions_path"):
            selected_decisions = registry_loader.path(self.model_info.get("decisions_path"))
            if selected_decisions and selected_decisions.exists():
                configured_decisions = selected_decisions
        elif dataset_decisions and dataset_decisions.exists():
            configured_decisions = dataset_decisions
        allow_registered_fallback = selected_dataset.get("adapter") != "raw_frac_construction"
        self.eval_path = configured_eval if configured_eval and configured_eval.exists() else (
            registry_loader.table("hmi", "rl_evaluation.csv") or registry_loader.output("hmi", "rl_evaluation")
            if allow_registered_fallback else None
        )
        self.decision_path = configured_decisions if configured_decisions and configured_decisions.exists() else (
            registry_loader.table("hmi", "human_machine_decisions.json") or registry_loader.output("hmi", "human_machine_decisions")
            if allow_registered_fallback else None
        )
        self.rows = enrich_control_rows(read_csv(self.eval_path))
        self.working_types = self._load_working_types()
        self.decisions = self._load_decisions()

    def _load_working_types(self) -> dict[str, str]:
        path = PATHS.data / "raw_frac" / "segment_working_type_labels.csv"
        if not path.exists():
            return {}
        result: dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    labels = [item.strip() for item in str(row.get("labels", "")).split("|") if item.strip() and item.strip() != "??"]
                    label = labels[0] if labels else "unknown"
                    counts = {}
                    for item in str(row.get("label_counts", "")).split("|"):
                        if ":" in item:
                            key, value = item.rsplit(":", 1)
                            try:
                                counts[key.strip()] = int(value)
                            except ValueError:
                                pass
                    if counts:
                        label = max(((key, value) for key, value in counts.items() if key != "??"), key=lambda item: item[1], default=(label, 0))[0]
                    source = Path(str(row.get("source_file", ""))).stem.lower()
                    segment = str(row.get("segment_id", "")).strip().lower()
                    if source:
                        result[source] = label
                    if segment:
                        result.setdefault(segment, label)
        except (OSError, csv.Error):
            return {}
        return result

    def _load_decisions(self) -> list[dict[str, Any]]:
        if not self.decision_path or not self.decision_path.exists():
            return []
        try:
            value = json.loads(self.decision_path.read_text(encoding="utf-8-sig"))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            return [value] if isinstance(value, dict) else []
        except (OSError, json.JSONDecodeError):
            return []

    def at(self, index: int, normalized_count: int | None = None) -> dict[str, Any]:
        if not self.rows:
            return {}
        count = normalized_count or len(self.rows)
        source_index = min(round(index / max(count - 1, 1) * max(len(self.rows) - 1, 0)), len(self.rows) - 1)
        item = dict(self.rows[source_index])
        segment = str(item.get("segment_id", "")).strip().lower()
        working_type = self.working_types.get(segment, "unknown")
        item["working_type"] = working_type
        item["abnormal_type"] = working_type if working_type not in {"unknown", "正常", "主缝延伸"} else "none"
        if working_type == "砂堵":
            item["rule_hit"] = "砂堵相关规则（注册标签）"
        return item

    def decision_at(self, index: int, normalized_count: int) -> dict[str, Any]:
        if not self.decisions:
            return {}
        source_index = round(index / max(normalized_count - 1, 1) * max(len(self.decisions) - 1, 0))
        return dict(self.decisions[min(max(source_index, 0), len(self.decisions) - 1)])

    def validation(self) -> dict[str, Any]:
        return self.summary.get("validation_180s", {}) if isinstance(self.summary, dict) else {}
