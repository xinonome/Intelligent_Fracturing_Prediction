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
        current_sand = row.get("pre_action_sand_ratio_percent", "") or last_by_episode.get(episode, (flow, sand))[1]
        row["current_flow_m3_min"] = current_flow
        row["current_sand_ratio_percent"] = current_sand
        row["current_control_source"] = "environment_pre_action_state" if row.get("pre_action_flow_m3_min") else "previous_action_fallback"
        last_by_episode[episode] = (flow, sand)
        output.append(row)
    return output


class HMILoader:
    def __init__(self, registry_loader) -> None:
        self.registry = registry_loader
        self.module = registry_loader.module("hmi")
        self.summary = registry_loader.summary("hmi")
        self.eval_path = registry_loader.table("hmi", "rl_evaluation.csv") or registry_loader.output("hmi", "rl_evaluation")
        self.decision_path = registry_loader.table("hmi", "human_machine_decisions.json") or registry_loader.output("hmi", "human_machine_decisions")
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
