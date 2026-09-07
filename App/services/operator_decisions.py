"""Append-only local audit log for human review of advisory actions."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from ..core.paths import PATHS


DEFAULT_LOG = PATHS.app_outputs / "operator_decisions.jsonl"
_LOG_LOCK = RLock()


def append_decision(entry: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    record = dict(entry)
    record["record_id"] = uuid4().hex
    record.setdefault("record_type", "review")
    record.setdefault("recorded_at", datetime.now().isoformat(timespec="seconds"))
    # This service is an audit trail, never an actuator or dispatch queue.
    record["execution_state"] = "人工撤销记录，未下发现场" if record["record_type"] == "rollback" else "人工已记录，未下发现场"
    target = Path(path or DEFAULT_LOG)
    with _LOG_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record


def load_decisions(
    path: Path | None = None, limit: int | None = 100, *,
    dataset_id: str | None = None, model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read history; filter before limiting so other wells cannot hide records.

    Legacy rows remain readable, but missing identity is never inferred from
    the current selection. Such rows cannot be rolled back as a named review.
    """
    target = Path(path or DEFAULT_LOG)
    if not target.exists():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(value, dict)
                and (dataset_id is None or value.get("dataset_id") == dataset_id)
                and (model_id is None or str(value.get("model_id", "")).lower() == model_id.lower())):
            rows.append(value)
    return rows if limit is None else rows[-max(int(limit), 1):]


def rollback_decision(
    record_id: str, *, dataset_id: str, model_id: str, path: Path | None = None,
) -> dict[str, Any]:
    """Append a reversal of one explicit confirmation; leave the original intact."""
    if not record_id or not dataset_id or not model_id:
        raise ValueError("撤销必须指定记录编号、井段和模型")
    with _LOG_LOCK:
        rows = load_decisions(path, limit=None)
        target = next((row for row in rows if row.get("record_id") == record_id), None)
        if target is None:
            raise ValueError("所选记录不存在或缺少可追溯编号")
        if target.get("dataset_id") != dataset_id or str(target.get("model_id", "")).lower() != model_id.lower():
            raise ValueError("所选记录不属于当前井段和模型")
        if target.get("record_type", "review") != "review" or target.get("decision") not in {"确认采用", "修改后采用"}:
            raise ValueError("只能撤销明确的采用确认记录")
        if any(row.get("rollback_of") == record_id for row in rows):
            raise ValueError("该确认记录已经撤销")
        return append_decision({
            "record_type": "rollback", "rollback_of": record_id,
            "dataset_id": dataset_id, "model_id": model_id.lower(),
            "time_s": target.get("time_s"), "frame_id": target.get("frame_id"),
            "decision": "撤销确认记录", "advisory_source": target.get("advisory_source"),
        }, path)


__all__ = ["DEFAULT_LOG", "append_decision", "load_decisions", "rollback_decision"]
