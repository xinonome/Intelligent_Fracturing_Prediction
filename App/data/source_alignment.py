"""Source alignment utilities with explicit missing-data provenance."""

from __future__ import annotations

from typing import Any

from ..core.timeline import AlignedValue, align_categorical, align_numeric, build_time_axis


def align_records(
    records: list[dict[str, Any]],
    target_times: list[float],
    *,
    time_key: str = "time_s",
    numeric_fields: list[str] | None = None,
    categorical_fields: list[str] | None = None,
    source_name: str = "source",
) -> list[dict[str, Any]]:
    numeric_fields = numeric_fields or []
    categorical_fields = categorical_fields or []
    source_times = [float(row[time_key]) for row in records if row.get(time_key) not in (None, "")]
    output = [{"time_s": time, "quality": {}} for time in target_times]
    for field in numeric_fields:
        values = [row.get(field) for row in records if row.get(time_key) not in (None, "")]
        aligned = align_numeric(source_times, values, target_times, source_name)
        for row, item in zip(output, aligned):
            row[field] = item.value
            row["quality"][field] = item.as_dict()
    for field in categorical_fields:
        values = [row.get(field) for row in records if row.get(time_key) not in (None, "")]
        aligned = align_categorical(source_times, values, target_times, source_name)
        for row, item in zip(output, aligned):
            row[field] = item.value
            row["quality"][field] = item.as_dict()
    return output


def build_common_axis(*records: list[dict[str, Any]], time_key: str = "time_s", step_s: float = 1.0) -> list[float]:
    return build_time_axis(*([row.get(time_key) for row in source] for source in records), step_s=step_s)
