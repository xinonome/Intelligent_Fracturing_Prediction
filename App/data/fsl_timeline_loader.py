"""Load independent construction timelines for the first APP module.

The first-part page is a stage-level view. In particular, a workbook that
contains many FDBH groups must not be treated as one physical stage. This
adapter therefore reads the independent FDBH workbooks and combines their
records only for building the stage selector.
"""

from __future__ import annotations

from collections import Counter, deque
import csv
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import re
from typing import Any
import zipfile
import xml.etree.ElementTree as ET

from ..core.paths import PATHS
from ..services.fsl_risk_service import RISK_RUNTIME_VERSION, analyze_stage, selected_model_id


_BLANK = {"", "nan", "none", "null", "nat"}
_ABNORMAL = {"砂堵", "缝口暂堵", "缝内暂堵", "滤失过大", "延伸受阻"}
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"^([A-Za-z]+)")


def _text(value: Any) -> str:
    if value is None:
        return ""
    result = str(value).strip()
    return "" if result.lower() in _BLANK else result


def _number(value: Any) -> float | None:
    try:
        if value is None or _text(value) == "":
            return None
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _stage_key(value: str) -> tuple[int, str]:
    # Stage labels are normally generated from the source workbook (for
    # example FDBH1.1).  Keep the numeric part in the natural order while
    # retaining the full label as a deterministic tie breaker.
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if match:
        try:
            return (0, f"{float(match.group(1)):012.3f}_{str(value).lower()}")
        except ValueError:
            pass
    try:
        return (0, f"{int(float(value)):08d}")
    except (TypeError, ValueError):
        return (1, str(value))


def _fmt_time(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return value.strftime("%m-%d %H:%M:%S")
    except AttributeError:
        return _text(value) or "--"


def _parse_datetime(value: Any) -> datetime | None:
    """Parse the timestamp formats used by construction workbooks."""

    if isinstance(value, datetime):
        return value
    text = _text(value)
    if not text:
        return None
    for candidate in (text, text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate.replace("T", " "))
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    number = _number(value)
    if number is not None and 1.0 < number < 300000.0:
        return datetime(1899, 12, 30) + timedelta(days=number)
    return None


def _column_number(cell_ref: str) -> int:
    match = _CELL_REF.match(cell_ref or "")
    if not match:
        return 0
    value = 0
    for character in match.group(1).upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _xlsx_cell_value(cell, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", _XLSX_NS)
        return "" if inline is None else "".join(inline.itertext())
    value_node = cell.find("m:v", _XLSX_NS)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, TypeError, ValueError):
            return ""
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def _read_xlsx_rows(path: Path) -> list[list[Any]]:
    """Read the first worksheet with the standard library only."""

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(item.itertext()) for item in root.findall("m:si", _XLSX_NS)]
        worksheet = "xl/worksheets/sheet1.xml"
        if worksheet not in archive.namelist():
            worksheet = next(
                (
                    name
                    for name in archive.namelist()
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                ),
                "",
            )
        if not worksheet:
            return []
        root = ET.fromstring(archive.read(worksheet))
        rows: list[list[Any]] = []
        for row_node in root.findall(".//m:sheetData/m:row", _XLSX_NS):
            cells: dict[int, Any] = {}
            for cell in row_node.findall("m:c", _XLSX_NS):
                cells[_column_number(cell.attrib.get("r", ""))] = _xlsx_cell_value(cell, shared_strings)
            if not cells:
                rows.append([])
                continue
            row = [""] * (max(cells) + 1)
            for index, value in cells.items():
                row[index] = value
            rows.append(row)
        return rows


def _read_csv_rows(path: Path) -> list[list[Any]]:
    """Read a UTF-8/UTF-8-BOM construction table without pandas."""

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [list(row) for row in csv.reader(handle)]
        except (OSError, UnicodeError):
            continue
    return []


def _read_xlsx_records(
    path: Path,
    *,
    headers: list[str] | None = None,
    has_header: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _read_xlsx_rows(path)
    if not rows:
        return [], list(headers or [])
    if headers is None:
        headers = [_text(value) or f"column_{index + 1}" for index, value in enumerate(rows[0])]
    headers = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(headers)]
    data_rows = rows[1:] if has_header else rows
    records = _records_from_rows(data_rows, headers)
    return records, headers


def _records_from_rows(rows: list[list[Any]], headers: list[str]) -> list[dict[str, Any]]:
    # Some exported construction workbooks contain repeated headers (notably
    # ``PL``: the first column is the measured flow and a later annotation
    # column is often empty).  A plain dict comprehension silently lets the
    # later duplicate overwrite the measured value.  Keep the first header
    # under its original name and suffix later occurrences instead.
    unique_headers: list[str] = []
    seen: dict[str, int] = {}
    for index, header in enumerate(headers):
        base = str(header).strip() or f"column_{index + 1}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique_headers.append(base if count == 1 else f"{base}_{count}")
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(unique_headers)}
        for row in rows
    ]


def _safe_max(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _safe_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _rolling_next_point_forecast(
    times: list[float],
    values: list[float | None],
    *,
    lookback: int = 6,
) -> list[float | None]:
    """Generate a transparent one-step-ahead point forecast.

    The first-part release currently contains the measured construction
    timeline and the frozen risk/classification outputs, but does not contain
    a serialized pressure/flow regression model with point-level predictions.
    This small predictor is therefore deliberately explicit: each point is
    predicted from the preceding ``lookback`` valid observations only.  It is
    used as a displayable baseline until a registered point-prediction model
    is supplied, and must not be presented as a trained offline metric.
    """

    result: list[float | None] = [None] * len(values)
    if lookback < 2:
        return result
    recent: deque[tuple[float, float]] = deque(maxlen=lookback)
    for target_index, actual in enumerate(values):
        target_time = times[target_index] if target_index < len(times) else float(target_index)
        if not isinstance(target_time, (int, float)) or not math.isfinite(float(target_time)):
            target_time = float(target_index)
        # Capture history before adding the target observation. Missing runs
        # remain O(n * lookback), rather than scanning the whole past per point.
        history = [item for item in recent if item[0] < target_time]
        value = _number(actual)
        if value is not None:
            recent.append((float(target_time), value))
        if len(history) < 2:
            continue
        mean_time = sum(item[0] for item in history) / len(history)
        mean_value = sum(item[1] for item in history) / len(history)
        denominator = sum((item[0] - mean_time) ** 2 for item in history)
        if denominator <= 1e-12:
            prediction = mean_value
        else:
            slope = sum((item[0] - mean_time) * (item[1] - mean_value) for item in history) / denominator
            prediction = mean_value + slope * (float(target_time) - mean_time)
        if math.isfinite(float(prediction)):
            result[target_index] = float(prediction)
    return result


def _point_prediction_metrics(
    actual: list[float | None],
    predicted: list[float | None],
    *,
    tolerance: float,
) -> dict[str, float | int | None]:
    """Return display metrics for paired point predictions."""

    errors = [
        float(predicted[index]) - float(actual[index])
        for index in range(min(len(actual), len(predicted)))
        if actual[index] is not None and predicted[index] is not None
    ]
    if not errors:
        return {
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "within_tolerance_rate": None,
        }
    absolute = [abs(error) for error in errors]
    return {
        "sample_count": len(errors),
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "within_tolerance_rate": sum(error <= tolerance for error in absolute) / len(errors),
    }


class FSLTimelineLoader:
    """Registry-backed loader for independent stage construction timelines."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self.module = registry.module("fsl")
        self.source_paths = self._resolve_sources()
        self.source_path = self.source_paths[0] if self.source_paths else None
        self.cache_path = PATHS.app_outputs / "fsl_timeline_cache.json"
        self.status = "not_available"
        self.status_reason = "未找到第一部分时序数据"
        self._stages: dict[str, dict[str, Any]] = {}
        if not self._load_cache():
            self._load()

    @staticmethod
    def _is_composite_source(path: Path) -> bool:
        lowered = path.name.lower()
        return any(marker in lowered for marker in ("便签数据", "综合", "aggregate", "combined"))

    @classmethod
    def _is_ignored_source(cls, path: Path) -> bool:
        lowered = path.name.lower()
        if "withfiltered" in lowered or cls._is_composite_source(path):
            return True
        # These CSV files are derived label/statistics tables, not raw
        #施工时序.  Treating them as source workbooks creates fake one-row
        # stages such as FDBH0 with a duration of 0 s.
        if path.suffix.lower() == ".csv" and any(
            marker in lowered for marker in ("label", "distribution", "summary")
        ):
            return True
        return False

    def _resolve_sources(self) -> list[Path]:
        """Return all independent workbook sources; never the aggregate export."""

        configured = self.registry.path(self.module.get("timeline_source"))
        configured_directory = self.registry.path(self.module.get("timeline_source_directory"))
        directory = configured_directory or (PATHS.data / "raw_frac")
        candidates: list[Path] = []
        if directory and directory.exists() and directory.is_dir():
            candidates.extend(
                sorted(path for path in directory.iterdir() if path.suffix.lower() in {".xlsx", ".csv"})
            )
        if configured and configured.exists() and configured.is_file():
            candidates.insert(0, configured)

        result: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve()).lower()
            if key in seen or self._is_ignored_source(path):
                continue
            seen.add(key)
            result.append(path)
        return result

    def _resolve_source(self) -> Path | None:
        """Compatibility helper for integrations that expect one source."""

        return self.source_paths[0] if self.source_paths else None

    def _reference_columns(self) -> list[str] | None:
        reference = self.registry.path(self.module.get("timeline_header_source"))
        if not reference or not reference.exists() or self._is_ignored_source(reference):
            return None
        try:
            _, columns = _read_xlsx_records(reference)
            return columns or None
        except Exception:
            return None

    def _read_source_records(
        self,
        source_path: Path,
        reference_columns: list[str] | None,
    ) -> list[dict[str, Any]]:
        # Read each workbook once.  Headerless exports previously went
        # through the XML parser twice, which made loading all independent
        # stages unnecessarily slow.
        rows = _read_csv_rows(source_path) if source_path.suffix.lower() == ".csv" else _read_xlsx_rows(source_path)
        if not rows:
            return []
        detected_columns = [_text(value) or f"column_{index + 1}" for index, value in enumerate(rows[0])]
        if "SGSJ" not in detected_columns and reference_columns:
            columns = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(reference_columns)]
            records = _records_from_rows(rows, columns)
        else:
            columns = detected_columns
            records = _records_from_rows(rows[1:], columns)
        if "FDBH" not in columns or "SGSJ" not in columns:
            return []
        for record in records:
            record["_source_name"] = source_path.name
        return records

    def _source_signature(self) -> list[dict[str, Any]]:
        result = []
        for path in self.source_paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            result.append({"name": path.name, "path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return result

    def _load_cache(self) -> bool:
        """Load the independent-stage cache so APP startup is not blocked."""

        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 10
            or not isinstance(payload.get("stages"), dict)
        ):
            return False
        if payload.get("risk_model_id") != selected_model_id():
            return False
        if payload.get("risk_runtime_version") != RISK_RUNTIME_VERSION:
            return False
        cached_sources = payload.get("source_files", [])
        if any(self._is_composite_source(Path(str(name))) for name in cached_sources):
            return False
        # If source workbooks are present, invalidate the cache when any
        # source changes. If a release contains only the generated cache, it
        # remains usable without shipping the raw construction tables.
        if self.source_paths and payload.get("source_signature") != self._source_signature():
            return False
        self._stages = {
            str(stage_id): dict(stage)
            for stage_id, stage in payload.get("stages", {}).items()
            if isinstance(stage, dict)
        }
        # Migrate older timeline caches in memory.  The new table shows all
        # source conditions, while schema-9 caches only stored label_counts.
        # Keeping this migration avoids forcing a full workbook read on every
        # existing installation.
        for stage in self._stages.values():
            if "source_condition_labels" not in stage:
                counts = stage.get("label_counts") or {}
                if isinstance(counts, dict):
                    stage["source_condition_labels"] = [
                        str(label) for label in counts if str(label).strip() and str(label) != "未标注"
                    ]
                else:
                    stage["source_condition_labels"] = []
        if not self._stages:
            return False
        self.status = "ready"
        self.status_reason = f"已加载 {len(self._stages)} 个独立 FDBH 井段；按独立施工表展示"
        return True

    def _write_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 10,
                "risk_model_id": selected_model_id(),
                "risk_runtime_version": RISK_RUNTIME_VERSION,
                "source_files": [path.name for path in self.source_paths],
                "source_signature": self._source_signature(),
                "stages": self._stages,
            }
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            # The APP can still operate from the in-memory data in a
            # read-only release directory.
            return

    def _load(self) -> None:
        if not self.source_paths:
            return
        try:
            reference_columns = self._reference_columns()
            records: list[dict[str, Any]] = []
            for source_path in self.source_paths:
                records.extend(self._read_source_records(source_path, reference_columns))
            if not records:
                self.status_reason = "独立井段时序表缺少 FDBH 或 SGSJ 字段"
                return
            self._build_stages(records)
        except Exception as exc:  # pragma: no cover - depends on workbook structure
            self.status_reason = f"第一部分时序数据读取失败：{exc}"
            return
        self.status = "ready" if self._stages else "not_available"
        self.status_reason = (
            f"已加载 {len(self._stages)} 个独立 FDBH 井段；按独立施工表展示"
            if self._stages
            else "独立井段时序表没有可用井段"
        )
        if self._stages:
            self._write_cache()

    def _build_stages(self, records: list[dict[str, Any]]) -> None:
        # FDBH is only unique within a source well/export.  In the released
        # data, for example, FDBH1.xlsx and FDBH1.1.xlsx both contain FDBH=1
        # but belong to different wells and different calendar dates.  The
        # old FDBH-only key silently merged them and created a multi-year
        # fake duration on the first-part timeline.
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            stage_id = _text(record.get("FDBH"))
            timestamp = _parse_datetime(record.get("SGSJ"))
            if not stage_id or timestamp is None:
                continue
            item = dict(record)
            item["_stage"] = stage_id
            item["_time"] = timestamp
            source_name = _text(record.get("_source_name")) or "unknown_source"
            grouped.setdefault((source_name, stage_id), []).append(item)

        base_counts = Counter(source_stage_id for _, source_stage_id in grouped)
        for (source_name, source_stage_id), group in grouped.items():
            group.sort(key=lambda item: item["_time"])
            stage_id = self._unique_stage_label(
                source_name,
                source_stage_id,
                duplicate=base_counts[source_stage_id] > 1,
            )
            self._stages[stage_id] = self._build_stage(
                stage_id,
                group,
                source_name=source_name,
                source_stage_id=source_stage_id,
            )
        self._stages = dict(sorted(self._stages.items(), key=lambda item: _stage_key(item[0])))

    @staticmethod
    def _unique_stage_label(source_name: str, source_stage_id: str, *, duplicate: bool) -> str:
        """Build a stable, human-readable key for one source workbook."""

        stem = Path(source_name).stem.strip() or "FDBH"
        # The workbook name is not always the actual FDBH value (for example
        # FDBH16.xlsx contains a separate FDBH12 group).  Use the value from
        # the record as the primary display label and append the source only
        # when the same FDBH is present in multiple source exports.
        base = source_stage_id
        if not base.lower().startswith("fdbh"):
            base = f"FDBH{base}"
        if not duplicate:
            return base
        if stem.lower().startswith("fdbh"):
            stem = "FDBH" + stem[4:]
        return f"{base} · {stem}"

    def _build_stage(
        self,
        stage_id: str,
        group: list[dict[str, Any]],
        *,
        source_name: str,
        source_stage_id: str,
    ) -> dict[str, Any]:
        start = group[0]["_time"]
        end = group[-1]["_time"]
        relative_s = [float((item["_time"] - start).total_seconds()) for item in group]

        def values(column: str) -> list[float | None]:
            return [_number(item.get(column)) for item in group]

        pressure = values("SGBY")
        flow = values("PL")
        sand = values("SB")
        predicted_pressure = _rolling_next_point_forecast(relative_s, pressure)
        predicted_flow = _rolling_next_point_forecast(relative_s, flow)
        predicted_sand = [
            max(0.0, value) if value is not None else None
            for value in _rolling_next_point_forecast(relative_s, sand)
        ]
        labels = [_text(item.get("WORKING_TYPE")) or "未标注" for item in group]
        source_probability = values("PROBABILITY")
        work_chance = values("WORKINGCHANCE")
        suggestions = [_text(item.get(column)) for item in group for column in ("SUGGESTION", "SUGGEST")]
        suggestions = [value for value in suggestions if value]

        intervals = self._intervals(group, labels, start)
        risk_analysis = analyze_stage(relative_s, group)
        annotated_labels = [label for label in labels if label != "未标注"]
        label_counts = Counter(annotated_labels)
        main_label = label_counts.most_common(1)[0][0] if label_counts else "未标注"
        source_condition_labels = list(dict.fromkeys(annotated_labels))
        risk_values = [value for value in risk_analysis.probability if value is not None]
        source_risk_values = [value for value in source_probability + work_chance if value is not None]
        abnormal = any(label in _ABNORMAL for label in annotated_labels)
        suggestion = Counter(suggestions).most_common(1)[0][0] if suggestions else None
        if not suggestion and "红色" in risk_analysis.levels:
            suggestion = "暂停加砂并人工复核压力、排量和砂比响应"
        elif not suggestion and "黄色" in risk_analysis.levels:
            suggestion = "保持排量稳定，复核压力持续上升与砂比波动"
        display_indices = self._sample_indices(len(group), 1000)
        source_names = sorted({str(item.get("_source_name", "")) for item in group if item.get("_source_name")})
        pressure_metrics = _point_prediction_metrics(pressure, predicted_pressure, tolerance=2.0)
        flow_metrics = _point_prediction_metrics(flow, predicted_flow, tolerance=0.5)
        sand_metrics = _point_prediction_metrics(sand, predicted_sand, tolerance=1.0)
        point_prediction_count = max(
            int(pressure_metrics["sample_count"]),
            int(flow_metrics["sample_count"]),
            int(sand_metrics["sample_count"]),
        )

        return {
            "stage_id": stage_id,
            "source_stage_id": source_stage_id,
            "source_file": source_name,
            "well_id": _text(group[0].get("JTBH")) or None,
            "start_time": _fmt_time(start),
            "end_time": _fmt_time(end),
            "duration_s": float((end - start).total_seconds()),
            "sample_count": int(len(group)),
            "time_s": [relative_s[index] for index in display_indices],
            "pressure_mpa": [pressure[index] for index in display_indices],
            "flow_m3_min": [flow[index] for index in display_indices],
            "sand_ratio_pct": [sand[index] for index in display_indices],
            "predicted_pressure_mpa": [predicted_pressure[index] for index in display_indices],
            "predicted_flow_m3_min": [predicted_flow[index] for index in display_indices],
            "predicted_sand_ratio_pct": [predicted_sand[index] for index in display_indices],
            "risk_probability": [risk_analysis.probability[index] for index in display_indices],
            "risk_level": [risk_analysis.levels[index] for index in display_indices],
            "rule_risk_level": [risk_analysis.rule_levels[index] for index in display_indices],
            "source_risk_probability": [source_probability[index] for index in display_indices],
            "working_chance": [work_chance[index] for index in display_indices],
            "intervals": intervals,
            "actual_condition_intervals": intervals,
            "rule_condition_intervals": risk_analysis.rule_intervals,
            "actual_condition_source": "源表 WORKING_TYPE 标注（非在线规则检测）",
            "predicted_condition_intervals": risk_analysis.predicted_intervals,
            "condition_prediction_status": risk_analysis.model_status,
            "condition_prediction_reason": risk_analysis.model_reason,
            "risk_model_id": risk_analysis.model_id,
            "risk_model_source": risk_analysis.model_source,
            "main_label": main_label,
            "source_condition_labels": source_condition_labels,
            "label_counts": dict(label_counts),
            "risk_max_pct": max(risk_values) * 100.0 if risk_values else None,
            "risk_status": (
                "红色" if "红色" in risk_analysis.levels else
                "黄色" if "黄色" in risk_analysis.levels else
                "绿色" if risk_values else
                ("标签异常" if abnormal else "源表标签" if annotated_labels else "未标注")
            ),
            "suggestion": suggestion,
            "pressure_max": _safe_max(pressure),
            "flow_max": _safe_max(flow),
            "sand_max": _safe_max(sand),
            "pressure_mean": _safe_mean(pressure),
            "flow_mean": _safe_mean(flow),
            "sand_mean": _safe_mean(sand),
            "prediction_record_count": len(risk_values),
            "source_prediction_record_count": len(source_risk_values),
            "data_source": ", ".join(source_names) if source_names else "--",
            "prediction_semantics": f"{risk_analysis.model_source}逐点风险结果；{risk_analysis.model_reason}",
            "point_prediction_source": "下一采样点逐点预测",
            "point_prediction_method": "rolling_linear_6",
            "point_prediction_semantics": "仅用此前最多6个有效历史点预测下一采样点；砂比下限截为0；指标为本井段回放误差，非独立测试性能",
            "point_prediction_count": point_prediction_count,
            "point_prediction_metrics": {
                "pressure": pressure_metrics,
                "flow": flow_metrics,
                "sand": sand_metrics,
            },
        }

    @staticmethod
    def _sample_indices(length: int, limit: int) -> list[int]:
        if length <= limit:
            return list(range(length))
        step = max(1, math.ceil((length - 1) / max(limit - 1, 1)))
        indices = list(range(0, length, step))
        if indices[-1] != length - 1:
            indices.append(length - 1)
        return indices

    @staticmethod
    def _intervals(group: list[dict[str, Any]], labels: list[str], stage_start) -> list[dict[str, Any]]:
        labeled = [index for index, label in enumerate(labels) if label != "未标注"]
        if not labeled:
            return []
        draw_start = [_parse_datetime(item.get("DRAW_TIME")) for item in group]
        draw_end = [_parse_datetime(item.get("DRAW_TIME_END")) for item in group]
        result: list[dict[str, Any]] = []
        if any(draw_start[index] is not None for index in labeled):
            seen: set[tuple[str, datetime, datetime]] = set()
            for index in labeled:
                begin = draw_start[index]
                if begin is None:
                    continue
                finish = draw_end[index] or group[index]["_time"]
                key = (labels[index], begin, finish)
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "label": labels[index],
                    "start_s": max(0.0, float((begin - stage_start).total_seconds())),
                    "end_s": max(0.0, float((finish - stage_start).total_seconds())),
                    "kind": "actual",
                    "trigger_reason": "时序表 WORKING_TYPE 与 DRAW_TIME 事件边界",
                })
            return sorted(result, key=lambda item: (item["start_s"], item["label"]))

        begin = labeled[0]
        previous = begin
        for index in labeled[1:] + [None]:
            if index is not None and labels[index] == labels[begin] and index == previous + 1:
                previous = index
                continue
            result.append({
                "label": labels[begin],
                "start_s": max(0.0, float((group[begin]["_time"] - stage_start).total_seconds())),
                "end_s": max(0.0, float((group[previous]["_time"] - stage_start).total_seconds())),
                "kind": "actual",
                "trigger_reason": "时序表连续 WORKING_TYPE 标签",
            })
            if index is not None:
                begin = previous = index
        return result

    def stage_ids(self) -> list[str]:
        return list(self._stages)

    def stage(self, stage_id: str) -> dict[str, Any]:
        return dict(self._stages.get(str(stage_id), {}))

    def pressure_replay_stage(self, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a pressure-only view when the selected stage has no FSL labels.

        The global selector also contains DT datasets such as JY84-Z1 Stage 08.
        Those datasets are not part of the independent FDBH label table, but
        their registered DT cache still contains measured pressure, flow and
        sand-ratio series.  Reuse that cache for the first-page replay rather
        than replacing a valid curve with a blank state.  No FSL event labels
        or point predictions are invented here.
        """

        dataset = dict(dataset or {})
        cache_path = self.registry.frame_source()
        if cache_path is None or not cache_path.exists():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        scenarios = payload.get("scenarios", {})
        scenario_id = str(getattr(self.registry, "scenario_id", "") or "")
        selected = scenarios.get(scenario_id, payload) if isinstance(scenarios, dict) else payload
        if not isinstance(selected, dict):
            return {}
        arrays = selected.get("arrays", {}) or {}
        raw_times = selected.get("timeline_s", []) or []
        if not isinstance(raw_times, list) or len(raw_times) < 2 or not isinstance(arrays, dict):
            return {}

        def numeric_values(name: str) -> list[float | None]:
            values = arrays.get(name, []) or []
            if not isinstance(values, list):
                return []
            result: list[float | None] = []
            for value in values:
                result.append(_number(value))
            return result

        times = [_number(value) for value in raw_times]
        if any(value is None for value in times):
            return {}
        normalized_times = [float(value) - float(times[0]) for value in times if value is not None]
        pressure = numeric_values("surface_pressure_mpa")
        flow = numeric_values("flow_rate_m3_min")
        sand = numeric_values("sand_ratio_percent")
        if not any(value is not None for value in pressure + flow + sand):
            return {}

        limit = 1000
        if len(normalized_times) > limit:
            step = max(1, math.ceil((len(normalized_times) - 1) / max(limit - 1, 1)))
            indices = list(range(0, len(normalized_times), step))
            if indices[-1] != len(normalized_times) - 1:
                indices.append(len(normalized_times) - 1)
        else:
            indices = list(range(len(normalized_times)))

        def sampled(values: list[float | None]) -> list[float | None]:
            return [values[index] if index < len(values) else None for index in indices]

        display_name = str(dataset.get("display_name") or dataset.get("stage_id") or "当前井段")
        source = self.registry.path(dataset.get("pressure_source"))
        sample_count = len(normalized_times)
        return {
            "stage_id": display_name,
            "well_id": dataset.get("well_id"),
            "source_file": source.name if source else str(dataset.get("pressure_source") or "DT缓存"),
            "start_time": "--",
            "end_time": "--",
            "duration_s": float(normalized_times[-1]) if normalized_times else 0.0,
            "sample_count": sample_count,
            "time_s": [normalized_times[index] for index in indices],
            "pressure_mpa": sampled(pressure),
            "flow_m3_min": sampled(flow),
            "sand_ratio_pct": sampled(sand),
            "predicted_pressure_mpa": [],
            "predicted_flow_m3_min": [],
            "predicted_sand_ratio_pct": [],
            "risk_probability": [],
            "working_chance": [],
            "intervals": [],
            "actual_condition_intervals": [],
            "rule_condition_intervals": [],
            "predicted_condition_intervals": [],
            "condition_prediction_status": "no_labels",
            "condition_prediction_reason": "当前 DT 压力回放缓存没有第一部分工况标签",
            "source_condition_labels": [],
            "risk_max_pct": None,
            "risk_status": "\\",
            "suggestion": None,
            "pressure_max": _safe_max(pressure),
            "flow_max": _safe_max(flow),
            "sand_max": _safe_max(sand),
            "pressure_mean": _safe_mean(pressure),
            "flow_mean": _safe_mean(flow),
            "sand_mean": _safe_mean(sand),
            "prediction_record_count": 0,
            "data_source": f"DT缓存：{cache_path.name}",
            "point_prediction_source": "未接入",
            "point_prediction_method": "",
            "point_prediction_semantics": "当前仅回放已登记的压力、排量和砂比数据",
            "point_prediction_count": 0,
            "point_prediction_metrics": {},
            "timeline_mode": "pressure_replay",
            "chart_title": "施工压力回放",
        }

    def summaries(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._stages.values()]

    def refresh(self) -> list[dict[str, Any]]:
        """Refresh the timeline, rebuilding only when the validated cache is stale.

        The refresh button is also used after imports and model changes.  When
        neither the source signature nor the selected model has changed,
        forcing a full parse of every workbook only adds latency and makes the
        UI look hung.  ``_load_cache`` already validates all of those inputs,
        so reuse it here and rebuild only when validation fails.
        """

        self.module = self.registry.module("fsl")
        self.source_paths = self._resolve_sources()
        self.source_path = self.source_paths[0] if self.source_paths else None
        self.status = "not_available"
        self.status_reason = "未找到第一部分时序数据"
        self._stages = {}
        if not self._load_cache():
            self._load()
        if not self._stages:
            self._write_cache()
        return self.summaries()
