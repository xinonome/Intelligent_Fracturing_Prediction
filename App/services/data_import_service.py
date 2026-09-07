"""Import and preflight new construction tables for the desktop APP.

Imported tables are copied into the existing ``Data/raw_frac`` source area so
the two stage-level loaders can discover them using the same rules as the
shipped data. Legacy tables are copied unchanged. Original nine-column TXT
exports are strictly validated and mapped to CSV with explicit user identity
and lossless source provenance; no missing measurements or labels are invented.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4
import zipfile
import xml.etree.ElementTree as ET

from ..core.paths import PATHS
from ..data.dt_dataset_registry import clear_dataset_catalog_cache
from .txt_parser_service import TXT_FIELD_MAPPING, TXT_HEADERS, parse_txt


SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".txt"}
COMPOSITE_MARKERS = ("便签数据", "综合", "aggregate", "combined")
STANDARD_HEADERS = (
    "ID", "JTBH", "FDBH", "SGRQ", "SGSJ", "BZJDH", "YX", "YTND", "SGBY", "PL",
    "LJJLTJ", "SND", "SB", "LJSL", "ZCJLX", "ZDCLLX", "ZDCLYL", "QYSYSB",
    "QYTLSB", "SQTLSB", "SWTLSB", "SY", "ZDJ", "ZDQ", "LJYL", "ZDZSL", "ZDZYL",
    "JDPL", "JDSL", "BZJD", "ID_2", "JTH", "JD", "YL", "SHABI", "PL_2",
    "WORKING_TYPE", "PROBABILITY", "SUGGESTION", "MARK_TIME", "MARK_USER",
)
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"^([A-Za-z]+)")


@dataclass
class ImportInspection:
    source: Path
    extension: str
    headers: list[str] = field(default_factory=list)
    field_profile: str = "待识别"
    suggested_scenarios: list[str] = field(default_factory=list)
    can_import: bool = False
    reason: str = ""


@dataclass
class ImportResult:
    imported: list[Path] = field(default_factory=list)
    rejected: list[tuple[Path, str]] = field(default_factory=list)


def inspect_table(path: str | Path) -> ImportInspection:
    source = Path(path).expanduser().resolve()
    extension = source.suffix.lower()
    if not source.is_file():
        return ImportInspection(source, extension, reason="文件不存在")
    if extension not in SUPPORTED_EXTENSIONS:
        return ImportInspection(source, extension, reason="仅支持 .xlsx、.xls、.csv 和 .txt")
    lowered = source.name.lower()
    if any(marker.lower() in lowered for marker in COMPOSITE_MARKERS):
        return ImportInspection(
            source, extension, field_profile="综合数据",
            reason="检测为多井段综合文件，不作为单井段数据导入",
        )

    if extension == ".txt":
        try:
            parsed = parse_txt(source)
        except (OSError, ValueError) as exc:
            return ImportInspection(source, extension, field_profile="九列 TXT·校验失败", reason=str(exc))
        return ImportInspection(
            source, extension, headers=list(TXT_HEADERS), field_profile="九列 TXT 施工表",
            suggested_scenarios=["工况识别与风险预测", "无 DAS 原始数据登记"],
            can_import=True,
            reason=f"{len(parsed.rows)} 行通过完整数值/时间校验；需填写真实井名和井段，转为标准 CSV 并保留原文件",
        )

    headers = _read_headers(source)
    normalized = {_normalize_header(value) for value in headers if value}
    has_stage_time = {"fdbh", "sgsj"}.issubset(normalized)
    has_signals = bool(normalized.intersection({"pl", "s", "snd", "sb", "yx", "pressure", "pressurempa"}))
    if has_stage_time:
        profile = "标准施工表"
        reason = "已识别 FDBH、SGSJ；可导入并按独立井段扫描"
        can_import = True
    elif extension == ".xls":
        profile = "Excel旧格式·待转换"
        reason = "未能读取 .xls 表头；可保存原文件，但需转换为 .xlsx 后才能自动扫描"
        can_import = True
    elif headers and len(headers) >= 5:
        profile = "自定义表头·待字段映射"
        reason = "已读取表头，但未识别标准 FDBH/SGSJ；导入后需补充字段映射"
        can_import = True
    else:
        profile = "无表头施工表" if len(headers) >= 5 else "字段不足"
        reason = (
            "未发现表头，将按现有施工表位置字段读取；请确认文件为独立井段数据"
            if len(headers) >= 5 else "首行字段不足，暂不导入"
        )
        can_import = len(headers) >= 5
    if has_signals and has_stage_time:
        reason += "；已发现压力/排量/砂比相关字段"
    return ImportInspection(
        source=source,
        extension=extension,
        headers=headers,
        field_profile=profile,
        suggested_scenarios=["工况识别与风险预测", "无 DAS 原始数据登记"],
        can_import=can_import,
        reason=reason,
    )


def import_tables(
    paths: list[str | Path], destination: str | Path | None = None, *,
    well_name: str | None = None, stage_id: str | None = None,
    metadata_by_path: dict[str | Path, dict[str, str]] | None = None,
) -> ImportResult:
    """Import legacy tables unchanged; TXT requires explicit well/stage metadata.

    ``metadata_by_path`` overrides the common well_name/stage_id per source.
    TXT returns the standard CSV path, with ``.source.json`` beside it and
    the exact validated TXT bytes under ``originals/``. Missing standard
    measurements/labels remain blank, never fabricated as zero or NORMAL.
    """
    target = Path(destination) if destination else PATHS.data / "raw_frac"
    result = ImportResult()
    metadata = {
        str(Path(path).expanduser().resolve()).casefold(): value
        for path, value in (metadata_by_path or {}).items()
    }
    for raw_path in paths:
        inspection = inspect_table(raw_path)
        if not inspection.can_import:
            result.rejected.append((inspection.source, inspection.reason))
            continue
        if inspection.extension == ".txt":
            info = metadata.get(str(inspection.source).casefold(), {})
            try:
                destination_path = _import_txt(
                    inspection.source, target,
                    well_name=info.get("well_name", well_name), stage_id=info.get("stage_id", stage_id),
                )
            except (OSError, ValueError) as exc:
                result.rejected.append((inspection.source, str(exc)))
                continue
            result.imported.append(destination_path)
            continue
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_stem = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", inspection.source.stem).strip("_") or "table"
        destination_path = target / f"imported_{safe_stem}_{stamp}{inspection.extension}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inspection.source, destination_path)
        except OSError as exc:
            result.rejected.append((inspection.source, f"复制失败：{exc}"))
            continue
        result.imported.append(destination_path)
    if result.imported:
        clear_dataset_catalog_cache()
    return result


def _identity(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TXT 导入必须填写{label}；不会从文件名推断")
    value = value.strip()
    if value.casefold() in {"nan", "none", "null", "nat"} or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label}不能是空值标记或包含控制字符")
    return value


def _import_txt(source: Path, target: Path, *, well_name: str | None, stage_id: str | None) -> Path:
    well_name = _identity(well_name, "真实井名")
    stage_id = _identity(stage_id, "井段")
    parsed = parse_txt(source)  # Revalidate at commit, not just the UI preflight.
    imported_at = datetime.now().astimezone()
    # Physical identity is only in JTBH/FDBH, never inferred from this filename.
    stem = f"imported_txt_{imported_at:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:12]}"
    csv_path = target / f"{stem}.csv"
    original_path = target / "originals" / f"{stem}.txt"
    metadata_path = csv_path.with_suffix(".source.json")
    source_sha256 = parsed.sha256
    provenance = {
        "schema_version": 1,
        "source_format": "original_nine_column_txt",
        "source_path": str(source), "source_name": source.name,
        "source_sha256": source_sha256, "source_size_bytes": len(parsed.source_bytes),
        "original_file": original_path.relative_to(target).as_posix(),
        "standard_csv": csv_path.name, "imported_at": imported_at.isoformat(),
        "well_name": well_name, "stage_id": stage_id, "identity_source": "user_input",
        "encoding": parsed.encoding, "delimiter": parsed.delimiter, "has_header": parsed.has_header,
        "row_count": len(parsed.rows), "source_headers": list(TXT_HEADERS),
        "field_mapping": TXT_FIELD_MAPPING,
        "time_policy": "full local datetime; strictly increasing; no inferred date or resampling",
        "value_policy": "finite nonnegative numeric values; no imputation or unit conversion",
    }
    target.mkdir(parents=True, exist_ok=True)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish the CSV last so scanners never see a partially written table.
    with TemporaryDirectory(prefix=".txt_import_", dir=target) as staging:
        pending = Path(staging)
        (pending / "original.txt").write_bytes(parsed.source_bytes)
        (pending / "source.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        with (pending / "table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*STANDARD_HEADERS, *TXT_HEADERS, "SOURCE_FILE", "SOURCE_LINE", "SOURCE_SHA256"])
            writer.writeheader()
            for row in parsed.rows:
                record = {TXT_FIELD_MAPPING[field]: value.strip() for field, value in zip(TXT_HEADERS, row.values)}
                record.update(dict(zip(TXT_HEADERS, row.values)))
                record.update({
                    "ID": row.line_number, "JTBH": well_name, "FDBH": stage_id,
                    "SGRQ": row.timestamp.date().isoformat(),
                    "SOURCE_FILE": source.name, "SOURCE_LINE": row.line_number, "SOURCE_SHA256": source_sha256,
                })
                writer.writerow(record)
        published: list[Path] = []
        try:
            for staged, final in (("original.txt", original_path), ("source.json", metadata_path), ("table.csv", csv_path)):
                (pending / staged).rename(final)
                published.append(final)
        except OSError:
            for path in reversed(published):
                path.unlink()
            raise
    return csv_path


def _read_headers(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.reader(handle), [])
            return [str(value).strip() for value in row]
        if path.suffix.lower() == ".xls":
            return []
        return _read_xlsx_first_row(path)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile):
        return []


def _read_xlsx_first_row(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(item.itertext()) for item in root.findall("m:si", _XLSX_NS)]
        worksheet = next(
            (name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")),
            "",
        )
        if not worksheet:
            return []
        root = ET.fromstring(archive.read(worksheet))
        row_node = root.find(".//m:sheetData/m:row", _XLSX_NS)
        if row_node is None:
            return []
        cells: dict[int, str] = {}
        for cell in row_node.findall("m:c", _XLSX_NS):
            ref = _CELL_REF.match(cell.attrib.get("r", ""))
            if not ref:
                continue
            index = 0
            for character in ref.group(1).upper():
                index = index * 26 + ord(character) - ord("A") + 1
            value_node = cell.find("m:v", _XLSX_NS)
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell.attrib.get("t") == "s":
                try:
                    value = shared_strings[int(value)]
                except (IndexError, ValueError):
                    value = ""
            if cell.attrib.get("t") == "inlineStr":
                inline = cell.find("m:is", _XLSX_NS)
                value = "" if inline is None else "".join(inline.itertext())
            cells[index - 1] = value
        return [cells.get(index, "") for index in range(max(cells) + 1)] if cells else []


def _normalize_header(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value).strip().lower())


__all__ = ["ImportInspection", "ImportResult", "SUPPORTED_EXTENSIONS", "inspect_table", "import_tables"]
