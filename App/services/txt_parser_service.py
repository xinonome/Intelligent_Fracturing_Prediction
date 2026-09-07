"""Strict reader for the original, positional nine-column construction TXT.

Comma and tab separated exports, with an optional exact Chinese header, are
accepted. All eight non-time columns are numeric (including type codes).
No rows are silently dropped and no missing/invalid values are imputed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re


TXT_HEADERS = (
    "累计液量", "排量", "累计砂量", "黏度", "支撑剂类型", "时间", "泵压", "砂比", "液体类型",
)
TXT_FIELD_MAPPING = dict(zip(
    TXT_HEADERS, ("LJYL", "PL", "LJSL", "YTND", "ZCJLX", "SGSJ", "SGBY", "SB", "YX"),
))
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_TIME = re.compile(
    r"[0-9]{4}(?P<sep>[-/])[0-9]{2}(?P=sep)[0-9]{2}[ T]"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?\Z"
)


class TxtValidationError(ValueError):
    """An actionable error referencing the physical source line and field."""


@dataclass(frozen=True)
class TxtRow:
    line_number: int
    values: tuple[str, ...]
    timestamp: datetime


@dataclass(frozen=True)
class ParsedTxt:
    rows: tuple[TxtRow, ...]
    encoding: str
    delimiter: str
    has_header: bool
    source_bytes: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()


def parse_txt(path: str | Path) -> ParsedTxt:
    """Validate the entire file; preserve the exact bytes used for validation."""
    source_bytes = Path(path).read_bytes()
    if source_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            text = source_bytes.decode(encoding, errors="strict")
            break
        except UnicodeError:
            continue
    else:
        raise TxtValidationError("TXT 编码无效；请使用 UTF-8、GB18030 或带 BOM 的 UTF-16")
    if "\x00" in text or "\ufffd" in text:
        raise TxtValidationError("TXT 含无效字符或编码损坏，不能替换字符后导入")

    rows: list[TxtRow] = []
    delimiter = ""
    has_header = False
    previous_time: datetime | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        first_line = not delimiter
        if first_line:
            delimiter = "\t" if "\t" in line else ","
        try:
            values = next(csv.reader([line], delimiter=delimiter, strict=True, skipinitialspace=True))
        except csv.Error as exc:
            raise TxtValidationError(f"第 {line_number} 行：分隔或引号格式错误：{exc}") from exc
        if len(values) != len(TXT_HEADERS):
            raise TxtValidationError(f"第 {line_number} 行：需要 9 列，实际 {len(values)} 列（仅支持逗号或制表符分隔）")
        stripped = tuple(value.strip() for value in values)
        if first_line and stripped == TXT_HEADERS:
            has_header = True
            continue
        for index, (field, value) in enumerate(zip(TXT_HEADERS, stripped)):
            if index == 5:
                continue
            if not _NUMBER.fullmatch(value):
                raise TxtValidationError(f"第 {line_number} 行「{field}」：必须为有限非负数值，实际 {value!r}")
            number = float(value)
            nonzero_mantissa = any(char in "123456789" for char in value.lower().split("e")[0])
            if not math.isfinite(number) or number < 0 or (number == 0 and nonzero_mantissa):
                raise TxtValidationError(f"第 {line_number} 行「{field}」：负值或数值超出可用范围，实际 {value!r}")
        value = stripped[5]
        try:
            if not _TIME.fullmatch(value):
                raise ValueError("需要完整日期和时分秒")
            timestamp = datetime.fromisoformat(value.replace("/", "-"))
        except ValueError as exc:
            raise TxtValidationError(
                f"第 {line_number} 行「时间」：无效时间 {value!r}；需要 YYYY-MM-DD HH:MM:SS（可带微秒）"
            ) from exc
        if previous_time is not None and timestamp <= previous_time:
            raise TxtValidationError(f"第 {line_number} 行「时间」：必须严格递增，不允许重复或倒序")
        previous_time = timestamp
        rows.append(TxtRow(line_number, tuple(values), timestamp))
    if not rows:
        raise TxtValidationError("TXT 没有数据行（空文件或仅有表头）")
    return ParsedTxt(tuple(rows), encoding, delimiter, has_header, source_bytes)


__all__ = ["TXT_HEADERS", "TXT_FIELD_MAPPING", "TxtValidationError", "TxtRow", "ParsedTxt", "parse_txt"]
