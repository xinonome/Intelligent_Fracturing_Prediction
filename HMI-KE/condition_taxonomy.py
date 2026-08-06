"""Shared working-condition taxonomy for the control pipeline.

Normal operating windows remain useful negative samples, but only explicit
abnormal working-condition labels are used as the abnormal-risk target.
Pressure limits, posterior error and cluster imbalance are dynamic signals;
they are not labels and are evaluated separately by the environment.
"""

from __future__ import annotations

from typing import Iterable


NORMAL_OPERATION_LABELS = frozenset({"正常", "正常工况", "主缝延伸"})
UNKNOWN_LABELS = frozenset({"", "nan", "none", "null", "??", "?", "未知"})

# These labels describe a condition requiring risk monitoring in the current
# project. "其他" is retained as an unresolved abnormal label rather than
# silently treating an annotated event as normal.
ABNORMAL_LABEL_PATTERNS = (
    "砂堵",
    "缝口暂堵",
    "缝内暂堵",
    "延伸受阻",
    "滤失过大",
    "缝高延伸",
    "新缝开启",
    "异常",
    "其他",
)


def split_labels(value: object) -> set[str]:
    """Normalize pipe-separated labels and discard blank/unknown markers."""

    if value is None:
        return set()
    text = str(value).strip()
    if text.lower() in UNKNOWN_LABELS:
        return set()
    return {
        part.strip().replace("正常工况", "正常")
        for part in text.split("|")
        if part.strip() and part.strip().lower() not in UNKNOWN_LABELS
    }


def is_abnormal_label(label: object) -> bool:
    """Return whether a label is an explicit abnormal-condition label."""

    text = str(label).strip().replace("正常工况", "正常")
    if not text or text in NORMAL_OPERATION_LABELS or text.lower() in UNKNOWN_LABELS:
        return False
    return any(pattern in text for pattern in ABNORMAL_LABEL_PATTERNS)


def has_abnormal_label(labels: Iterable[object]) -> bool:
    return any(is_abnormal_label(label) for label in labels)


def has_sand_plug_label(labels: Iterable[object]) -> bool:
    return any("砂堵" in str(label) for label in labels)


def classify_labels(labels: Iterable[object]) -> str:
    """Classify a label set as ``normal`` or explicit ``abnormal``."""

    normalized = split_labels("|".join(str(label) for label in labels))
    return "abnormal" if has_abnormal_label(normalized) else "normal"


RISK_POLICY = {
    "normal_labels": sorted(NORMAL_OPERATION_LABELS),
    "abnormal_label_patterns": list(ABNORMAL_LABEL_PATTERNS),
    "unknown_markers_excluded_from_positive_target": sorted(UNKNOWN_LABELS),
    "dynamic_risk_signals": [
        "pressure_limit",
        "cluster_imbalance",
        "posterior_error",
        "out_of_distribution",
    ],
    "planned_temporary_plugging_note": (
        "缝口暂堵/缝内暂堵当前按标注异常风险处理；若甲方提供泵序或施工意图，"
        "可进一步拆分 planned_diversion 与 unplanned_plugging。"
    ),
}
