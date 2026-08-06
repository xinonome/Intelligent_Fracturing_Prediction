from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from condition_taxonomy import has_abnormal_label, split_labels


NORMAL_LABEL = "正常"
REAL_SCENARIOS = (
    "baseline",
    "normal_growth",
    "sand_plug_risk",
    "cluster_imbalance",
    "pressure_limit",
    "diversion_stage",
    "other_abnormal",
)
SCENARIO_CLASSES = tuple(name for name in REAL_SCENARIOS if name != "baseline")


def _labels(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return {NORMAL_LABEL}
    parts = split_labels(value)
    return parts or {NORMAL_LABEL}


def classify_label_set(labels: set[str]) -> str:
    text = "|".join(labels)
    abnormal = {label for label in labels if has_abnormal_label([label])}
    if not abnormal:
        return "normal_growth"
    if "砂堵" in text:
        return "sand_plug_risk"
    if "缝口暂堵" in text:
        return "diversion_stage"
    if "缝内暂堵" in text:
        return "cluster_imbalance"
    return "other_abnormal"


def annotate_real_scenarios(meta: pd.DataFrame) -> pd.DataFrame:
    result = meta.reset_index(drop=True).copy()
    classes: list[str] = []
    labels_text: list[str] = []
    for _, row in result.iterrows():
        labels = _labels(row.get("state_working_types", "")) | _labels(row.get("future_working_types", ""))
        labels.discard("")
        labels_text.append("|".join(sorted(labels)))
        classes.append(classify_label_set(labels))
    result["observed_working_types"] = labels_text
    result["real_scenario_class"] = classes

    pressure = pd.to_numeric(result.get("current_pressure", pd.Series(np.zeros(len(result)))), errors="coerce")
    threshold = float(pressure.quantile(0.90)) if pressure.notna().any() else np.inf
    high_pressure = pressure >= threshold
    result.loc[high_pressure & result["real_scenario_class"].isin(["normal_growth", "other_abnormal"]), "real_scenario_class"] = "pressure_limit"
    return result


def select_real_scenario(
    features: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
    context: pd.DataFrame,
    scenario_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if scenario_name not in REAL_SCENARIOS:
        raise ValueError(f"Unsupported real scenario: {scenario_name}. Choices: {list(REAL_SCENARIOS)}")
    annotated = annotate_real_scenarios(meta)
    if scenario_name == "baseline":
        mask = np.ones(len(annotated), dtype=bool)
    else:
        mask = annotated["real_scenario_class"].eq(scenario_name).to_numpy()
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        counts = annotated["real_scenario_class"].value_counts().to_dict()
        raise ValueError(f"No samples for real scenario '{scenario_name}'. Available counts: {counts}")

    selected_meta = annotated.iloc[indices].reset_index(drop=True)
    selected_context = context.iloc[indices].reset_index(drop=True).copy()
    abnormal = selected_meta["real_scenario_class"].isin(
        {"sand_plug_risk", "diversion_stage", "cluster_imbalance", "pressure_limit", "other_abnormal"}
    ).astype(float).to_numpy()
    sand_plug = selected_meta["real_scenario_class"].eq("sand_plug_risk").astype(float).to_numpy()
    selected_context["abnormal_probability"] = abnormal
    selected_context["sand_plug_probability"] = sand_plug
    selected_context["scenario_name"] = selected_meta["real_scenario_class"].to_numpy()
    selected_meta["scenario_name"] = selected_meta["real_scenario_class"]

    counts = annotated["real_scenario_class"].value_counts().sort_index().to_dict()
    spec = {
        "name": scenario_name,
        "display_name": f"FSL真实场景：{scenario_name}",
        "description": "由同段300秒状态窗口、未来60秒动作窗口及WORKING_TYPE标签直接筛选，未对施工参数做人为缩放。",
        "source": "FSL-Expert/Data/raw_frac",
        "selected_samples": int(len(indices)),
        "all_scenario_counts": {str(key): int(value) for key, value in counts.items()},
    }
    return features[indices], targets[indices], selected_meta, selected_context, spec
