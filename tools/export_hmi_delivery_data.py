"""Export the redacted HMI evaluation sequence into the standalone delivery package."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "hmi" / "bugfix_validation" / "ppo_20260812_184048" / "rl_evaluation.csv"
TARGET = ROOT / "deliverables" / "third_part_hmi_delivery" / "data" / "hmi_demo_sample.json"
MANIFEST = ROOT / "deliverables" / "third_part_hmi_delivery" / "manifest.json"


def as_float(value, digits: int = 3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def derive_risk(row):
    abnormal = as_float(row.get("abnormal_probability"), 6) or 0.0
    sand_plug = as_float(row.get("sand_plug_probability"), 6) or 0.0
    posterior_error = as_float(row.get("posterior_error"), 6) or 0.0
    unsafe = as_bool(row.get("unsafe")) or as_bool(row.get("severe_pressure_violation"))
    uncertain = as_bool(row.get("uncertain")) or as_bool(row.get("pkn_update_skipped"))
    high_sand_boundary = as_bool(row.get("sand_ratio_requires_confirmation")) or as_bool(
        row.get("sand_ratio_limit_reached")
    )
    if high_sand_boundary:
        risk = "high"
        recommendation = "砂比已达到高风险边界，禁止自动升砂，需工程师确认后才能改变施工砂比。"
    elif unsafe or abnormal >= 0.45 or sand_plug >= 0.25:
        risk = "high"
        recommendation = "降低排量和砂比，暂停激进调整并请求工程师确认。"
    elif row.get("high_level_option", "").strip().lower() == "hold" or uncertain or posterior_error > 0.30:
        risk = "medium"
        recommendation = "保持当前动作，等待下一观测更新后再决定是否调整。"
    else:
        risk = "low"
        recommendation = "允许小幅调整排量和砂比，继续监测压力和分簇响应。"
    return risk, "medium" if uncertain or posterior_error > 0.30 else "low", unsafe, uncertain, recommendation


def export() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("HMI evaluation source is empty")

    records = []
    for index, row in enumerate(rows):
        risk, uncertainty, unsafe, uncertain, recommendation = derive_risk(row)
        records.append({
            "sample_id": f"DEMO-{index + 1:03d}",
            "decision_index": index,
            "current_flow_m3_min": as_float(row.get("pre_action_flow_m3_min")),
            "recommended_flow_m3_min": as_float(row.get("flow_m3_min")),
            "current_sand_ratio_percent": as_float(row.get("pre_action_sand_ratio_percent")),
            "recommended_sand_ratio_percent": as_float(row.get("sand_ratio_percent")),
            "sand_reference_ratio_percent": as_float(row.get("sand_reference_ratio_percent")),
            "sand_delta_from_reference_percent": as_float(row.get("sand_delta_from_reference_percent")),
            "sand_ratio_limit_reached": as_bool(row.get("sand_ratio_limit_reached")),
            "sand_ratio_requires_confirmation": as_bool(row.get("sand_ratio_requires_confirmation")),
            "sand_control_mode": row.get("sand_control_mode", ""),
            "action": row.get("high_level_option", "unknown"),
            "risk_level": risk,
            "uncertainty": uncertainty,
            "requires_confirmation": risk in {"medium", "high"},
            "bottomhole_pressure_mpa": as_float(row.get("bottomhole_pressure_mpa")),
            "net_pressure_mpa": as_float(row.get("net_pressure_mpa")),
            "posterior_error": as_float(row.get("posterior_error")),
            "abnormal_probability": as_float(row.get("abnormal_probability")),
            "sand_plug_probability": as_float(row.get("sand_plug_probability")),
            "unsafe": unsafe,
            "uncertain": uncertain,
            "integrated_reward": as_float(row.get("integrated_reward")),
            "effectiveness_reward": as_float(row.get("effectiveness_reward")),
            "pressure_safety_penalty": as_float(row.get("pressure_safety_penalty")),
            "abnormal_risk_penalty": as_float(row.get("abnormal_risk_penalty")),
            "construction_cost_penalty": as_float(row.get("construction_cost_penalty")),
            "recommendation": recommendation,
        })

    package = {
        "schema_version": 1,
        "data_scope": {
            "source_record_count": len(rows),
            "package_record_count": len(records),
            "sampling_ratio": 1.0,
            "sampling_ratio_display": "100% of the HMI evaluation sequence",
            "anonymized": True,
            "rounded": True,
            "timestamps_rebased": True,
            "source_type": "HMI evaluation output; no raw fiber data included",
            "purpose": "third_part_hmi_demo_only",
        },
        "safety_policy": {
            "control_mode": "observed_reference_micro_adjustment",
            "max_advisory_sand_increase_percent": 0.5,
            "max_advisory_sand_decrease_percent": 3.0,
            "absolute_high_sand_boundary_percent": 14.0,
            "boundary_crossing_requires_confirmation": True,
            "note": "推荐砂比不得脱离当前观测值自行累加；高砂比不作为模型默认目标。",
        },
        "validation_summary": {
            "window_seconds": 180,
            "safe_rate": 1.0,
            "status": "demo_only",
            "note": "示例摘要，不代表现场验收通过。",
        },
        "records": records,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update({
        "source_record_count": len(rows),
        "included_record_count": len(records),
        "included_ratio": 1.0,
        "data_type": "HMI evaluation output; no raw fiber data included",
    })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_records": len(rows), "package_records": len(records), "output": str(TARGET)}, ensure_ascii=False))


if __name__ == "__main__":
    export()
