from __future__ import annotations

import numpy as np
import pandas as pd


def geometry_metrics(snapshots, stage_id: str) -> pd.DataFrame:
    rows = []
    for snapshot in snapshots:
        front = np.asarray(snapshot.front_coordinates_local, dtype=float)
        if front.size == 0:
            u = np.array([0.0]); v = np.array([0.0])
        else:
            u, v = front[:, 0], front[:, 1]
        half_length = float(np.max(np.abs(u)))
        half_height = float(np.max(np.abs(v)))
        full_height = float(np.max(v) - np.min(v))
        width = float(snapshot.width_m)
        pressure = float(snapshot.fluid_pressure_pa)
        rows.append({
            "stage_id": stage_id,
            "time_s": float(snapshot.time_s),
            "half_length_m": half_length,
            "half_height_m": half_height,
            "full_height_m": full_height,
            "aspect_ratio": half_length / max(half_height, 1.0e-12),
            "injection_point_width_m": width,
            "max_width_m": width,
            "mean_width_m": width,
            "max_pressure_pa": pressure,
            "mean_pressure_pa": pressure,
            "net_pressure_pa": float(snapshot.net_pressure_pa),
            "fracture_volume_m3": float(snapshot.fracture_volume_m3),
            "front_speed_max_m_s": float(snapshot.front_velocity_m_s),
            "front_speed_mean_m_s": float(snapshot.front_velocity_m_s),
            "injected_volume_m3": float(snapshot.injected_volume_m3),
            "efficiency": float(snapshot.efficiency),
        })
    return pd.DataFrame(rows)


def barrier_contacts(metrics: pd.DataFrame, snapshots, stage_top_v: float, stage_bottom_v: float, tolerance_m: float) -> pd.DataFrame:
    rows = []
    for metric, snapshot in zip(metrics.to_dict("records"), snapshots):
        front = np.asarray(snapshot.front_coordinates_local, dtype=float)
        v = front[:, 1] if front.size else np.array([0.0])
        top = bool(abs(stage_top_v - float(np.max(v))) <= tolerance_m or float(np.max(v)) >= stage_top_v - tolerance_m)
        bottom = bool(abs(float(np.min(v)) - stage_bottom_v) <= tolerance_m or float(np.min(v)) <= stage_bottom_v + tolerance_m)
        row = dict(metric)
        row.update({"top_contact": top, "bottom_contact": bottom, "full_vertical_containment": top and bottom})
        rows.append(row)
    return pd.DataFrame(rows)


def handover_diagnostic(metrics: pd.DataFrame, contacts: pd.DataFrame, min_consecutive: int, vertical_eps: float, lateral_min: float) -> tuple[pd.DataFrame, float | None]:
    evidence = contacts.copy()
    if evidence.empty:
        return evidence, None
    t = evidence["time_s"].to_numpy(dtype=float)
    L = evidence["half_length_m"].to_numpy(dtype=float)
    H = evidence["full_height_m"].to_numpy(dtype=float)
    if len(evidence) == 1:
        dL = np.zeros(1); dH = np.zeros(1)
    else:
        dL = np.gradient(L, t); dH = np.gradient(H, t)
    evidence["dL_dt_m_s"] = dL
    evidence["dH_dt_m_s"] = dH
    candidate = (evidence["top_contact"] & evidence["bottom_contact"] & (np.abs(dH) <= vertical_eps) & (dL >= lateral_min)).to_numpy()
    handover: float | None = None
    for index in range(0, len(candidate) - min_consecutive + 1):
        if bool(candidate[index:index + min_consecutive].all()):
            handover = float(t[index])
            evidence["handover_candidate"] = candidate
            evidence["handover_time_s"] = handover
            return evidence, handover
    evidence["handover_candidate"] = candidate
    evidence["handover_time_s"] = np.nan
    return evidence, None
