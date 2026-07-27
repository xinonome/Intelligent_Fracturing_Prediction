from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FiberApiTables:
    """Normalized tables parsed from the client's fiber-monitoring JSON."""

    meta: dict[str, Any]
    amplitude: pd.DataFrame
    stage_info: pd.DataFrame
    controls: pd.DataFrame


def fetch_fiber_json(
    base_url: str,
    well_name: str | None = None,
    stage: str | int | None = None,
    method: str = "GET",
    timeout: float = 10.0,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch the client-style fiber JSON through HTTP GET or POST.

    The expected service shape is like:
    http://host/gxjc?wellName=xx-xx&stage=2
    """

    params = dict(extra_params or {})
    if well_name is not None:
        params["wellName"] = well_name
    if stage is not None:
        params["stage"] = stage

    method = method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError("method must be GET or POST")

    if method == "GET":
        query = urllib.parse.urlencode(params)
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}{query}" if query else base_url
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    data = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(base_url, data=data, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_fiber_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_fiber_api_payload(payload: dict[str, Any], default_step_seconds: float = 10.0) -> FiberApiTables:
    """Parse client JSON into amplitude/stage/controls tables.

    Important field mapping:
    - amp.sj -> time axis for amplitude heatmap.
    - amp.js -> depth axis for amplitude heatmap.
    - amp.db rows -> [time_index, depth_index, amplitude].
    - stageInfo[].cluster[] -> per-cluster fluid/sand intake.
    """

    meta = {
        "status": payload.get("status"),
        "wellName": payload.get("wellName"),
        "stage": payload.get("stage"),
        "msg": payload.get("msg"),
    }
    amplitude = parse_amplitude(payload.get("amp") or {})
    stage_info = parse_stage_info(payload.get("stageInfo") or [], meta)
    controls = stage_info_to_controls(stage_info, default_step_seconds=default_step_seconds)
    return FiberApiTables(meta=meta, amplitude=amplitude, stage_info=stage_info, controls=controls)


def parse_amplitude(amp: dict[str, Any]) -> pd.DataFrame:
    times = list(amp.get("sj") or [])
    depths = list(amp.get("js") or [])
    rows = []
    for item in amp.get("db") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        time_idx = _safe_int(item[0])
        depth_idx = _safe_int(item[1])
        rows.append(
            {
                "time_index": time_idx,
                "depth_index": depth_idx,
                "time": times[time_idx] if 0 <= time_idx < len(times) else None,
                "depth_m": _safe_float(depths[depth_idx]) if 0 <= depth_idx < len(depths) else np.nan,
                "amplitude": _safe_float(item[2]),
            }
        )
    return pd.DataFrame(rows)


def parse_stage_info(stage_info: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    meta = meta or {}
    for step, record in enumerate(stage_info):
        clusters = record.get("cluster") or []
        for cluster in clusters:
            cluster_id = _safe_int(cluster.get("ch"))
            rows.append(
                {
                    "well_name": meta.get("wellName"),
                    "stage": record.get("yldh") or meta.get("stage"),
                    "step": step,
                    "time": record.get("sj"),
                    "balance_degree": _safe_float(record.get("lfjhcd")),
                    "cumulative_balance_degree": _safe_float(record.get("ljlfjhcd")),
                    "cluster_id": cluster_id,
                    "fracture_id": cluster_id,
                    "liquid_volume_m3": _safe_float(cluster.get("jyl")),
                    "sand_mass_t": _safe_float(cluster.get("jsl")),
                    "cumulative_liquid_volume_m3": _safe_float(cluster.get("ljjyl")),
                    "cumulative_sand_mass_t": _safe_float(cluster.get("ljjsl")),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "well_name",
                "stage",
                "step",
                "time",
                "balance_degree",
                "cumulative_balance_degree",
                "cluster_id",
                "fracture_id",
                "liquid_volume_m3",
                "sand_mass_t",
                "cumulative_liquid_volume_m3",
                "cumulative_sand_mass_t",
            ]
        )
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values(["step", "cluster_id"]).reset_index(drop=True)


def stage_info_to_controls(stage_info: pd.DataFrame, default_step_seconds: float = 10.0) -> pd.DataFrame:
    """Convert stageInfo cluster rows to the controls needed by PKN/EnKF.

    Output columns intentionally match existing demos:
    - fracture_id / cluster_id
    - flow_rate_m3_min
    - liquid_volume_m3
    - sand_mass_t
    - allocation_weight
    """

    if stage_info.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "time",
                "fracture_id",
                "cluster_id",
                "allocation_weight",
                "flow_rate_m3_min",
                "liquid_volume_m3",
                "sand_mass_t",
                "cumulative_liquid_volume_m3",
                "cumulative_sand_mass_t",
            ]
        )

    df = stage_info.copy()
    df["step_seconds"] = _infer_step_seconds(df, default_step_seconds)
    # Prefer incremental jyl/jsl. If they are zero but cumulative values exist,
    # recover increments from cumulative columns.
    df["liquid_increment_m3"] = df["liquid_volume_m3"].astype(float)
    df["sand_increment_t"] = df["sand_mass_t"].astype(float)
    for value_col, cum_col, out_col in [
        ("liquid_volume_m3", "cumulative_liquid_volume_m3", "liquid_increment_m3"),
        ("sand_mass_t", "cumulative_sand_mass_t", "sand_increment_t"),
    ]:
        if df[value_col].fillna(0.0).abs().sum() <= 1e-12 and df[cum_col].fillna(0.0).abs().sum() > 0:
            df[out_col] = (
                df.sort_values(["cluster_id", "step"])
                .groupby("cluster_id")[cum_col]
                .diff()
                .fillna(df[cum_col])
                .clip(lower=0.0)
            )

    df["liquid_increment_m3"] = df["liquid_increment_m3"].fillna(0.0).clip(lower=0.0)
    df["sand_increment_t"] = df["sand_increment_t"].fillna(0.0).clip(lower=0.0)
    df["flow_rate_m3_min"] = df["liquid_increment_m3"] / np.maximum(df["step_seconds"] / 60.0, 1e-9)
    total_by_step = df.groupby("step")["liquid_increment_m3"].transform("sum")
    df["allocation_weight"] = np.where(
        total_by_step > 1e-12,
        df["liquid_increment_m3"] / total_by_step,
        1.0 / df.groupby("step")["cluster_id"].transform("count").clip(lower=1),
    )
    return df[
        [
            "step",
            "time",
            "fracture_id",
            "cluster_id",
            "allocation_weight",
            "flow_rate_m3_min",
            "liquid_increment_m3",
            "sand_increment_t",
            "cumulative_liquid_volume_m3",
            "cumulative_sand_mass_t",
            "balance_degree",
            "cumulative_balance_degree",
        ]
    ].rename(columns={"liquid_increment_m3": "liquid_volume_m3", "sand_increment_t": "sand_mass_t"})


def controls_for_step(controls: pd.DataFrame, step: int) -> pd.DataFrame:
    return controls[controls["step"] == step].sort_values("cluster_id").reset_index(drop=True)


def _infer_step_seconds(df: pd.DataFrame, default_step_seconds: float) -> pd.Series:
    result = pd.Series(default_step_seconds, index=df.index, dtype=float)
    if "time" not in df or df["time"].isna().all():
        return result
    step_times = df.groupby("step")["time"].first().sort_index()
    diffs = step_times.diff().dt.total_seconds()
    diffs = diffs.fillna(diffs[diffs > 0].median() if (diffs > 0).any() else default_step_seconds)
    diffs = diffs.replace(0, default_step_seconds).fillna(default_step_seconds)
    return df["step"].map(diffs).fillna(default_step_seconds).astype(float)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default
