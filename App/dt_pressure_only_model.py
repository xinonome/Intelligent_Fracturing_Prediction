"""Pressure-only PKN stage and assumed-cluster display model.

This module is intentionally a transparent display-level estimator.  It uses
the pressure-corrected bottom-hole pressure and the construction schedule to
produce a stage-level PKN evolution.  When a cluster count is supplied, the
stage result is distributed through a small stress-shadow/flow-competition
model.  The resulting cluster values are estimates, never DAS observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_assumed_cluster_positions(
    trajectory: pd.DataFrame,
    cluster_count: int,
    md_start_m: float | None = None,
    md_end_m: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interpolate assumed cluster positions along a configured stage interval."""

    count = max(int(cluster_count), 0)
    if trajectory.empty or count == 0:
        return [], {"status": "unavailable", "mode": "assumed_trajectory_interpolated"}
    required = {"measured_depth_m", "vertical_depth_m", "north_m", "east_m"}
    if not required.issubset(trajectory.columns):
        return [], {"status": "unavailable", "mode": "assumed_trajectory_interpolated"}
    ordered = trajectory.sort_values("measured_depth_m").drop_duplicates("measured_depth_m")
    md = ordered["measured_depth_m"].to_numpy(dtype=float)
    start = float(md[0] if md_start_m is None else np.clip(md_start_m, md[0], md[-1]))
    end = float(md[-1] if md_end_m is None else np.clip(md_end_m, md[0], md[-1]))
    if end <= start:
        start, end = float(md[0]), float(md[-1])
    assumed_md = np.linspace(start, end, count)
    positions = []
    for cluster_id, md_value in enumerate(assumed_md):
        positions.append(
            {
                "cluster_id": int(cluster_id),
                "measured_depth_m": float(md_value),
                "vertical_depth_m": float(np.interp(md_value, md, ordered["vertical_depth_m"])),
                "north_m": float(np.interp(md_value, md, ordered["north_m"])),
                "east_m": float(np.interp(md_value, md, ordered["east_m"])),
                "source_north_m": float(np.interp(md_value, md, ordered["north_m"])),
                "display_north_m": float(np.interp(md_value, md, ordered["north_m"])),
                "geometry_provenance": "assumed_cluster_count_trajectory_interpolated",
            }
        )
    return positions, {
        "status": "estimated",
        "mode": "assumed_trajectory_interpolated",
        "cluster_count": count,
        "md_start_m": start,
        "md_end_m": end,
        "warning": "簇位置由假设簇数和井轨迹插值得到，不是现场分簇几何观测。",
    }


def build_straight_assumed_cluster_positions(
    cluster_count: int,
    md_start_m: float = 0.0,
    md_end_m: float = 1600.0,
    vertical_depth_m: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create straight-line assumed cluster positions when no trajectory exists."""

    count = max(int(cluster_count), 0)
    if count == 0:
        return [], {"status": "unavailable", "mode": "straight_assumed_stage"}
    start = float(md_start_m)
    end = float(md_end_m if md_end_m > md_start_m else md_start_m + max(count - 1, 1) * 300.0)
    positions = []
    for cluster_id, md_value in enumerate(np.linspace(start, end, count)):
        positions.append(
            {
                "cluster_id": int(cluster_id),
                "measured_depth_m": float(md_value),
                "vertical_depth_m": float(vertical_depth_m),
                "north_m": 0.0,
                "east_m": float(md_value),
                "source_north_m": 0.0,
                "display_north_m": 0.0,
                "geometry_provenance": "assumed_straight_stage_no_trajectory",
            }
        )
    return positions, {
        "status": "estimated",
        "mode": "straight_assumed_stage",
        "cluster_count": count,
        "md_start_m": start,
        "md_end_m": end,
        "warning": "无井轨迹时采用直线假设井段；簇位置和簇级裂缝均为模型估计。",
    }


def estimate_pressure_only_multicluster(
    *,
    timeline_s: np.ndarray,
    flow_rate_m3_min: np.ndarray,
    cumulative_liquid_m3: np.ndarray,
    corrected_bhp_mpa: np.ndarray,
    min_horizontal_stress_mpa: float,
    cluster_positions: list[dict[str, Any]],
    e_prime_gpa: float = 32.47,
    viscosity_pa_s: float = 0.1015,
    fracture_height_m: float = 30.0,
    interaction_strength: float = 0.85,
    interaction_length_m: float = 450.0,
) -> dict[str, Any]:
    """Estimate PKN stage growth and redistribute it across assumed clusters.

    The base equations follow the project's PKN forward-model convention.  A
    bounded net-pressure multiplier makes the displayed expansion respond to
    the pressure-only correction.  The cluster redistribution is a reduced
    stress-shadow/competition term: nearby growing fractures suppress local
    intake, while the normalized total stage length remains conserved.
    """

    times = np.asarray(timeline_s, dtype=float)
    flow = np.maximum(np.asarray(flow_rate_m3_min, dtype=float), 0.0)
    cumulative = np.maximum.accumulate(np.maximum(np.asarray(cumulative_liquid_m3, dtype=float), 0.0))
    bhp = np.asarray(corrected_bhp_mpa, dtype=float)
    n = len(times)
    count = len(cluster_positions)
    if n == 0 or count == 0:
        return {"stage": {}, "clusters": {}, "metadata": {"status": "not_available"}}
    if not (len(flow) == len(cumulative) == len(bhp) == n):
        raise ValueError("pressure-only PKN inputs must share the timeline length")

    elapsed = np.maximum(times, 1.0)
    q_current = flow / 60.0
    # Fracture growth should follow effective injection time, not wall-clock
    # time.  Otherwise a long shut-in/idle interval in a construction export
    # creates a fictitious long fracture even though no liquid is entering the
    # formation.  Use cumulative injected volume relative to the first valid
    # sample and accumulate time only while the measured rate is positive.
    injected = np.maximum(cumulative - cumulative[0], 0.0)
    delta_t = np.diff(np.r_[elapsed[0], elapsed])
    delta_t = np.maximum(delta_t, 0.0)
    active = flow > 0.05
    active_elapsed = np.cumsum(np.where(active, delta_t, 0.0))
    active_elapsed = np.maximum(active_elapsed, 1.0)
    q_average = injected / active_elapsed
    q_effective = np.maximum(q_average, q_current * 0.25)
    e_prime_pa = max(float(e_prime_gpa), 1.0e-6) * 1.0e9
    height = max(float(fracture_height_m), 1.0)
    viscosity = max(float(viscosity_pa_s), 1.0e-7)
    width_base = 2.5 * ((q_effective**3 * viscosity) / (e_prime_pa * height**3)) ** 0.2 * active_elapsed**0.2
    length_base = 0.68 * ((q_effective**3 * e_prime_pa) / (viscosity * height**4)) ** 0.2 * active_elapsed**0.8

    net_raw = bhp - float(min_horizontal_stress_mpa)
    net = np.maximum(net_raw, 0.0)
    active_net = net[net > 1.0e-6]
    reference_net = float(np.percentile(active_net, 75)) if active_net.size else 1.0
    pressure_multiplier = np.clip((np.maximum(net, 1.0e-6) / max(reference_net, 1.0e-6)) ** 0.18, 0.70, 1.30)
    # A fracture front is a state variable: it does not retreat merely
    # because the instantaneous rate falls or the pressure trace has a
    # transient.  Retain the largest reached half-length for the display
    # evolution while allowing width and pressure to respond dynamically.
    stage_length_raw = np.maximum(length_base * pressure_multiplier, 0.0)
    stage_length = np.maximum.accumulate(stage_length_raw)
    stage_width = np.maximum(width_base * pressure_multiplier**0.35, 0.0)
    stage_volume = stage_width * (2.0 * stage_length / 1.25) * (np.pi * height / 4.0)

    md = np.asarray([float(row.get("measured_depth_m", index)) for index, row in enumerate(cluster_positions)])
    distance = np.abs(md[:, None] - md[None, :])
    kernel = np.exp(-distance / max(float(interaction_length_m), 1.0))
    np.fill_diagonal(kernel, 0.0)
    lengths = np.empty((count, n), dtype=float)
    shares = np.empty((count, n), dtype=float)
    factors = np.empty((count, n), dtype=float)
    shadow_loads = np.empty((count, n), dtype=float)
    equal_length = stage_length / count
    for time_index in range(n):
        current = np.full(count, equal_length[time_index], dtype=float)
        for _ in range(6):
            relative_length = current / max(float(np.mean(current)), 1.0e-9)
            # Do not row-normalize the kernel: edge clusters have fewer
            # neighbours and therefore receive the classic edge-relief
            # effect, while interior clusters carry a larger shadow load.
            shadow = kernel @ relative_length
            intake = np.exp(-float(interaction_strength) * shadow)
            intake = np.maximum(intake, 1.0e-6)
            share = intake / intake.sum()
            current = stage_length[time_index] * (share**0.6) / max(float(np.sum(share**0.6)), 1.0e-9)
        shares[:, time_index] = share
        lengths[:, time_index] = current
        factors[:, time_index] = current / max(float(equal_length[time_index]), 1.0e-9)
        shadow_loads[:, time_index] = shadow

    cluster_data: dict[str, dict[str, list[float]]] = {}
    for index in range(count):
        cluster_data[str(index)] = {
            "observed_liquid_share": [],
            "observed_sand_share": [],
            "estimated_liquid_share": shares[index].tolist(),
            "prior_half_length_m": equal_length.tolist(),
            "posterior_half_length_m": lengths[index].tolist(),
            "estimated_half_length_m": lengths[index].tolist(),
            "posterior_cluster_factor": factors[index].tolist(),
            "interaction_factor": factors[index].tolist(),
            "stress_shadow_load": shadow_loads[index].tolist(),
        }
    return {
        "stage": {
            "pkn_stage_half_length_m": stage_length.tolist(),
            "pkn_stage_max_aperture_mm": (2.0 * stage_width * 1000.0).tolist(),
            "pkn_stage_fracture_volume_m3": stage_volume.tolist(),
            "pkn_stage_net_pressure_mpa": net.tolist(),
            "pkn_stage_pressure_multiplier": pressure_multiplier.tolist(),
        },
        "clusters": cluster_data,
        "metadata": {
            "status": "estimated",
            "model": "pressure_corrected_pkn_with_stress_shadow_competition",
            "cluster_count": count,
            "interaction_strength": float(interaction_strength),
            "interaction_length_m": float(interaction_length_m),
            "length_conservation": "sum estimated cluster half-lengths equals stage PKN half-length",
            "warning": "簇级结果由假设簇数和簇间影响模型估计，不是 DAS/FracMonitor 观测。",
        },
    }
