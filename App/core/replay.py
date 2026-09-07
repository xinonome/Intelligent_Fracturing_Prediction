"""Build the single frame stream consumed by every APP page."""

from __future__ import annotations

import os
from typing import Any

from .schemas import DTState, FSLState, HMIState, ReplayFrame
from .model_runtime import resolve_runtime_selection
from ..data.dt_loader import DTLoader, number
from ..data.hmi_loader import HMILoader, truthy
from ..data.registry_loader import RegistryLoader


def _float(value: Any, default: float = 0.0) -> float:
    result = number(value, None)
    return default if result is None else float(result)


def _optional_float(value: Any) -> float | None:
    result = number(value, None)
    return None if result is None else float(result)


def _hmi_value(value: Any, *, connected: bool) -> float | None:
    """Keep absent HMI measurements absent instead of turning them into 0."""

    return _optional_float(value) if connected else None


def build_replay_decision(hmi: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    option = str(hmi.get("high_level_option", "")).strip().lower()
    abnormal = _float(hmi.get("abnormal_probability"), 0.0)
    sand_plug = _float(hmi.get("sand_plug_probability"), 0.0)
    posterior_error = _float(hmi.get("posterior_error"), 0.0)
    unsafe = truthy(hmi.get("unsafe")) or truthy(hmi.get("severe_pressure_violation"))
    uncertain = truthy(hmi.get("uncertain")) or truthy(hmi.get("pkn_update_skipped"))

    if option == "safe" or unsafe or abnormal >= 0.45 or sand_plug >= 0.25:
        risk, main_risk = "high", "压力或异常风险接近安全边界"
        recommendation, requires_confirmation = "降低排量和砂比，暂停激进调整并请求工程师确认。", True
    elif option == "divert":
        risk, main_risk = "medium", "分簇进液/携砂不均衡"
        recommendation, requires_confirmation = "限制砂比增幅，优先进行分簇均衡并观察关键簇响应。", True
    elif option == "hold" or uncertain or posterior_error > 0.30:
        risk, main_risk = "medium", "模型不确定性偏高"
        recommendation, requires_confirmation = "保持当前动作，等待下一观测更新后再决定是否调整。", True
    else:
        risk, main_risk = "low", "当前未触发异常风险规则"
        recommendation, requires_confirmation = "允许小幅增加排量/砂比，继续监测压力和分簇响应。", False

    result = dict(fallback or {})
    result.update({
        "risk_level": risk,
        "main_risk": main_risk,
        "recommendation": recommendation,
        "requires_confirmation": requires_confirmation,
        "uncertainty": "medium" if uncertain or posterior_error > 0.30 else "low",
        "evidence": {
            "max_abnormal_probability": abnormal,
            "max_sand_plug_probability": sand_plug,
            "posterior_error": posterior_error,
            "unsafe": unsafe,
            "uncertain": uncertain,
        },
        "source": "rl_evaluation.csv（按统一时间进度映射）",
        "episode": hmi.get("episode", "--"),
        "step": hmi.get("step", "--"),
        "high_level_option": option or "--",
    })
    return result


def _cache_value(cache: dict[str, Any], name: str, index: int) -> float | None:
    values = cache.get("arrays", {}).get(name, [])
    if not values:
        return None
    return _float(values[min(max(index, 0), len(values) - 1)], 0.0)


def build_replay_frames(
    registry_loader: RegistryLoader | None = None,
    agent_model: str | None = None,
) -> list[dict[str, Any]]:
    """Build aligned frames, optionally using a selected agent replay.

    ``agent_model`` changes only the HMI replay source.  DT/FSL data remain
    aligned to the selected scenario and dataset, so the UI can compare SAC,
    TD3 and PPO without relabeling one model's output as another's.
    """

    registry_loader = registry_loader or RegistryLoader()
    runtime_selection = resolve_runtime_selection().to_dict()
    dt = DTLoader(registry_loader)
    hmi = HMILoader(registry_loader, model_id=agent_model)
    selected_model = hmi.model_info or {}
    if selected_model:
        runtime_selection["active_agent_model"] = selected_model.get("model_id")
        runtime_selection["active_agent_model_source"] = selected_model.get("evaluation_path")
    hmi_available = dt.hmi_available()
    hmi_rows = hmi.rows if hmi_available else []
    hmi_decisions = hmi.decisions if hmi_available else []
    hmi_connected = hmi_available and bool(hmi_rows or hmi_decisions)
    if hmi_connected:
        replay_length = max(len(dt.history), len(hmi_rows), len(hmi_decisions), dt.timeline_length())
    else:
        replay_length = dt.timeline_length()
    # Keep full-resolution arrays for charts, but use representative replay
    # nodes.  The DT history can contain thousands of one-second records while
    # the action/evaluation stream is much shorter; materialising every DT row
    # makes the first page wait tens of seconds without adding visible value.
    # The time axis and chart data remain full resolution in their cache.
    try:
        max_replay_frames = max(24, int(os.environ.get("IFP_MAX_REPLAY_FRAMES", "240")))
    except ValueError:
        max_replay_frames = 240
    if not hmi_available:
        max_replay_frames = min(max_replay_frames, 240)
    replay_length = min(replay_length, max_replay_frames)
    if replay_length == 0:
        return []

    scenario = registry_loader.scenario()
    stage_id = str(scenario.get("stage_id", "08") or "08")
    frames: list[dict[str, Any]] = []
    for index in range(replay_length):
        dt_state = dt.at(index, replay_length)
        hmi_row = hmi.at(index, replay_length) if hmi_connected else {}
        if hmi_connected:
            decision = build_replay_decision(hmi_row, hmi.decision_at(index, replay_length))
        else:
            decision = {
                "risk_level": "unknown",
                "main_risk": "智能体数据未接入当前井段",
                "recommendation": "当前仅展示施工曲线和压力校正结果，未接入本井段智能体建议。",
                "requires_confirmation": True,
                "uncertainty": "not_available",
                "evidence": {},
                "source": "未接入本井段 HMI/智能体结果",
                "high_level_option": "--",
            }
        posterior_error = dt_state.get("posterior_error")
        if posterior_error is None:
            posterior_error = _float(hmi_row.get("posterior_error"), 0.0)
        abnormal = _float(hmi_row.get("abnormal_probability"), 0.0)
        sand_plug = _float(hmi_row.get("sand_plug_probability"), 0.0)
        dt_source_length = max(len(dt.history), dt.timeline_length(), 1)
        dt_index = min(round(index / max(replay_length - 1, 1) * max(dt_source_length - 1, 0)), max(dt_source_length - 1, 0))
        cluster_balance_degree = _optional_float(hmi_row.get("cluster_balance_degree"))
        if cluster_balance_degree is None:
            cluster_balance_degree = _optional_float(dt_state.get("fiber_balance_degree"))
        frame = ReplayFrame(
            frame_id=index + 1,
            time_s=float(dt_state.get("time_s", index + 1)),
                stage=stage_id,
            fsl=FSLState(
                working_type=str(hmi_row.get("working_type", hmi_row.get("condition", "unknown")) or "unknown"),
                normal_probability=(1.0 - abnormal) if hmi_row.get("abnormal_probability") not in (None, "") else None,
                abnormal_probability=abnormal if hmi_row.get("abnormal_probability") not in (None, "") else None,
                abnormal_type=str(hmi_row.get("abnormal_type", "none") or "none"),
                rule_hits=[str(hmi_row.get("rule_hit"))] if hmi_row.get("rule_hit") else [],
            ),
            dt=DTState(
                surface_pressure_mpa=dt_state.get("surface_pressure_mpa"),
                prior_bottomhole_pressure_mpa=dt_state.get("prior_bottomhole_pressure_mpa"),
                observed_bottomhole_pressure_mpa=dt_state.get("observed_bottomhole_pressure_mpa"),
                bottomhole_pressure_mpa=dt_state.get("bottomhole_pressure_mpa"),
                net_pressure_mpa=dt_state.get("net_pressure_mpa"),
                cluster_balance_degree=dt_state.get("cluster_balance_degree"),
                fracture_length_m=dt_state.get("fracture_length_m"),
                fracture_width_m=dt_state.get("fracture_width_m"),
                cumulative_liquid_m3=_cache_value(
                    dt.cache,
                    "fiber_cumulative_liquid_m3" if dt.cache.get("arrays", {}).get("fiber_cumulative_liquid_m3") else "cumulative_liquid_m3",
                    dt_index,
                ),
                cumulative_sand_t=_cache_value(
                    dt.cache,
                    "fiber_cumulative_sand_t" if dt.cache.get("arrays", {}).get("fiber_cumulative_sand_t") else "cumulative_sand_t",
                    dt_index,
                ),
                prior_parameters=dt_state.get("prior_parameters", {}),
                posterior_parameters=dt_state.get("posterior_parameters", {}),
                prior_half_lengths_m=dt_state.get("prior_half_lengths_m", []),
                posterior_half_lengths_m=dt_state.get("posterior_half_lengths_m", []),
                prior_error=dt_state.get("prior_error"),
                posterior_error=posterior_error,
                prior_pressure_error=dt_state.get("prior_pressure_error"),
                posterior_pressure_error=dt_state.get("posterior_pressure_error"),
                runtime_ms=dt_state.get("runtime_ms"),
                allocation_mode=dt_state.get("allocation_mode", "unknown"),
                parameterized_allocation=bool(dt_state.get("parameterized_allocation", False)),
                clusters=dt_state.get("clusters", []),
                quality=dt_state.get("quality", {}),
            ),
            hmi=HMIState(
                current_flow_m3_min=_hmi_value(hmi_row.get("current_flow_m3_min"), connected=hmi_connected),
                current_sand_ratio_percent=_hmi_value(hmi_row.get("current_sand_ratio_percent"), connected=hmi_connected),
                recommended_flow_m3_min=_hmi_value(hmi_row.get("flow_m3_min"), connected=hmi_connected),
                # ``sand_ratio_percent`` is the environment output/current
                # action column in older replays.  The displayed
                # recommendation must prefer the explicit enriched field;
                # otherwise current and recommended sand ratio can be shown
                # as identical even when the policy proposed a change.
                recommended_sand_ratio_percent=_hmi_value(
                    hmi_row.get("recommended_sand_ratio_percent")
                    or hmi_row.get("sand_ratio_percent"),
                    connected=hmi_connected,
                ),
                high_level_action=str(hmi_row.get("high_level_option", "unknown") or "unknown"),
                risk_level=str(decision.get("risk_level", "unknown")),
                uncertainty=str(decision.get("uncertainty", "unknown")),
                requires_confirmation=bool(decision.get("requires_confirmation", True)),
                reward_components={
                    "integrated_reward": _float(hmi_row.get("integrated_reward"), 0.0),
                    "effectiveness": _float(hmi_row.get("effectiveness_reward"), 0.0),
                    "pressure_safety": _float(hmi_row.get("pressure_safety_penalty"), 0.0),
                    "abnormal_risk": _float(hmi_row.get("abnormal_risk_penalty"), 0.0),
                    "construction_cost": _float(hmi_row.get("construction_cost_penalty"), 0.0),
                    "cluster_balance_degree": cluster_balance_degree,
                    "cluster_balance_reward": _optional_float(hmi_row.get("cluster_balance_reward")),
                    "cluster_balance_improvement": _optional_float(hmi_row.get("cluster_balance_improvement")),
                    "cluster_balance_available": (
                        bool(truthy(hmi_row.get("cluster_balance_available")))
                        if hmi_row.get("cluster_balance_available") not in (None, "")
                        else cluster_balance_degree is not None
                    ),
                },
                warning_5min={"abnormal_probability": abnormal, "sand_plug_probability": sand_plug},
                validation_180s=hmi.validation() if hmi_connected else {},
                quality={
                    "source": str(hmi.eval_path) if hmi.eval_path else "missing",
                    "valid": bool(hmi_row),
                    "connected": hmi_connected,
                    "model_id": selected_model.get("model_id"),
                    "model_label": selected_model.get("label"),
                },
            ),
            alignment={
                "method": "normalized_progress" if len(dt.history) != len(hmi.rows) else "native_row_alignment",
                "dt_source": str(dt.history_path) if dt.history_path else "missing",
            "hmi_source": str(hmi.eval_path) if hmi_connected and hmi.eval_path else "not_connected",
            },
        )
        payload = frame.to_dict()
        payload["runtime_models"] = runtime_selection
        # Compatibility projection for existing reports and callers.
        clusters = dt_state.get("clusters", [])
        payload.update({
            "index": index,
            "replay_index": index + 1,
            "replay_total": replay_length,
            "dt_index": dt_index,
            "time_s": frame.time_s,
            "phase": dt_state.get("phase", "unknown"),
            "prior_bhp": _float(dt_state.get("prior_bottomhole_pressure_mpa")),
            "posterior_bhp": _float(dt_state.get("bottomhole_pressure_mpa")),
            "observed_bhp": _float(dt_state.get("observed_bottomhole_pressure_mpa")),
            "prior_liquid_error": _float(dt_state.get("prior_liquid_error")),
            "posterior_liquid_error": _float(dt_state.get("posterior_liquid_error")),
            "prior_pressure_error": _float(dt_state.get("prior_pressure_error")),
            "posterior_pressure_error": _float(dt_state.get("posterior_pressure_error")),
            "prior_sand_error": _float(dt_state.get("prior_sand_error")),
            "posterior_sand_error": _float(dt_state.get("posterior_sand_error")),
            "kalman_gain": _float(dt_state.get("kalman_gain")),
            "posterior_eprime": _float(dt_state.get("posterior_parameters", {}).get("E_prime_gpa")),
            "posterior_leakoff": _float(dt_state.get("posterior_parameters", {}).get("C_L_m_sqrt_s")),
            "posterior_viscosity": _float(dt_state.get("posterior_parameters", {}).get("mu_pa_s")),
            "posterior_min_stress": _float(dt_state.get("posterior_parameters", {}).get("sigma_min_mpa")),
            "within_15": bool(dt_state.get("within_15", False)),
            "clusters": clusters,
            "current_flow": _hmi_value(hmi_row.get("current_flow_m3_min"), connected=hmi_connected),
            "current_sand": _hmi_value(hmi_row.get("current_sand_ratio_percent"), connected=hmi_connected),
            "action_flow": _hmi_value(hmi_row.get("flow_m3_min"), connected=hmi_connected),
            "action_sand": _hmi_value(
                hmi_row.get("recommended_sand_ratio_percent")
                or hmi_row.get("sand_ratio_percent"),
                connected=hmi_connected,
            ),
            "hmi_pressure": _hmi_value(hmi_row.get("bottomhole_pressure_mpa"), connected=hmi_available),
            "hmi_abnormal": abnormal,
            "hmi_sand_plug": sand_plug,
            "hmi_reward": _float(hmi_row.get("integrated_reward")),
            "hmi_option": hmi_row.get("high_level_option", "--"),
            "hmi_episode": hmi_row.get("episode", "--"),
            "hmi_step": hmi_row.get("step", "--"),
            "hmi_model_id": selected_model.get("model_id"),
            "hmi_model_label": selected_model.get("label"),
            "hmi_model_source": selected_model.get("evaluation_path"),
            "decision": decision,
        })
        frames.append(payload)
    return frames
