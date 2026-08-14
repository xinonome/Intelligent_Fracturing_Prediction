"""Build the single frame stream consumed by every APP page."""

from __future__ import annotations

from typing import Any

from .schemas import DTState, FSLState, HMIState, ReplayFrame
from ..data.dt_loader import DTLoader, number
from ..data.hmi_loader import HMILoader, truthy
from ..data.registry_loader import RegistryLoader


def _float(value: Any, default: float = 0.0) -> float:
    result = number(value, None)
    return default if result is None else float(result)


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


def build_replay_frames(registry_loader: RegistryLoader | None = None) -> list[dict[str, Any]]:
    registry_loader = registry_loader or RegistryLoader()
    dt = DTLoader(registry_loader)
    hmi = HMILoader(registry_loader)
    replay_length = max(len(dt.history), len(hmi.rows), len(hmi.decisions))
    if replay_length == 0:
        return []

    frames: list[dict[str, Any]] = []
    for index in range(replay_length):
        dt_state = dt.at(index, replay_length)
        hmi_row = hmi.at(index, replay_length)
        decision = build_replay_decision(hmi_row, hmi.decision_at(index, replay_length))
        posterior_error = dt_state.get("posterior_error")
        if posterior_error is None:
            posterior_error = _float(hmi_row.get("posterior_error"), 0.0)
        abnormal = _float(hmi_row.get("abnormal_probability"), 0.0)
        sand_plug = _float(hmi_row.get("sand_plug_probability"), 0.0)
        dt_index = min(round(index / max(replay_length - 1, 1) * max(len(dt.history) - 1, 0)), max(len(dt.history) - 1, 0))
        frame = ReplayFrame(
            frame_id=index + 1,
            time_s=float(dt_state.get("time_s", index + 1)),
            stage="08",
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
                cumulative_liquid_m3=_cache_value(dt.cache, "fiber_cumulative_liquid_m3", dt_index),
                cumulative_sand_t=_cache_value(dt.cache, "fiber_cumulative_sand_t", dt_index),
                prior_parameters=dt_state.get("prior_parameters", {}),
                posterior_parameters=dt_state.get("posterior_parameters", {}),
                prior_half_lengths_m=dt_state.get("prior_half_lengths_m", []),
                posterior_half_lengths_m=dt_state.get("posterior_half_lengths_m", []),
                prior_error=dt_state.get("prior_error"),
                posterior_error=posterior_error,
                prior_pressure_error=dt_state.get("prior_pressure_error"),
                posterior_pressure_error=dt_state.get("posterior_pressure_error"),
                runtime_ms=dt_state.get("runtime_ms"),
                quality=dt_state.get("quality", {}),
            ),
            hmi=HMIState(
                current_flow_m3_min=_float(hmi_row.get("current_flow_m3_min"), 0.0),
                current_sand_ratio_percent=_float(hmi_row.get("current_sand_ratio_percent"), 0.0),
                recommended_flow_m3_min=_float(hmi_row.get("flow_m3_min"), 0.0),
                recommended_sand_ratio_percent=_float(hmi_row.get("sand_ratio_percent"), 0.0),
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
                },
                warning_5min={"abnormal_probability": abnormal, "sand_plug_probability": sand_plug},
                validation_180s=hmi.validation(),
                quality={"source": str(hmi.eval_path) if hmi.eval_path else "missing", "valid": bool(hmi_row)},
            ),
            alignment={
                "method": "normalized_progress" if len(dt.history) != len(hmi.rows) else "native_row_alignment",
                "dt_source": str(dt.history_path) if dt.history_path else "missing",
                "hmi_source": str(hmi.eval_path) if hmi.eval_path else "missing",
            },
        )
        payload = frame.to_dict()
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
            "current_flow": _float(hmi_row.get("current_flow_m3_min")),
            "current_sand": _float(hmi_row.get("current_sand_ratio_percent")),
            "action_flow": _float(hmi_row.get("flow_m3_min")),
            "action_sand": _float(hmi_row.get("sand_ratio_percent")),
            "hmi_pressure": _float(hmi_row.get("bottomhole_pressure_mpa")),
            "hmi_abnormal": abnormal,
            "hmi_sand_plug": sand_plug,
            "hmi_reward": _float(hmi_row.get("integrated_reward")),
            "hmi_option": hmi_row.get("high_level_option", "--"),
            "hmi_episode": hmi_row.get("episode", "--"),
            "hmi_step": hmi_row.get("step", "--"),
            "decision": decision,
        })
        frames.append(payload)
    return frames
