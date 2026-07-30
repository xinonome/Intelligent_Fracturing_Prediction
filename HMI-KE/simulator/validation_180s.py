from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Validation180sConfig:
    action_seconds: float = 60.0
    validation_seconds: float = 180.0
    bottomhole_pressure_max_mpa: float = 110.0
    net_pressure_max_mpa: float = 35.0
    abnormal_probability_max: float = 0.45
    sand_plug_probability_max: float = 0.35
    posterior_error_uncertainty_threshold: float = 0.30
    require_full_horizon: bool = True
    target_safe_rate: float = 1.0
    recovery_requires_final_safe: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def validate_180s(
    evaluation: pd.DataFrame,
    config: Validation180sConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Estimate whether a policy remains safe for 180s after each action.

    This function uses the rollout rows produced by train_rl_control_agent.py.
    It is a validation proxy until same-well field feedback is available.
    """

    cfg = config or Validation180sConfig()
    if evaluation.empty:
        return pd.DataFrame(), {"validation_available": False, "reason": "empty_evaluation"}

    horizon_steps = max(1, int(round(cfg.validation_seconds / max(cfg.action_seconds, 1e-6))))
    rows: list[dict] = []
    ordered = evaluation.sort_values(["episode", "step"]).reset_index(drop=True)
    candidate_windows = 0
    incomplete_windows = 0
    for episode, group in ordered.groupby("episode", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            candidate_windows += 1
            # Each row is the response after one 60-second action. Including
            # the current row plus the following two rows therefore covers the
            # complete 180-second post-decision interval.
            window = group.iloc[idx : idx + horizon_steps]
            if len(window) < horizon_steps:
                incomplete_windows += 1
                if cfg.require_full_horizon:
                    continue
            if window.empty:
                continue
            max_bhp = float(window.get("bottomhole_pressure_mpa", window.get("simulated_pressure_mpa", pd.Series([np.nan]))).max())
            max_net = float(window.get("net_pressure_mpa", pd.Series([np.nan])).max()) if "net_pressure_mpa" in window else np.nan
            max_abnormal = float(window.get("abnormal_probability", pd.Series([0.0])).max())
            max_sand_plug = float(window.get("sand_plug_probability", pd.Series([0.0])).max())
            max_posterior_error = float(window.get("posterior_error", pd.Series([0.0])).max())
            explicit_pre_action = "pre_action_abnormal_probability" in group.columns
            if explicit_pre_action:
                pre_source = row
                pre_bhp = float(row.get("pre_action_bottomhole_pressure_mpa", np.nan))
                pre_net = float(row.get("pre_action_net_pressure_mpa", np.nan))
                pre_abnormal = float(row.get("pre_action_abnormal_probability", 0.0))
                pre_sand_plug = float(row.get("pre_action_sand_plug_probability", 0.0))
                pre_action_known = True
            elif idx > 0:
                # Backward-compatible validation for historical rollout CSVs:
                # the previous 60-second response is the current pre-action state.
                pre_source = group.iloc[idx - 1]
                pre_bhp = float(pre_source.get("bottomhole_pressure_mpa", pre_source.get("simulated_pressure_mpa", np.nan)))
                pre_net = float(pre_source.get("net_pressure_mpa", np.nan))
                pre_abnormal = float(pre_source.get("abnormal_probability", 0.0))
                pre_sand_plug = float(pre_source.get("sand_plug_probability", 0.0))
                pre_action_known = True
            else:
                pre_bhp = pre_net = np.nan
                pre_abnormal = pre_sand_plug = np.nan
                pre_action_known = False
            pre_action_unsafe = bool(
                pre_action_known
                and ((np.isfinite(pre_bhp) and pre_bhp > cfg.bottomhole_pressure_max_mpa)
                or (np.isfinite(pre_net) and pre_net > cfg.net_pressure_max_mpa)
                or pre_abnormal > cfg.abnormal_probability_max
                or pre_sand_plug > cfg.sand_plug_probability_max)
            )
            uncertain_180s = bool(
                max_posterior_error > cfg.posterior_error_uncertainty_threshold
                or bool(window.get("uncertain", pd.Series([False])).astype(bool).any())
            )
            unsafe_180s = bool(
                (np.isfinite(max_bhp) and max_bhp > cfg.bottomhole_pressure_max_mpa)
                or (np.isfinite(max_net) and max_net > cfg.net_pressure_max_mpa)
                or max_abnormal > cfg.abnormal_probability_max
                or max_sand_plug > cfg.sand_plug_probability_max
                or bool(window.get("unsafe", pd.Series([False])).astype(bool).any())
            )
            final = window.iloc[-1]
            final_safe = bool(
                float(final.get("bottomhole_pressure_mpa", final.get("simulated_pressure_mpa", 0.0)))
                <= cfg.bottomhole_pressure_max_mpa
                and float(final.get("net_pressure_mpa", 0.0)) <= cfg.net_pressure_max_mpa
                and float(final.get("abnormal_probability", 0.0)) <= cfg.abnormal_probability_max
                and float(final.get("sand_plug_probability", 0.0)) <= cfg.sand_plug_probability_max
                and not bool(final.get("unsafe", False))
            )
            rows.append(
                {
                    "episode": int(episode),
                    "step": int(row["step"]),
                    "validation_window_steps": int(len(window)),
                    "max_bottomhole_or_simulated_pressure_mpa": max_bhp,
                    "max_net_pressure_mpa": max_net,
                    "max_abnormal_probability": max_abnormal,
                    "max_sand_plug_probability": max_sand_plug,
                    "max_posterior_error": max_posterior_error,
                    "unsafe_within_180s": unsafe_180s,
                    "uncertain_within_180s": uncertain_180s,
                    "pre_action_unsafe": pre_action_unsafe,
                    "pre_action_known": pre_action_known,
                    "preventive_window": bool(pre_action_known and not pre_action_unsafe),
                    "recovery_window": bool(pre_action_known and pre_action_unsafe),
                    "final_safe": final_safe,
                    "recovered_within_180s": bool(pre_action_unsafe and final_safe),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {
            "validation_available": False,
            "reason": "no_complete_180s_window",
            "candidate_windows": candidate_windows,
            "incomplete_windows": incomplete_windows,
            "config": cfg.to_dict(),
        }
    unsafe_rate = float(result["unsafe_within_180s"].mean())
    safe_rate = 1.0 - unsafe_rate
    preventive = result.loc[result["preventive_window"]]
    recovery = result.loc[result["recovery_window"]]
    unknown = result.loc[~result["pre_action_known"]]
    preventive_safe_rate = (
        float((~preventive["unsafe_within_180s"]).mean()) if not preventive.empty else None
    )
    recovery_rate = float(recovery["recovered_within_180s"].mean()) if not recovery.empty else None
    summary = {
        "validation_available": True,
        "validation_seconds": cfg.validation_seconds,
        "horizon_steps": horizon_steps,
        "candidate_windows": candidate_windows,
        "eligible_complete_windows": int(len(result)),
        "incomplete_windows": incomplete_windows,
        "coverage_rate": float(len(result) / max(candidate_windows, 1)),
        "unsafe_within_180s_rate": unsafe_rate,
        "safe_within_180s_rate": safe_rate,
        "preventive_windows": int(len(preventive)),
        "preventive_safe_within_180s_rate": preventive_safe_rate,
        "pass_preventive_180s_safety": bool(
            preventive_safe_rate is not None and preventive_safe_rate >= cfg.target_safe_rate
        ),
        "recovery_windows": int(len(recovery)),
        "recovered_within_180s_rate": recovery_rate,
        "unknown_pre_action_windows": int(len(unknown)),
        "uncertain_within_180s_rate": float(result["uncertain_within_180s"].mean()),
        "pass_180s_safety": bool(safe_rate >= cfg.target_safe_rate),
        "max_abnormal_probability_180s": float(result["max_abnormal_probability"].max()),
        "max_sand_plug_probability_180s": float(result["max_sand_plug_probability"].max()),
        "max_posterior_error_180s": float(result["max_posterior_error"].max()),
        "config": cfg.to_dict(),
    }
    return result, summary
