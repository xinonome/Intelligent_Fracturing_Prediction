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
    for episode, group in ordered.groupby("episode", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            window = group.iloc[idx + 1 : idx + 1 + horizon_steps]
            if window.empty:
                window = group.iloc[[idx]]
            max_bhp = float(window.get("bottomhole_pressure_mpa", window.get("simulated_pressure_mpa", pd.Series([np.nan]))).max())
            max_net = float(window.get("net_pressure_mpa", pd.Series([np.nan])).max()) if "net_pressure_mpa" in window else np.nan
            max_abnormal = float(window.get("abnormal_probability", pd.Series([0.0])).max())
            max_sand_plug = float(window.get("sand_plug_probability", pd.Series([0.0])).max())
            unsafe_180s = bool(
                (np.isfinite(max_bhp) and max_bhp > cfg.bottomhole_pressure_max_mpa)
                or (np.isfinite(max_net) and max_net > cfg.net_pressure_max_mpa)
                or max_abnormal > cfg.abnormal_probability_max
                or max_sand_plug > cfg.sand_plug_probability_max
                or bool(window.get("unsafe", pd.Series([False])).astype(bool).any())
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
                    "unsafe_within_180s": unsafe_180s,
                }
            )
    result = pd.DataFrame(rows)
    summary = {
        "validation_available": True,
        "validation_seconds": cfg.validation_seconds,
        "horizon_steps": horizon_steps,
        "unsafe_within_180s_rate": float(result["unsafe_within_180s"].mean()),
        "safe_within_180s_rate": float(1.0 - result["unsafe_within_180s"].mean()),
        "max_abnormal_probability_180s": float(result["max_abnormal_probability"].max()),
        "max_sand_plug_probability_180s": float(result["max_sand_plug_probability"].max()),
        "config": cfg.to_dict(),
    }
    return result, summary
