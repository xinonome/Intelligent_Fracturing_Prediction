from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Warning5MinConfig:
    action_seconds: float = 60.0
    warning_seconds: float = 300.0
    abnormal_warning_threshold: float = 0.30
    sand_plug_warning_threshold: float = 0.22
    pressure_warning_ratio: float = 0.90
    bottomhole_pressure_max_mpa: float = 110.0
    net_pressure_max_mpa: float = 35.0
    require_full_horizon: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def annotate_5min_warnings(
    evaluation: pd.DataFrame,
    config: Warning5MinConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate a five-minute rolling warning on simulated policy rollouts.

    This is an offline digital-twin validation. Each decision row looks at the
    following five 60-second simulated responses. It does not claim field lead
    time until the same interface is connected to live observations.
    """

    cfg = config or Warning5MinConfig()
    if evaluation.empty:
        return evaluation.copy(), {"available": False, "reason": "empty_evaluation"}
    horizon = max(1, int(round(cfg.warning_seconds / max(cfg.action_seconds, 1e-6))))
    rows: list[dict] = []
    for episode, group in evaluation.sort_values(["episode", "step"]).groupby("episode", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            future = group.iloc[idx + 1 : idx + 1 + horizon]
            if len(future) < horizon and cfg.require_full_horizon:
                continue
            if future.empty:
                continue
            bhp = future.get("bottomhole_pressure_mpa", future.get("simulated_pressure_mpa", pd.Series(0.0)))
            net = future.get("net_pressure_mpa", pd.Series(np.zeros(len(future))))
            abnormal = future.get("abnormal_probability", pd.Series(np.zeros(len(future))))
            sand_plug = future.get("sand_plug_probability", pd.Series(np.zeros(len(future))))
            pressure_warning = bool(
                float(bhp.max()) >= cfg.bottomhole_pressure_max_mpa * cfg.pressure_warning_ratio
                or float(net.max()) >= cfg.net_pressure_max_mpa * cfg.pressure_warning_ratio
            )
            risk_warning = bool(
                float(abnormal.max()) >= cfg.abnormal_warning_threshold
                or float(sand_plug.max()) >= cfg.sand_plug_warning_threshold
            )
            event_mask = (
                future.get("unsafe", pd.Series(False, index=future.index)).astype(bool)
                | (abnormal > 0.45)
                | (sand_plug > 0.35)
                | (bhp > cfg.bottomhole_pressure_max_mpa)
                | (net > cfg.net_pressure_max_mpa)
            )
            event = bool(event_mask.any())
            first_event_offset = int(np.argmax(event_mask.to_numpy()) + 1) if event else None
            warning = bool(pressure_warning or risk_warning)
            rows.append(
                {
                    "episode": int(episode),
                    "step": int(row["step"]),
                    "warning_horizon_seconds": float(cfg.warning_seconds),
                    "warning_5min": warning,
                    "predicted_event_within_5min": event,
                    "warning_reason": "pressure" if pressure_warning else ("abnormal_risk" if risk_warning else "none"),
                    "predicted_lead_seconds": (
                        float(first_event_offset * cfg.action_seconds) if first_event_offset is not None else np.nan
                    ),
                    "max_predicted_abnormal_probability": float(abnormal.max()),
                    "max_predicted_sand_plug_probability": float(sand_plug.max()),
                    "max_predicted_bottomhole_pressure_mpa": float(bhp.max()),
                    "max_predicted_net_pressure_mpa": float(net.max()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {"available": False, "reason": "no_complete_5min_window", "config": cfg.to_dict()}
    event_rows = result.loc[result["predicted_event_within_5min"]]
    full_lead_events = event_rows.loc[event_rows["predicted_lead_seconds"] >= cfg.warning_seconds]
    safe_rows = result.loc[~result["predicted_event_within_5min"]]
    summary = {
        "available": True,
        "scientific_status": "offline_digital_twin_rollout",
        "candidate_windows": int(len(result)),
        "event_windows": int(len(event_rows)),
        "warning_windows": int(result["warning_5min"].sum()),
        "event_warning_recall": (
            float(event_rows["warning_5min"].mean()) if not event_rows.empty else None
        ),
        "safe_window_false_warning_rate": (
            float(safe_rows["warning_5min"].mean()) if not safe_rows.empty else None
        ),
        "median_predicted_lead_seconds": (
            float(event_rows["predicted_lead_seconds"].median()) if not event_rows.empty else None
        ),
        "events_with_full_5min_lead": int(len(full_lead_events)),
        "full_5min_lead_warning_rate": (
            float(full_lead_events["warning_5min"].mean()) if not full_lead_events.empty else None
        ),
        "pass_strict_5min_lead": bool(
            not full_lead_events.empty and full_lead_events["warning_5min"].all()
        ),
        "config": cfg.to_dict(),
    }
    return result, summary


def summarize_decision_latency(evaluation: pd.DataFrame, limit_seconds: float = 15.0) -> dict:
    if "decision_compute_seconds" not in evaluation or evaluation.empty:
        return {"available": False, "limit_seconds": limit_seconds}
    values = pd.to_numeric(evaluation["decision_compute_seconds"], errors="coerce").dropna().to_numpy()
    if not len(values):
        return {"available": False, "limit_seconds": limit_seconds}
    p95 = float(np.percentile(values, 95))
    return {
        "available": True,
        "samples": int(len(values)),
        "mean_seconds": float(np.mean(values)),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": p95,
        "max_seconds": float(np.max(values)),
        "limit_seconds": float(limit_seconds),
        "pass_15s": bool(p95 <= limit_seconds),
    }


def evaluate_direct_5min_warning(
    predictions: pd.DataFrame,
    threshold: float = 0.30,
    target_recall: float = 0.90,
) -> dict:
    """Score a model whose target is any abnormal condition in the next 300s."""

    required = {"future_abnormal", "predicted_abnormal_probability"}
    if predictions.empty or not required.issubset(predictions.columns):
        return {"available": False, "reason": "missing_prediction_columns"}
    truth = pd.to_numeric(predictions["future_abnormal"], errors="coerce").fillna(0).astype(int).to_numpy()
    score = pd.to_numeric(predictions["predicted_abnormal_probability"], errors="coerce").fillna(0).to_numpy()
    pred = score >= threshold
    positive = truth == 1
    negative = ~positive
    tp = int(np.sum(pred & positive))
    fn = int(np.sum(~pred & positive))
    fp = int(np.sum(pred & negative))
    tn = int(np.sum(~pred & negative))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    false_alarm = fp / max(fp + tn, 1)
    return {
        "available": True,
        "scientific_status": "held_out_segment_300s_horizon",
        "samples": int(len(truth)),
        "positive_samples": int(np.sum(positive)),
        "threshold": float(threshold),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": float(recall),
        "precision": float(precision),
        "false_alarm_rate": float(false_alarm),
        "target_recall": float(target_recall),
        "pass_5min_warning_recall": bool(np.sum(positive) > 0 and recall >= target_recall),
        "boundary": "Target means an abnormal label occurs anywhere in the next 300 seconds; field prospective validation remains required.",
    }
