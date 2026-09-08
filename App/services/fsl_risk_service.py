"""Runtime bridge for the reviewed FSL rule and risk-model deliverables.

The research package contains two kinds of outputs:

* the causal sand-warning rules used by ``8.4RULE.py``;
* Z6HF/Z7HF ``KnowledgeTemporalRiskGNN`` state dictionaries.

The checkpoints do not contain their fitted imputer/scaler.  They are therefore
exposed as experimental models and use stage-local standardisation.  The rule
engine remains the auditable production fallback and never needs PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from ..core.paths import PATHS, resolve


FEATURE_COLUMNS = ("SGBY", "PL", "SB", "LJYL", "LJSL", "BZJD", "YTND", "JDPL", "JDSL")
LEVEL_NAMES = ("绿色", "黄色", "红色")
MODEL_CONFIG = PATHS.config / "fsl_risk_models.json"
RUNTIME_CONFIG = PATHS.config / "runtime_config.json"
RISK_RUNTIME_VERSION = "2026.09.08-1"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _std(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return float("inf")
    mean = sum(numbers) / len(numbers)
    return math.sqrt(sum((value - mean) ** 2 for value in numbers) / len(numbers))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def model_catalog() -> dict[str, dict[str, Any]]:
    models = _load_json(MODEL_CONFIG).get("models", {})
    return {str(key): dict(value) for key, value in models.items() if isinstance(value, dict)}


def selected_model_id() -> str:
    default = str(_load_json(MODEL_CONFIG).get("default_model", "rule_84"))
    configured = str(_load_json(RUNTIME_CONFIG).get("fsl_risk_model", default))
    return configured if configured in model_catalog() else default


def save_model_selection(model_id: str) -> None:
    if model_id not in model_catalog():
        raise ValueError(f"未知风险模型：{model_id}")
    payload = _load_json(RUNTIME_CONFIG)
    payload["fsl_risk_model"] = model_id
    RUNTIME_CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def model_options() -> list[tuple[str, str]]:
    result = []
    for model_id, spec in model_catalog().items():
        path = resolve(spec.get("path"))
        enabled = model_id in {"auto", "rule_84"} or bool(path and path.is_file())
        title = str(spec.get("display_name") or model_id)
        result.append((model_id, title if enabled else f"{title}（权重缺失）"))
    return result


@dataclass
class RiskAnalysis:
    probability: list[float | None]
    levels: list[str]
    rule_levels: list[str]
    rule_intervals: list[dict[str, Any]]
    predicted_intervals: list[dict[str, Any]]
    model_id: str
    model_source: str
    model_status: str
    model_reason: str


def _window_start(times: list[float], end: int, seconds: float = 60.0) -> int:
    start = end
    while start > 0 and times[end] - times[start - 1] <= seconds:
        start -= 1
    return start


def _rule_series(
    times: list[float], pressure: list[float | None], rate: list[float | None], sand: list[float | None]
) -> tuple[list[str], list[float | None], list[str]]:
    """Port the causal sand-warning core from 8.4RULE to point inference."""

    levels = ["绿色"] * len(times)
    probabilities: list[float | None] = [None] * len(times)
    reasons = [""] * len(times)
    previous_peak: float | None = None
    previous_spread: float | None = None
    for end in range(len(times)):
        start = _window_start(times, end)
        p = pressure[start : end + 1]
        q = rate[start : end + 1]
        s = sand[start : end + 1]
        p_valid = [value for value in p if value is not None]
        q_valid = [value for value in q if value is not None]
        s_valid = [value for value in s if value is not None]
        if len(p_valid) < 3 or len(p_valid) < len(p) * 0.7:
            continue
        q_std = _std(q_valid)
        s_std = _std(s_valid)
        q_rising = (
            len(q_valid) >= 2
            and q_valid[-1] > q_valid[0] + 0.2
            and max(q_valid) - min(q_valid) > 0.3
        )
        p_delta = p_valid[-1] - p_valid[0]
        rise_score = max(0.0, min(p_delta / 5.0, 1.0))
        drop_score = max(0.0, min(-p_delta / 5.0, 1.0))
        rate_stability = max(0.0, min(1.0 - q_std / 0.2, 1.0))
        sand_stability = max(0.0, min(1.0 - s_std / 1.0, 1.0))
        probabilities[end] = max(0.02, min(0.98, 0.65 * rise_score * rate_stability * sand_stability + 0.90 * drop_score))
        if q_rising or q_std > 0.2 or s_std > 1.0:
            probabilities[end] = min(probabilities[end] or 0.02, 0.45)
            previous_peak = None
            previous_spread = None
            reasons[end] = "排量或砂比处于调整期，规则不触发"
            continue
        differences = [right - left for left, right in zip(p_valid, p_valid[1:])]
        increasing = all(value >= -1e-9 for value in differences)
        decreasing = all(value <= 1e-9 for value in differences)
        if decreasing and p_delta < -5.0:
            levels[end] = "红色"
            probabilities[end] = max(probabilities[end] or 0.0, 0.90)
            reasons[end] = f"60秒窗口压力持续下降 {abs(p_delta):.2f} MPa，命中砂堵风险规则"
            previous_peak = None
            previous_spread = None
            continue
        if increasing and p_delta > 0.5:
            peak = max(p_valid)
            spread = _std(p_valid)
            peak_increasing = previous_peak is not None and peak > previous_peak
            spread_increasing = previous_spread is not None and spread > previous_spread
            duration = max(times[end] - times[start], 0.0)
            slope_per_minute = p_delta / (duration / 60.0) if duration > 0 else 0.0
            if peak_increasing and spread_increasing and (duration >= 60.0 or slope_per_minute > 5.0):
                levels[end] = "黄色"
                probabilities[end] = max(probabilities[end] or 0.0, 0.55)
                reasons[end] = f"压力持续上涨 {p_delta:.2f} MPa，波峰和波动同步增大"
            previous_peak = peak
            previous_spread = spread
        else:
            probabilities[end] = min(probabilities[end] or 0.02, 0.45)
            previous_peak = None
            previous_spread = None
    return levels, probabilities, reasons


def _intervals(
    times: list[float], levels: list[str], reasons: list[str], *, kind: str, source: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start: int | None = None
    active = "绿色"
    for index in range(len(levels) + 1):
        level = levels[index] if index < len(levels) else "绿色"
        if level != active:
            if start is not None and active != "绿色":
                end_index = max(start, index - 1)
                result.append(
                    {
                        "label": "砂堵迹象" if active == "黄色" else "砂堵风险",
                        "risk_level": active,
                        "start_s": float(times[start]),
                        "end_s": float(times[end_index]),
                        "kind": kind,
                        "trigger_reason": reasons[end_index] or source,
                        "source": source,
                    }
                )
            start = index if level != "绿色" else None
            active = level
    return result


def _neural_predictions(
    model_id: str,
    records: list[dict[str, Any]],
    times: list[float],
    rule_levels: list[str],
) -> tuple[list[float | None], list[str], str]:
    """Run a reviewed state-dict with explicit stage-local preprocessing."""

    spec = model_catalog()[model_id]
    checkpoint = resolve(spec.get("path"))
    if not checkpoint or not checkpoint.is_file():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint}")
    import numpy as np
    raw = np.asarray(
        [[np.nan if _number(row.get(name)) is None else float(row.get(name)) for name in FEATURE_COLUMNS] for row in records],
        dtype=np.float32,
    )
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(raw), raw, medians)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = (filled - mean) / scale
    valid_indices = [
        index
        for index in range(5, len(times))
        if all(0.0 < times[item] - times[item - 1] <= 30.0 for item in range(index - 4, index + 1))
    ]
    probability: list[float | None] = [None] * len(times)
    levels = ["绿色"] * len(times)
    if not valid_indices:
        return probability, levels, "当前井段没有连续的6点模型窗口"
    windows = np.stack([normalized[index - 5 : index + 1] for index in valid_indices]).astype(np.float32)
    rules = []
    current = []
    for index in valid_indices:
        window = filled[index - 5 : index + 1]
        p_delta = float(window[-1, 0] - window[0, 0])
        p_rise = float(np.clip(max(p_delta, 0.0) / 5.0, 0.0, 1.0))
        p_drop = float(np.clip(max(-p_delta, 0.0) / 5.0, 0.0, 1.0))
        stability = float(np.clip(1.0 - np.std(window[:, 1]) / 0.2, 0.0, 1.0) * np.clip(1.0 - np.std(window[:, 2]) / 1.0, 0.0, 1.0))
        rules.append([p_rise, p_drop, stability])
        current.append(LEVEL_NAMES.index(rule_levels[index]))
    context = np.eye(3, dtype=np.float32)[np.asarray(current, dtype=np.int64)]
    if checkpoint.suffix.lower() == ".npz":
        output = _numpy_model_predict(
            windows,
            np.asarray(rules, dtype=np.float32),
            context,
            str(checkpoint),
            float(spec.get("prior_strength", 1.0)),
        )
    else:
        import torch

        model = _load_neural_model(model_id, str(checkpoint))
        output_batches = []
        with torch.no_grad():
            for start in range(0, len(windows), 1024):
                stop = min(start + 1024, len(windows))
                context_batch = torch.from_numpy(context[start:stop])
                logits = model(
                    torch.from_numpy(windows[start:stop]),
                    torch.tensor(rules[start:stop], dtype=torch.float32),
                    context_batch,
                )
                logits = logits + float(spec.get("prior_strength", 1.0)) * context_batch
                output_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        output = np.concatenate(output_batches, axis=0)
    for row_index, point_index in enumerate(valid_indices):
        level_index = int(output[row_index].argmax())
        levels[point_index] = LEVEL_NAMES[level_index]
        probability[point_index] = float(output[row_index, 1] + output[row_index, 2])
    return probability, levels, "权重已加载；缺失的训练预处理器由当前井段自适应重建"


@lru_cache(maxsize=2)
def _load_neural_model(model_id: str, checkpoint_text: str):
    import torch
    from torch import nn

    expert = PATHS.root / "FSL-Expert"
    if str(expert) not in sys.path:
        sys.path.insert(0, str(expert))
    from frac_gnn.future_risk_gnn import KnowledgeTemporalRiskGNN

    model = KnowledgeTemporalRiskGNN(
        base_feature_dim=9,
        window_size=6,
        hidden_dim=48,
        dropout=0.2,
        use_rule_nodes=True,
        context_dim=3,
    )
    model.classifier[-1] = nn.Linear(48, 3)
    state = torch.load(Path(checkpoint_text), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=2)
def _load_numpy_state(checkpoint_text: str) -> dict[str, Any]:
    import numpy as np

    with np.load(checkpoint_text) as archive:
        return {key: archive[key].astype(np.float32) for key in archive.files}


def _softmax(values):
    import numpy as np

    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _layer_norm(values, weight, bias, epsilon: float = 1e-5):
    import numpy as np

    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon) * weight + bias


def _attention(nodes, adjacency, projection, source_weight, target_weight):
    import numpy as np

    hidden = nodes @ projection.T
    source = hidden @ source_weight.T
    target = (hidden @ target_weight.T).transpose(0, 2, 1)
    logits = source + target
    logits = np.where(logits >= 0.0, logits, logits * 0.2)
    logits = np.where(adjacency[None, :, :], logits, -1e30)
    weights = _softmax(logits)
    return weights @ hidden


def _numpy_model_predict(windows, rules, context, checkpoint_text: str, prior_strength: float):
    """Numerically mirror ``KnowledgeTemporalRiskGNN.eval()`` using NumPy."""

    import numpy as np

    state = _load_numpy_state(checkpoint_text)
    batch, window_size, base_dim = windows.shape
    time_types = np.zeros((batch, window_size, 4), dtype=np.float32)
    time_types[:, :, 0] = 1.0
    time_nodes = np.concatenate(
        [windows, time_types, np.zeros((batch, window_size, 1), dtype=np.float32)], axis=-1
    )
    rule_features = np.zeros((batch, 3, base_dim), dtype=np.float32)
    rule_types = np.zeros((batch, 3, 4), dtype=np.float32)
    for index in range(3):
        rule_types[:, index, index + 1] = 1.0
    rule_nodes = np.concatenate([rule_features, rule_types, rules[:, :3, None]], axis=-1)
    nodes = np.concatenate([time_nodes, rule_nodes], axis=1)
    adjacency = state["adjacency"].astype(bool)
    hidden = _attention(
        nodes,
        adjacency,
        state["layer1.projection.weight"],
        state["layer1.source_attention.weight"],
        state["layer1.target_attention.weight"],
    )
    hidden = np.maximum(
        _layer_norm(hidden, state["norm1.weight"], state["norm1.bias"]), 0.0
    )
    hidden = _attention(
        hidden,
        adjacency,
        state["layer2.projection.weight"],
        state["layer2.source_attention.weight"],
        state["layer2.target_attention.weight"],
    )
    hidden = np.maximum(
        _layer_norm(hidden, state["norm2.weight"], state["norm2.bias"]), 0.0
    )
    time_hidden = hidden[:, :window_size]
    pooled = np.concatenate([time_hidden.mean(axis=1), time_hidden[:, -1], context], axis=1)
    dense = np.maximum(
        pooled @ state["classifier.0.weight"].T + state["classifier.0.bias"], 0.0
    )
    logits = dense @ state["classifier.3.weight"].T + state["classifier.3.bias"]
    logits = logits + prior_strength * context
    return _softmax(logits)


def analyze_stage(
    times: list[float],
    records: list[dict[str, Any]],
    *,
    model_id: str | None = None,
) -> RiskAnalysis:
    selected = model_id or selected_model_id()
    pressure = [_number(row.get("SGBY")) for row in records]
    rate = [_number(row.get("PL")) for row in records]
    sand = [_number(row.get("SB")) for row in records]
    rule_levels, rule_probability, rule_reasons = _rule_series(times, pressure, rate, sand)
    rule_intervals = _intervals(times, rule_levels, rule_reasons, kind="rule", source="8.4因果砂堵规则")
    if selected == "rule_84":
        return RiskAnalysis(
            rule_probability,
            rule_levels,
            rule_levels,
            rule_intervals,
            [],
            selected,
            "8.4因果砂堵规则",
            "ready",
            "仅使用截至当前点的压力、排量和砂比窗口",
        )

    candidate = "z7_transfer" if selected == "auto" else selected
    try:
        probability, levels, reason = _neural_predictions(candidate, records, times, rule_levels)
        model_name = str(model_catalog()[candidate].get("display_name") or candidate)
        predicted = _intervals(times, levels, [model_name] * len(times), kind="predicted", source=model_name)
        return RiskAnalysis(
            probability,
            levels,
            rule_levels,
            rule_intervals,
            predicted,
            candidate,
            model_name,
            "experimental",
            reason,
        )
    except Exception as exc:
        return RiskAnalysis(
            rule_probability,
            rule_levels,
            rule_levels,
            rule_intervals,
            [],
            "rule_84",
            "8.4因果砂堵规则",
            "fallback",
            f"实验模型不可用，已回退规则：{type(exc).__name__}: {exc}",
        )


__all__ = [
    "RiskAnalysis",
    "analyze_stage",
    "model_catalog",
    "model_options",
    "save_model_selection",
    "selected_model_id",
    "RISK_RUNTIME_VERSION",
]
