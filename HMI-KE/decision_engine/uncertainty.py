from __future__ import annotations


def probability_uncertainty(max_probability: float) -> str:
    if max_probability >= 0.75:
        return "low"
    if max_probability >= 0.45:
        return "medium"
    return "high"


def residual_uncertainty(abs_residual: float, low_threshold: float = 3.0, high_threshold: float = 8.0) -> str:
    if abs_residual <= low_threshold:
        return "low"
    if abs_residual <= high_threshold:
        return "medium"
    return "high"


def combine_uncertainty(*levels: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    reverse = {0: "low", 1: "medium", 2: "high"}
    return reverse[max(order.get(level, 1) for level in levels)]
