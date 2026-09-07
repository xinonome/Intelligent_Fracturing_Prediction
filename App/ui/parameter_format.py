"""Engineering display names for physical and allocation parameters.

The data contract deliberately keeps stable machine-readable keys such as
``E_prime_gpa``.  This module is the presentation boundary: the APP shows
symbols, units and subscripts without changing the stored schema or the
calculation code.
"""

from __future__ import annotations

from html import escape
from typing import Any


_PARAMETER_LABELS: dict[str, tuple[str, str]] = {
    "E_prime_gpa": ("<i>E</i><sup>&prime;</sup>", "GPa"),
    "C_L_m_sqrt_s": ("<i>C</i><sub>L</sub>", "m&middot;s<sup>-1/2</sup>"),
    "mu_pa_s": ("&mu;", "Pa&middot;s"),
    "sigma_min_mpa": ("&sigma;<sub>min</sub>", "MPa"),
    "K_IC_pa_sqrt_m": ("<i>K</i><sub>IC</sub>", "Pa&middot;m<sup>1/2</sup>"),
    # These parameters are dimensionless.  The APP does not append the
    # literal unit ``1`` because it is visually easy to mistake it for an
    # additional value in a dense parameter line.
    "stress_shadow_scale": ("&gamma;<sub>sh</sub>", ""),
    "boundary_relief_scale": ("&gamma;<sub>e</sub>", ""),
    "allocation_exponent": ("&alpha;<sub>a</sub>", ""),
}


def _parameter_label(key: str) -> tuple[str, str]:
    if key in _PARAMETER_LABELS:
        return _PARAMETER_LABELS[key]
    if key.startswith("intake_capacity_C"):
        cluster = key.removeprefix("intake_capacity_C")
        return f"&kappa;<sub>C{escape(cluster)}</sub>", ""
    return escape(key), ""


def _format_value(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def format_parameter_map(values: dict[str, Any] | None) -> str:
    """Return one copyable engineering-formula item per line.

    The value is rendered by Qt ``QLabel`` as rich text.  ``<br>`` keeps the
    engineering symbols/subscripts while giving each parameter its own line,
    which is substantially easier to scan than a semicolon-separated string.
    """

    if not values:
        return "缺失 · 未接入"
    formatted = []
    for key, value in values.items():
        symbol, unit = _parameter_label(str(key))
        suffix = f" {unit}" if unit else ""
        formatted.append(f"{symbol} = {_format_value(value)}{suffix}")
    return "<br>".join(formatted)


__all__ = ["format_parameter_map"]
