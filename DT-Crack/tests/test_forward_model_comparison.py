from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DT_ROOT = Path(__file__).resolve().parents[1]
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from forward_models import build_length_forward_model  # noqa: E402


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([0.80, 1.00, 1.20, 0.90, 1.10, 1.00]),
        np.arange(6, dtype=float) * 20.0,
        np.full(6, 0.20 / 6.0),
    )


def _simulate(name: str, time_seconds: float = 1800.0):
    factors, positions, rates = _inputs()
    return build_length_forward_model(name).simulate_lengths(
        factors, positions, rates, 0.10, 3.2e10, 30.0, time_seconds
    ).table


def test_all_models_preserve_total_rate_and_return_finite_geometry() -> None:
    for name in ["pkn4", "bem_reduced", "data_surrogate"]:
        table = _simulate(name)
        assert np.isclose(table["Q_cluster_m3s"].sum(), 0.20, rtol=1.0e-8)
        assert np.isfinite(table[["half_length_m", "max_aperture_mm", "volume_m3"]]).all().all()
        assert (table[["half_length_m", "max_aperture_mm", "volume_m3"]] > 0.0).all().all()


def test_panel_bem_has_time_monotonic_half_length() -> None:
    early = _simulate("bem_reduced", 600.0)["half_length_m"].to_numpy()
    late = _simulate("bem_reduced", 1800.0)["half_length_m"].to_numpy()
    assert np.all(late >= early)


def test_surrogate_tracks_panel_bem_on_nominal_case() -> None:
    teacher = _simulate("bem_reduced")["half_length_m"].to_numpy()
    proxy = _simulate("data_surrogate")["half_length_m"].to_numpy()
    relative_error = np.abs(proxy - teacher) / teacher
    assert float(np.max(relative_error)) < 0.03

