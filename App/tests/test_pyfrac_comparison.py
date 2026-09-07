from __future__ import annotations

from App.data.pyfrac_loader import load_pyfrac_comparison
from App.data.registry_loader import RegistryLoader


def test_registered_pyfrac_comparison_contains_only_native_success_points() -> None:
    comparison = load_pyfrac_comparison(RegistryLoader())
    assert comparison.available is True
    assert comparison.engine_mode == "pyfrac_native_dynamic"
    assert [point.time_s for point in comparison.points] == [887.0, 4435.0]
    assert all(point.pyfrac_half_length_m is not None for point in comparison.points)
    assert all(point.pyfrac_max_aperture_mm is not None for point in comparison.points)
