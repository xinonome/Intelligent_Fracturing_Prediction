from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DT_ROOT = Path(__file__).resolve().parents[1]
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from forward_models import PyFracAdapter, PyFracConfig, build_length_forward_model  # noqa: E402
from inversion import PhysicalEnKFConfig, physical_values, pkn_with_carter_leakoff  # noqa: E402


def test_vendored_pyfrac_metadata_and_license_are_present() -> None:
    adapter = PyFracAdapter(PyFracConfig(), project_root=DT_ROOT.parent)
    metadata = adapter.verify_installation()
    assert metadata["source_exists"] is True
    assert metadata["license_exists"] is True
    assert metadata["gpl_exists"] is True
    assert (DT_ROOT / "third_party" / "PyFrac" / "PROJECT_METADATA.json").is_file()


def test_new_five_state_uses_kic_but_legacy_cluster_state_stays_compatible() -> None:
    cfg = PhysicalEnKFConfig()
    legacy = physical_values(np.r_[0.0, 0.0, 0.0, 60.0, 1.30, 0.70, 1, 1, 1, 1], cfg, 6)
    assert np.isclose(np.asarray(legacy["cluster_factors"])[0], 1.30)
    assert np.isclose(float(legacy["fracture_toughness_pa_sqrt_m"]), cfg.base_fracture_toughness_pa_sqrt_m)

    new_state = np.asarray([0.0, 0.0, 0.0, 60.0, np.log(2.0)])
    updated = physical_values(new_state, cfg, 6)
    assert np.isclose(float(updated["fracture_toughness_pa_sqrt_m"]), 2.0 * cfg.base_fracture_toughness_pa_sqrt_m)
    assert np.allclose(updated["cluster_factors"], np.ones(6))


def test_kic_update_changes_pkn_only_after_forward_recomputation() -> None:
    cfg = PhysicalEnKFConfig()
    q = np.full(6, 0.2 / 6.0)
    low_kic = pkn_with_carter_leakoff(np.asarray([0.0, 0.0, 0.0, 60.0, np.log(0.5)]), q, 1800.0, cfg)
    high_kic = pkn_with_carter_leakoff(np.asarray([0.0, 0.0, 0.0, 60.0, np.log(2.0)]), q, 1800.0, cfg)
    assert np.all(high_kic["half_length_m"] < low_kic["half_length_m"])
    assert np.all(np.isfinite(high_kic["half_length_m"]))


def test_pyfrac_alias_is_constructible_without_running_native_solver() -> None:
    model = build_length_forward_model("pyfrac")
    assert model.model_name == "pyfrac_reference"
