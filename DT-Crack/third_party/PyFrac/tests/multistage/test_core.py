from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multistage.diagnostics import barrier_contacts, geometry_metrics, handover_diagnostic
from multistage.exceptions import ConfigurationError, DataValidationError, GeometryError
from multistage.global_mapping import basis_at_stage, global_to_local, local_to_global
from multistage.pyfrac_adapter import build_injection_rate_history, estimate_memory_gb
from multistage.property_mapping import build_stage_material, local_stress_function
from multistage.schemas import StageDefinition, StageSnapshot
from multistage.stage_definition import slice_stage_logs
from multistage.validation import validate_config_values, validate_logs, validate_trajectory


def _frames():
    logs = pd.DataFrame({
        "MD_m": [0., 10., 20., 30.], "GR_API": [1.] * 4,
        "poisson_ratio": [.25] * 4, "young_modulus_Pa": [10e9] * 4,
        "sigma_hmin_Pa": [10e6, 20e6, 30e6, 40e6],
        "KIC_Pa_sqrt_m": [1.] * 4, "carter_m_sqrt_s": [1e-6] * 4,
    })
    trajectory = pd.DataFrame({"MD_m": [0., 10., 20., 30.], "X_m": [0., 10., 20., 30.], "Y_m": [0.] * 4, "TVD_m": [100., 110., 120., 130.]})
    return logs, trajectory


def test_strict_input_contract_and_inclusive_slice():
    logs, trajectory = _frames(); stage = StageDefinition("s", 10., 20.)
    assert slice_stage_logs(logs, stage)["MD_m"].tolist() == [10., 20.]
    assert validate_logs(logs, (stage,)) == []
    validate_trajectory(trajectory, (stage,))
    bad = logs.copy(); bad.loc[2, "poisson_ratio"] = .5
    with pytest.raises(DataValidationError): validate_logs(bad, (stage,))


def test_property_mapping_uses_eprime_median_and_no_extrapolation():
    logs, trajectory = _frames(); stage = StageDefinition("s", 10., 20.)
    class C:
        material = {"e_prime_warning_relative_spread": .2, "toughness": {"mode": "log"}, "leakoff": {"mode": "log"}}
    material = build_stage_material(logs, trajectory, stage, C())
    assert material.e_prime_pa == pytest.approx(10e9 / (1 - .25 ** 2))
    stress = local_stress_function(logs, stage, trajectory)
    assert stress(0., 0.) == pytest.approx(25e6)
    with pytest.raises(DataValidationError): stress(0., 1000.)


def test_mapping_round_trip_and_vertical_guard():
    _, trajectory = _frames(); stage = StageDefinition("s", 10., 20.)
    local = pd.DataFrame({"stage_id": ["s", "s"], "time_s": [1., 1.], "u_m": [2., -3.], "v_m": [4., -5.]})
    mapped = local_to_global(local, trajectory, stage)
    center, e_frac, e_up = basis_at_stage(trajectory, stage)
    recovered = global_to_local(mapped[["X_m", "Y_m", "Z_m"]].to_numpy(), center, e_frac, e_up)
    assert recovered == pytest.approx(local[["u_m", "v_m"]].to_numpy())
    vertical = trajectory.copy(); vertical["X_m"] = 0.; vertical["TVD_m"] = [100., 110., 120., 130.]
    with pytest.raises(GeometryError): basis_at_stage(vertical, stage)


def _snapshots():
    rows = []
    for t, length, height in ((1., 1., 1.), (2., 2., 2.), (3., 3., 2.), (4., 4., 2.)):
        front = [[-length, -height], [length, height]]
        rows.append(StageSnapshot("s", t, [], [], .1, 1., .5, 1., front, .01, 1., .01))
    return rows


def test_handover_requires_contact_and_consecutive_evidence():
    snapshots = _snapshots(); metrics = geometry_metrics(snapshots, "s")
    contacts = barrier_contacts(metrics, snapshots, 2., -2., 0.1)
    evidence, handover = handover_diagnostic(metrics, contacts, 2, .1, .1)
    assert handover == pytest.approx(3.)
    no_contact = barrier_contacts(metrics, snapshots, 20., -20., .1)
    _, no_handover = handover_diagnostic(metrics, no_contact, 2, .1, .1)
    assert no_handover is None


def test_schedule_and_memory_guard_values_are_explicit():
    schedule = build_injection_rate_history(.001, 30.)
    assert schedule.shape == (2, 2)
    assert schedule.tolist() == [[0., 30.], [.001, 0.]]
    assert estimate_memory_gb(41, 41) > 0
    with pytest.raises(ConfigurationError): validate_config_values({"project": {}})
