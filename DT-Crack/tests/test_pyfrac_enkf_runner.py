from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import pandas as pd

from forward_models.pyfrac_adapter import (
    PyFracAdapter,
    PyFracNativeSession,
    native_state_health_errors,
    repair_native_front_metadata,
    target_time_reached,
)
from forward_models.pyfrac_config import PyFracConfig
from forward_models.pyfrac_robustness import CheckpointManager
from inversion.run_pyfrac_enkf import derive_injection_rate, finite_or_none, make_schedule
from inversion.run_pyfrac_enkf_realtime import _load_latest_checkpoint
from inversion.run_pyfrac_enkf_synchronized import anchored_shadow_pressures, interpolate_native_parameters


def test_derive_injection_rate_preserves_measured_shut_in_after_first_sample() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "cumulative_liquid_m3": [0.0, 0.05, 0.10, 0.30],
            "flow_rate_m3_s": [0.0, 0.0, 0.20, 0.0],
        }
    )

    rate = derive_injection_rate(frame)

    # The leading zero-rate rows have no measured injection and use the
    # cumulative-volume fallback.  The explicit zero after the first valid
    # pump-rate sample is a true shut-in and must remain zero.
    np.testing.assert_allclose(rate, [0.05, 0.05, 0.20, 0.0])


def test_make_schedule_covers_target_and_is_non_negative() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "cumulative_liquid_m3": [0.0, 1.0, 2.0],
        }
    )
    rate = np.asarray([1.0, 1.0, 1.0])

    schedule = make_schedule(frame, rate, target_time_s=5.0, step_s=2.0)

    assert schedule.shape == (2, 4)
    assert schedule[0, 0] == 0.0
    assert schedule[0, -1] == 5.0
    assert np.all(np.diff(schedule[0]) > 0.0)
    assert np.all(schedule[1] >= 0.0)


def test_make_schedule_keeps_rate_change_events() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "cumulative_liquid_m3": [0.0, 1.0, 2.0, 2.0, 2.0],
            "flow_rate_m3_s": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    rate = derive_injection_rate(frame)
    schedule = make_schedule(frame, rate, target_time_s=4.0, step_s=3.0)

    # The compact regular grid does not contain t=2 or t=4 by itself; the
    # event-aware schedule must retain the shut-in transition exactly.
    assert 2.0 in schedule[0]
    assert schedule[1, np.where(schedule[0] == 2.0)[0][0]] == 0.0
    assert schedule[1, np.where(schedule[0] == 4.0)[0][0]] > 0.0


def test_make_schedule_does_not_turn_small_rate_jitter_into_events() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cumulative_liquid_m3": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "flow_rate_m3_s": [0.05, 0.051, 0.049, 0.05, 0.10, 0.10],
        }
    )
    rate = derive_injection_rate(frame)
    schedule = make_schedule(frame, rate, target_time_s=5.0, step_s=3.0)

    # t=1..3 are small sensor fluctuations; the material change at t=4 is
    # retained, even though the compact regular grid does not contain it.
    assert 1.0 not in schedule[0]
    assert 2.0 not in schedule[0]
    assert 3.0 not in schedule[0]
    assert 4.0 in schedule[0]


def test_finite_or_none_keeps_json_safe_numbers() -> None:
    assert finite_or_none(1.25) == 1.25
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none("not-a-number") is None


def test_native_continuation_disables_unbounded_mesh_extension() -> None:
    class DummySimulationProperties:
        pass

    session = object.__new__(PyFracNativeSession)
    session.modules = {"SimulationProperties": DummySimulationProperties}
    session.config = PyFracConfig(adaptive_mesh_enabled=True, enable_pyfrac_remeshing=True)
    session.max_time_steps = 5

    simulation = session._simulation(600.0, 15.0)

    assert simulation.enableRemeshing is True
    assert simulation.meshExtension == [False, False, False, False]
    assert simulation.meshExtensionAllDir is False
    assert simulation.enableGPU is False


def test_resume_rejects_oversized_dense_mesh_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(PyFracAdapter, "_load_modules", lambda self: {})
    safe = tmp_path / "accepted_safe.dill"
    oversized = tmp_path / "accepted_oversized.dill"
    for path, elements, time_s in ((safe, 4617, 680.0), (oversized, 14013, 708.0)):
        payload = {
            "time_s": time_s,
            "fracture": SimpleNamespace(mesh=SimpleNamespace(NumberOfElts=elements)),
        }
        with path.open("wb") as handle:
            dill.dump(payload, handle)
    # Ensure the unsafe checkpoint is the newest candidate.
    oversized.touch()

    loaded = _load_latest_checkpoint(tmp_path, max_mesh_elements=6000)

    assert loaded is not None
    assert loaded["checkpoint_path"].endswith("accepted_safe.dill")
    assert loaded["checkpoint_mesh_elements"] == 4617


def test_native_failure_keeps_member_recoverable_and_persists_reduced_step() -> None:
    class DummyController:
        def __init__(self, fracture, *_args):
            self.fracture = fracture

        def run(self) -> None:
            raise RuntimeError("synthetic continuation failure")

    session = object.__new__(PyFracNativeSession)
    session.config = PyFracConfig(
        dynamic_step_limit_s=30.0,
        min_dynamic_step_s=0.5,
        retry_time_step_factor=0.5,
        max_retries=0,
        checkpoint_enabled=True,
    )
    session.modules = {
        "FluidProperties": lambda **_kwargs: object(),
        "InjectionProperties": lambda *_args: object(),
        "Controller": DummyController,
    }
    session.mesh = SimpleNamespace()
    session.fracture = SimpleNamespace(
        time=360.0,
        pFluid=np.zeros(1),
        pNet=np.zeros(1),
        EltCrack=np.asarray([0]),
    )
    session.active = True
    session.last_parameters = {}
    session.checkpoints = CheckpointManager()
    session.checkpoint_dir = None
    session.last_checkpoint_id = None
    session.last_rollback_applied = False
    session.last_retry_count = 0
    session.total_successful_steps = 0
    session.total_failed_steps = 0
    session.mesh_decision = None
    session._continuation_step_limit_s = 30.0
    session.consecutive_failures = 0
    session._material = lambda *_args: SimpleNamespace(SigmaO=np.zeros(1))
    seen_limits: list[float] = []
    session._simulation = lambda _target, limit: seen_limits.append(float(limit)) or object()

    kwargs = {
        "height_m": 30.0,
        "viscosity_pa_s": 0.1,
        "e_prime_pa": 3.2e10,
        "leakoff_coefficient_m_sqrt_s": 1.0e-5,
        "min_horizontal_stress_pa": 60.0e6,
        "fracture_toughness_pa_sqrt_m": 1.0e6,
    }
    schedule = np.asarray([[0.0, 600.0], [0.1, 0.1]])

    first = session.advance_to(600.0, schedule, **kwargs)
    second = session.advance_to(600.0, schedule, **kwargs)

    assert not first.success and not second.success
    assert session.active is True
    assert session.consecutive_failures == 2
    assert seen_limits == [30.0, 15.0]
    assert session._continuation_step_limit_s == 7.5


def test_native_state_health_gate_rejects_corrupt_front_arrival_and_collapse() -> None:
    previous = SimpleNamespace(EltCrack=np.arange(100))
    fracture = SimpleNamespace(
        EltCrack=np.arange(20),
        EltChannel=np.asarray([0, 1]),
        EltTip=np.asarray([2]),
        EltRibbon=np.asarray([3]),
        Tarrival=np.asarray([0.0, np.nan, 0.0, 0.0] + [np.nan] * 16),
        TarrvlZrVrtx=np.asarray([0.0, 0.0, 1.0, np.nan] + [np.nan] * 16),
        w=np.ones(20),
        pNet=np.ones(20),
        pFluid=np.ones(20),
        v=np.asarray([-0.01]),
    )

    errors = native_state_health_errors(fracture, previous)

    assert "non-finite Tarrival on EltChannel" in errors
    assert "non-finite TarrvlZrVrtx on EltRibbon" in errors
    assert "negative or non-finite front velocity" in errors
    assert any("footprint collapsed" in error for error in errors)


def test_native_step_limit_recovers_after_three_healthy_windows() -> None:
    session = object.__new__(PyFracNativeSession)
    session.config = PyFracConfig(
        dynamic_step_limit_s=5.0,
        min_dynamic_step_s=0.5,
        retry_time_step_factor=0.5,
    )
    session.consecutive_failures = 2
    session._continuation_success_streak = 0
    session._continuation_step_limit_s = 0.625

    for _ in range(3):
        session._record_successful_continuation(0.625)

    assert session.consecutive_failures == 0
    assert session._continuation_success_streak == 0
    assert session._continuation_step_limit_s == 1.25


def test_native_session_can_backtrack_one_accepted_generation() -> None:
    session = object.__new__(PyFracNativeSession)
    session.config = PyFracConfig(
        dynamic_step_limit_s=5.0,
        min_dynamic_step_s=0.5,
        retry_time_step_factor=0.5,
    )
    session.checkpoints = CheckpointManager()
    old = SimpleNamespace(time=500.0, mesh=SimpleNamespace())
    current = SimpleNamespace(time=550.0, mesh=SimpleNamespace())
    old_checkpoint = session.checkpoints.save(
        time_s=500.0,
        fracture=old,
        last_parameters={"e_prime_pa": 3.2e10},
        successful_steps=20,
        failed_steps=1,
        active=True,
    )
    current_checkpoint = session.checkpoints.save(
        time_s=550.0,
        fracture=current,
        last_parameters={"e_prime_pa": 3.2e10},
        successful_steps=40,
        failed_steps=2,
        active=True,
    )
    session._accepted_state_history = [old_checkpoint, current_checkpoint]
    session.fracture = current
    session.mesh = current.mesh
    session.last_parameters = {"e_prime_pa": 3.2e10}
    session.total_successful_steps = 40
    session.total_failed_steps = 2
    session.active = True
    session.last_rollback_applied = False
    session._continuation_step_limit_s = 1.0
    session.consecutive_failures = 3
    session._continuation_success_streak = 0
    session.checkpoint_dir = None

    restored = session.rollback_to_previous_accepted(reason="test backtrack")

    assert restored is True
    assert session.fracture.time == 500.0
    assert session.active is True
    assert session._continuation_step_limit_s == 0.5


def test_front_metadata_repair_does_not_modify_pressure_width_or_volume_fields() -> None:
    fracture = SimpleNamespace(
        time=600.0,
        EltChannel=np.asarray([0, 1]),
        EltTip=np.asarray([2]),
        EltRibbon=np.asarray([3]),
        Tarrival=np.asarray([10.0, np.nan, np.nan, np.nan]),
        TarrvlZrVrtx=np.asarray([np.nan, np.nan, np.nan, np.nan]),
        v=np.asarray([-0.2, np.nan]),
        ZeroVertex=np.asarray([1, 2]),
        Ffront=np.ones((2, 4)),
        w=np.asarray([1.0, 2.0]),
        pNet=np.asarray([3.0, 4.0]),
        FractureVolume=5.0,
    )
    width = fracture.w.copy()
    pressure = fracture.pNet.copy()
    volume = fracture.FractureVolume

    repairs = repair_native_front_metadata(fracture)

    assert repairs
    assert np.isfinite(fracture.Tarrival[fracture.EltChannel]).all()
    assert np.isfinite(fracture.TarrvlZrVrtx[np.r_[fracture.EltTip, fracture.EltRibbon]]).all()
    assert (fracture.v >= 0.0).all()
    assert fracture.ZeroVertex.shape == (1,)
    assert fracture.Ffront.shape == (1, 4)
    np.testing.assert_array_equal(fracture.w, width)
    np.testing.assert_array_equal(fracture.pNet, pressure)
    assert fracture.FractureVolume == volume


def test_shadow_pressures_keep_native_anchor_and_pkn_parameter_deltas(monkeypatch) -> None:
    ensemble = np.asarray(
        [
            [0.0, 0.0, 0.0, 60.0, 0.0],
            [0.1, 0.0, 0.0, 60.0, 0.0],
            [-0.1, 0.0, 0.0, 60.0, 0.0],
        ]
    )
    prior = ensemble.mean(axis=0)

    def fake_pkn(state, *_args, **_kwargs):
        return {"bottomhole_pressure_mpa": 80.0 + 10.0 * float(state[0])}

    monkeypatch.setattr("inversion.run_pyfrac_enkf_synchronized.pkn_pressure", fake_pkn)
    predicted = anchored_shadow_pressures(
        ensemble,
        prior,
        105.0,
        SimpleNamespace(),
        np.asarray([[0.0, 1.0], [0.1, 0.1]]),
        1.0,
    )

    np.testing.assert_allclose(predicted, [105.0, 106.0, 104.0])
    assert np.mean(predicted) == 105.0


def test_target_time_reached_uses_numerical_not_percentage_tolerance() -> None:
    assert target_time_reached(4435.0, 4435.0)
    assert target_time_reached(4434.999, 4435.0)
    assert not target_time_reached(4430.565, 4435.0)
    assert not target_time_reached(569.5, 570.0)


def test_native_parameter_homotopy_reaches_exact_posterior() -> None:
    start = {"e_prime_pa": 1.0e10, "min_horizontal_stress_pa": 50.0e6, "height_m": 30.0}
    target = {"e_prime_pa": 4.0e10, "min_horizontal_stress_pa": 70.0e6, "height_m": 30.0}

    halfway = interpolate_native_parameters(start, target, 0.5)
    final = interpolate_native_parameters(start, target, 1.0)

    assert np.isclose(halfway["e_prime_pa"], 2.0e10)
    assert halfway["min_horizontal_stress_pa"] == 60.0e6
    assert final == target
