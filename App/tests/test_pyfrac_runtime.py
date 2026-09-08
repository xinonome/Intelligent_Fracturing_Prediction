from App.data.pyfrac_runtime_loader import load_pyfrac_runtime
from pathlib import Path
import json


def test_reference_pyfrac_runtime_contains_real_adaptive_points() -> None:
    runtime = load_pyfrac_runtime()
    assert runtime.available
    assert runtime.point_count == 370
    assert runtime.frames[-1].time_s == 4435.0
    assert runtime.frames[-1].successful_step == 370
    assert runtime.frames[-1].half_length_m is not None
    assert runtime.frames[-1].max_aperture_mm is not None


def test_reference_pyfrac_times_are_strictly_increasing() -> None:
    runtime = load_pyfrac_runtime()
    times = [frame.time_s for frame in runtime.frames]
    assert times == sorted(set(times))


def test_completed_app_run_keeps_real_front_and_pressure_field() -> None:
    run = Path("outputs/app/pyfrac_runs/20260903_160239")
    runtime = load_pyfrac_runtime(run)
    assert runtime.available
    assert runtime.completed
    assert runtime.frames[-1].front_geometry
    assert runtime.frames[-1].pressure_field


def test_short_completed_run_uses_real_final_state_without_inventing_history(tmp_path) -> None:
    (tmp_path / "spec.json").write_text(json.dumps({"target_time_s": 2.0}), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps({"result": {
        "success": True,
        "target_reached": True,
        "final_time_s": 2.0,
        "successful_time_steps": 2,
        "half_length_m": 2.5,
        "max_aperture_mm": 1.2,
        "front_geometry": [[-2.5, 0.0], [2.5, 0.0]],
    }}), encoding="utf-8")
    runtime = load_pyfrac_runtime(tmp_path)
    assert runtime.available and runtime.completed
    assert runtime.point_count == 1
    assert runtime.frames[0].time_s == 2.0
    assert runtime.frames[0].half_length_m == 2.5
    assert "中间点" in runtime.note
