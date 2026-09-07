from App.data.pyfrac_runtime_loader import load_pyfrac_runtime


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
