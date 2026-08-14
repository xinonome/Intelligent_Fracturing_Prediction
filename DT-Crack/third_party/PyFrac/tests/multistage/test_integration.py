from __future__ import annotations

from pathlib import Path

from multistage.config import load_config
from multistage.multistage_runner import run_project


def test_demo_three_stage_run_and_resume():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "well_x.yaml")
    output = root / "outputs" / "_pytest_demo_run"
    run_dir, metadata, results = run_project(config, output_dir=output)
    assert metadata["status"] == "PASSED"
    assert set(results) == {"stage_1", "stage_2", "stage_3"}
    for stage_id in results:
        assert (run_dir / stage_id / "metrics.csv").is_file()
        assert (run_dir / stage_id / "final_front_local.csv").is_file()
        assert (run_dir / stage_id / "final_front_global.csv").is_file()
    assert (run_dir / "combined" / "all_stages_metrics.csv").is_file()
