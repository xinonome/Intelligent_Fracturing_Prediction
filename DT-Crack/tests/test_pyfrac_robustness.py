from __future__ import annotations

import unittest

import numpy as np

from forward_models.pyfrac_robustness import (
    CheckpointManager,
    choose_mesh,
    evaluate_convergence,
    relaxed_update,
)


class TestPyFracRobustness(unittest.TestCase):
    def test_mesh_refines_when_front_is_under_resolved(self):
        decision = choose_mesh(
            estimated_half_length_m=70.0,
            height_m=30.0,
            base_half_length_m=80.0,
            base_half_height_m=45.0,
            base_nx=31,
            base_ny=21,
            min_front_cells=20,
            boundary_margin_cells=6,
            max_levels=3,
        )
        self.assertGreaterEqual(decision.level, 1)
        self.assertGreaterEqual(decision.cells_across_front, 20.0)
        self.assertGreaterEqual(decision.boundary_margin_m / decision.dx_m, 6.0)

    def test_relaxed_update_limits_parameter_jump(self):
        result = relaxed_update(
            np.zeros((2, 3)),
            np.ones((2, 3)),
            relaxation=0.5,
            max_step=np.asarray([0.2, 0.3, 0.4]),
        )
        np.testing.assert_allclose(result.state, [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]])
        self.assertEqual(result.clipped_components, 6)
        self.assertAlmostEqual(result.raw_update_norm, np.sqrt(6.0))

    def test_checkpoint_restores_mutated_state(self):
        manager = CheckpointManager()
        state = {"array": np.asarray([1.0, 2.0])}
        checkpoint = manager.save(
            time_s=120.0,
            fracture=state,
            last_parameters={"stress": 60.0},
            successful_steps=4,
            failed_steps=0,
            active=True,
        )
        state["array"][0] = 99.0
        restored = manager.restore(checkpoint, reason="solver failure", retry_count=1)
        np.testing.assert_allclose(restored["array"], [1.0, 2.0])
        self.assertEqual(manager.events[-1]["event"], "restored")

    def test_convergence_rejects_failed_or_nonconservative_run(self):
        records = [
            {"name": "medium", "success": True, "target_reached": True, "failed_time_steps": 0, "mass_balance_relative_error": 0.02, "half_length_m": 100.0},
            {"name": "fine", "success": True, "target_reached": True, "failed_time_steps": 0, "mass_balance_relative_error": 0.03, "half_length_m": 103.0},
        ]
        report = evaluate_convergence(records, reference_name="fine", comparison_name="medium", metrics=("half_length_m",))
        self.assertTrue(report["passed"])
        records[0]["mass_balance_relative_error"] = 0.11
        report = evaluate_convergence(records, reference_name="fine", comparison_name="medium", metrics=("half_length_m",))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
