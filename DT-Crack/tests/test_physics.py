from __future__ import annotations

import unittest

import numpy as np

from inversion import PhysicalEnKFConfig, denkf_update, pkn_with_carter_leakoff


class EnhancedPKNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PhysicalEnKFConfig()
        self.q = np.full(6, 0.2 / 6.0)
        self.state = np.r_[0.0, 0.0, 0.0, 60.0, np.ones(6)]

    def test_material_balance_and_leakoff_bound(self) -> None:
        result = pkn_with_carter_leakoff(self.state, self.q, 1800.0, self.cfg)
        self.assertLess(result["rate_conservation_error"], 1e-10)
        self.assertGreaterEqual(result["leakoff_fraction"], 0.0)
        self.assertLessEqual(result["leakoff_fraction"], self.cfg.max_leakoff_fraction + 1e-12)

    def test_outputs_are_finite_and_positive(self) -> None:
        result = pkn_with_carter_leakoff(self.state, self.q, 1800.0, self.cfg)
        self.assertTrue(np.all(np.isfinite(result["half_length_m"])))
        self.assertTrue(np.all(np.asarray(result["half_length_m"]) > 0.0))

    def test_intake_factor_changes_relative_growth(self) -> None:
        state = self.state.copy()
        state[4] = 1.30
        state[5] = 0.70
        result = pkn_with_carter_leakoff(state, self.q, 1800.0, self.cfg)
        self.assertGreater(result["half_length_m"][0], result["half_length_m"][1])

    def test_current_rate_updates_aperture_without_breaking_volume_balance(self) -> None:
        current_rate = np.full(6, 0.5 / 6.0)
        result = pkn_with_carter_leakoff(self.state, self.q, 1800.0, self.cfg, current_rate)
        self.assertAlmostEqual(float(np.asarray(result["q_current_nominal_m3_s"]).sum()), 0.5, places=10)
        self.assertTrue(np.all(np.asarray(result["max_aperture_mm"]) > 0.0))

    def test_deterministic_enkf_reduces_mean_observation_residual(self) -> None:
        rng = np.random.default_rng(7)
        ensemble = rng.normal(0.0, 1.0, size=(80, 2))
        predicted = ensemble[:, :1] + 0.2 * ensemble[:, 1:2]
        observed = np.asarray([1.5])
        before = abs(float(predicted.mean()) - float(observed[0]))
        updated, gain = denkf_update(ensemble, predicted, observed, np.asarray([0.15]))
        updated_prediction = updated[:, :1] + 0.2 * updated[:, 1:2]
        after = abs(float(updated_prediction.mean()) - float(observed[0]))
        self.assertEqual(gain.shape, (2, 1))
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
