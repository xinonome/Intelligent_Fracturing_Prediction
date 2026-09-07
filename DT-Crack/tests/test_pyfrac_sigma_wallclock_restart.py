from __future__ import annotations

import unittest

from inversion.run_pyfrac_sigma_wallclock_restart import (
    _failure_reason,
    _next_failure_candidate,
)


class TestPyFracSigmaWallClockFailureRecovery(unittest.TestCase):
    def test_timeout_is_classified_as_incomplete_not_solver_failure(self):
        reason = _failure_reason(
            {
                "error": "candidate worker timeout after 120s; fresh run discarded",
                "last_progress": {
                    "time_s": 23.2,
                    "final_time_s": 2503.0,
                    "failed_time_steps": 0,
                },
            }
        )
        self.assertEqual(reason, "timeout_before_target")

    def test_native_failed_step_is_classified(self):
        reason = _failure_reason(
            {
                "error": "native result rejected",
                "last_progress": {"failed_time_steps": 1},
            }
        )
        self.assertEqual(reason, "native_failed_time_step")

    def test_slow_progress_is_classified_by_simulated_to_wall_clock_ratio(self):
        reason = _failure_reason(
            {
                "error": (
                    "candidate worker watchdog terminated slow progress; "
                    "sim_to_wall_ratio=2.8; required>=3"
                )
            }
        )
        self.assertEqual(reason, "sim_to_wall_ratio_below_threshold")

    def test_zero_candidate_timeout_means_no_per_candidate_cap(self):
        # This is a contract test for the CLI default: long but sufficiently
        # fast candidates must not be rejected at an arbitrary 120 s boundary.
        from inversion.run_pyfrac_sigma_wallclock_restart import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.candidate_timeout_s, 0.0)

    def test_failure_candidate_contracts_last_update_direction(self):
        anchor = 78.85599999999999
        delta = 4.16801709665505
        current = anchor + delta
        next_sigma = _next_failure_candidate(
            current,
            anchor_sigma_mpa=anchor,
            update_delta_mpa=delta,
            failure_streak=1,
            lower_mpa=20.0,
            upper_mpa=120.0,
            contraction=0.8,
            fallback_step_mpa=5.0,
        )
        self.assertAlmostEqual(next_sigma, anchor + 0.8 * delta, places=8)

        second_sigma = _next_failure_candidate(
            next_sigma,
            anchor_sigma_mpa=anchor,
            update_delta_mpa=delta,
            failure_streak=2,
            lower_mpa=20.0,
            upper_mpa=120.0,
            contraction=0.8,
            fallback_step_mpa=5.0,
        )
        self.assertAlmostEqual(second_sigma, anchor + 0.8**2 * delta, places=8)

    def test_failure_without_previous_update_uses_bounded_fallback(self):
        current = 60.0
        next_sigma = _next_failure_candidate(
            current,
            anchor_sigma_mpa=None,
            update_delta_mpa=None,
            failure_streak=1,
            lower_mpa=20.0,
            upper_mpa=120.0,
            contraction=0.8,
            fallback_step_mpa=5.0,
        )
        self.assertEqual(next_sigma, 65.0)


if __name__ == "__main__":
    unittest.main()
