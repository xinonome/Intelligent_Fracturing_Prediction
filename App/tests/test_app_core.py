from __future__ import annotations

import unittest

from App.core.artifacts import ArtifactRegistry, build_preflight
from App.run_app import _build_replay_decision, load_playback_frames


class AppCoreTests(unittest.TestCase):
    def test_registered_statuses_are_explicit(self) -> None:
        modules = ArtifactRegistry().snapshot()["modules"]
        self.assertEqual(modules["fsl"]["status"], "validated")
        self.assertIn(modules["dt"]["status"], {"validated", "development_only"})
        self.assertEqual(modules["dt"]["summary"]["metrics"]["state_dimension"], 4)
        self.assertIn("fiber", modules["dt"]["summary"]["metrics"]["fiber_allocation_source"])
        self.assertEqual(modules["hmi"]["status"], "development_only")

    def test_hmi_quality_gate_is_not_hidden(self) -> None:
        hmi = ArtifactRegistry().module("hmi")
        # A historical replay can pass the internal safety gate while still
        # remaining non-field evidence because the artifact is demo_only.
        self.assertIn(hmi["summary"]["quality_gate"]["passed"], {True, False})
        self.assertNotEqual(hmi["status"], "validated")

    def test_preflight_checks_qt_target_and_real_data(self) -> None:
        preflight = build_preflight()
        self.assertIn("qt_probe", preflight)
        self.assertIn("qt_webengine_probe", preflight)
        self.assertTrue(any(item["path"].endswith("光纤本井监测08.txt") for item in preflight["data"]))

    def test_replay_decision_is_frame_local(self) -> None:
        grow = _build_replay_decision(
            {
                "high_level_option": "grow",
                "abnormal_probability": "0.08",
                "sand_plug_probability": "0.03",
                "posterior_error": "0.05",
            },
            {},
        )
        safe = _build_replay_decision(
            {
                "high_level_option": "safe",
                "abnormal_probability": "0.62",
                "sand_plug_probability": "0.03",
                "posterior_error": "0.05",
                "unsafe": "True",
            },
            {},
        )
        self.assertEqual(grow["risk_level"], "low")
        self.assertEqual(safe["risk_level"], "high")
        self.assertNotEqual(grow["risk_level"], safe["risk_level"])

    def test_joint_replay_keeps_all_hmi_decision_windows(self) -> None:
        frames = load_playback_frames()
        self.assertGreaterEqual(len(frames), 240)
        self.assertEqual(frames[0]["replay_index"], 1)
        self.assertEqual(frames[-1]["replay_index"], frames[-1]["replay_total"])
        options = {frame["hmi_option"] for frame in frames}
        self.assertTrue(options)
        self.assertTrue(options <= {"hold", "grow", "divert", "safe"})
        self.assertGreater(frames[0]["action_flow"], 0.0)
        self.assertGreater(frames[0]["current_flow"], 0.0)


if __name__ == "__main__":
    unittest.main()
