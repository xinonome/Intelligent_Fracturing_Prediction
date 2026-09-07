from __future__ import annotations

import unittest

from App.core.artifacts import ArtifactRegistry, build_preflight
from App.data.hmi_loader import HMILoader
from App.data.registry_loader import RegistryLoader
from App.run_app import _build_replay_decision, load_playback_frames
from App.ui.main_window import _edition_metadata


class AppCoreTests(unittest.TestCase):
    def test_software_copyright_editions_have_distinct_visible_boundaries(self) -> None:
        integrated = _edition_metadata("integrated")
        fsl = _edition_metadata("fsl")
        dt_hmi = _edition_metadata("dt_hmi")
        self.assertEqual(
            [page[0] for page in integrated["pages"]],
            ["fsl", "dt", "hmi", "resources"],
        )
        self.assertEqual(
            [page[2] for page in integrated["pages"]],
            ["工况与风险", "裂缝与参数", "智能调控", "资源中心"],
        )
        self.assertEqual([page[0] for page in fsl["pages"]], ["fsl"])
        self.assertEqual([page[0] for page in dt_hmi["pages"]], ["dt", "hmi", "integrated"])
        self.assertNotEqual(fsl["window_title"], dt_hmi["window_title"])

    def test_registered_statuses_are_explicit(self) -> None:
        modules = ArtifactRegistry().snapshot()["modules"]
        self.assertEqual(modules["fsl"]["status"], "validated")
        self.assertIn(modules["dt"]["status"], {"validated", "development_only"})
        self.assertEqual(modules["dt"]["summary"]["metrics"]["state_dimension"], 14)
        self.assertIn("observation_target_only", modules["dt"]["summary"]["metrics"]["fiber_allocation_source"])
        self.assertTrue(modules["dt"]["summary"]["metrics"]["parameterized_allocation"])
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

    def test_hmi_sand_current_and_recommendation_are_independent(self) -> None:
        loader = HMILoader(RegistryLoader())
        pairs = [
            (
                float(row["current_sand_ratio_percent"]),
                float(row["recommended_sand_ratio_percent"]),
            )
            for row in loader.rows
            if row.get("current_sand_ratio_percent") not in {None, ""}
            and row.get("recommended_sand_ratio_percent") not in {None, ""}
        ]
        self.assertTrue(pairs)
        self.assertTrue(any(abs(current - recommendation) > 1.0e-4 for current, recommendation in pairs))


if __name__ == "__main__":
    unittest.main()
