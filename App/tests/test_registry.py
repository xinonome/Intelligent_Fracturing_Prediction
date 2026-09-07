from __future__ import annotations

import json
import unittest

from App.core.replay import build_replay_frames
from App.data.dt_loader import DTLoader
from App.data.hmi_loader import HMILoader
from App.data.registry_loader import RegistryLoader


class RegistryTests(unittest.TestCase):
    def test_dt_registers_one_frame_source_and_one_html(self):
        loader = RegistryLoader()
        dt = loader.module("dt")
        self.assertEqual(dt["frame_count"], 7331)
        self.assertTrue(loader.frame_source().exists())
        self.assertEqual(loader.html().name, "dt_realtime_3d.html")
        self.assertEqual(
            loader.html("no_das_pressure_only").name,
            "dt_realtime_3d_no_das.html",
        )
        self.assertTrue(loader.html("no_das_pressure_only").exists())

    def test_fsl_risk_summary_is_registered_and_sanitized(self):
        loader = RegistryLoader()
        fsl = loader.module("fsl")
        risk = fsl["supporting"]["risk_prediction"]

        self.assertEqual(fsl["display_name"], "第一部分：参数预测与风险预警")
        self.assertAlmostEqual(
            risk["parameter_prediction"]["joint_accuracy"],
            0.9501108647,
            places=6,
        )
        self.assertEqual(
            risk["evaluation_status"],
            "frozen_offline_experiment",
        )
        serialized = json.dumps(risk, ensure_ascii=False)
        self.assertNotIn("C:\\Users", serialized)
        self.assertNotIn("OneDrive", serialized)

    def test_dt_dataset_registry_discovers_raw_frac_segments(self):
        loader = RegistryLoader(dataset_id="raw_fdbh15")
        self.assertIn("raw_fdbh15", loader.dataset_catalog()["datasets"])
        self.assertTrue(loader.dataset_ready())
        loader.set_scenario("no_das_pressure_only")
        self.assertEqual(loader.scenario()["observation_mode"], "pressure_only")
        self.assertEqual(loader.html().name, "dt_realtime_3d_no_das.html")

    def test_composite_workbook_is_not_exposed_as_a_dataset(self):
        loader = RegistryLoader()
        catalog = loader.dataset_catalog()["datasets"]
        self.assertNotIn("raw_1", catalog)
        self.assertFalse(any("便签数据" in str(item.get("pressure_source", "")) for item in catalog.values()))

    def test_dt_dataset_choices_follow_observation_scenario(self):
        loader = RegistryLoader()
        das = loader.datasets_for_scenario("das_cluster_observation")
        no_das = loader.datasets_for_scenario("no_das_pressure_only")

        self.assertTrue(das)
        self.assertTrue(no_das)
        self.assertTrue(all(item.get("fiber_source") for item in das))
        self.assertTrue(all(not item.get("fiber_source") for item in no_das))
        self.assertNotIn("jy84_z1_stage08", {item["dataset_id"] for item in no_das})
        self.assertNotIn("raw_fdbh15", {item["dataset_id"] for item in das})
        self.assertTrue(
            all(
                loader.dataset_ready_for_scenario(item["dataset_id"], "das_cluster_observation")
                for item in das
            )
        )
        self.assertTrue(
            all(
                loader.dataset_ready_for_scenario(item["dataset_id"], "no_das_pressure_only")
                for item in no_das
            )
        )

    def test_global_dataset_switch_selects_a_compatible_scene(self):
        loader = RegistryLoader(scenario_id="no_das_pressure_only", dataset_id="raw_fdbh1")
        loader.set_dataset("jy84_z1_stage08")
        self.assertEqual(loader.dataset_id, "jy84_z1_stage08")
        self.assertEqual(loader.scenario_id, "das_cluster_observation")

        loader.set_dataset("raw_fdbh1")
        self.assertEqual(loader.dataset_id, "raw_fdbh1")
        self.assertEqual(loader.scenario_id, "no_das_pressure_only")

    def test_raw_no_das_replay_contains_evolving_cluster_states(self):
        loader = RegistryLoader(dataset_id="raw_fdbh15", scenario_id="no_das_pressure_only")
        frames = build_replay_frames(loader)
        self.assertGreaterEqual(len(frames), 24)
        first = frames[0]["dt"]["posterior_half_lengths_m"]
        last = frames[-1]["dt"]["posterior_half_lengths_m"]
        self.assertEqual(len(first), 6)
        self.assertEqual(len(last), 6)
        self.assertNotEqual(first, last)

    def test_raw_no_das_has_independent_pressure_comparison_and_actions(self):
        loader = RegistryLoader(dataset_id="raw_fdbh15", scenario_id="no_das_pressure_only")
        dt = DTLoader(loader)
        hmi = HMILoader(loader)
        cache = dt.cache
        arrays = cache.get("arrays", {})
        self.assertTrue(dt.hmi_available())
        self.assertGreater(len(hmi.rows), 10)
        self.assertEqual(len(arrays["prior_bhp_mpa"]), len(arrays["observed_bhp_mpa"]))
        self.assertGreater(
            sum(
                abs(float(prior) - float(observed)) > 1.0e-6
                for prior, observed in zip(arrays["prior_bhp_mpa"], arrays["observed_bhp_mpa"])
            ),
            0,
        )
        self.assertGreater(
            sum(
                abs(
                    float(row.get("current_sand_ratio_percent", "nan"))
                    - float(row.get("recommended_sand_ratio_percent", "nan"))
                )
                > 1.0e-6
                for row in hmi.rows
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
