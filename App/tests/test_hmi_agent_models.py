import unittest

from App.data.hmi_loader import HMILoader, discover_agent_models
from App.core.replay import build_replay_frames
from App.data.registry_loader import RegistryLoader


class HMIModelSelectionTests(unittest.TestCase):
    def test_real_agent_replays_are_discoverable(self) -> None:
        models = discover_agent_models(RegistryLoader())
        model_ids = {item["model_id"] for item in models}
        self.assertTrue({"sac", "td3", "ppo"} <= model_ids)
        self.assertTrue(all(item["ready"] for item in models if item["model_id"] in {"sac", "td3", "ppo"}))
        self.assertTrue(all(item["evaluation_path"].endswith("rl_evaluation.csv") for item in models))

    def test_switching_model_changes_the_loaded_action_source(self) -> None:
        registry = RegistryLoader()
        sac = HMILoader(registry, model_id="sac")
        td3 = HMILoader(registry, model_id="td3")
        self.assertEqual(sac.model_info["model_id"], "sac")
        self.assertEqual(td3.model_info["model_id"], "td3")
        self.assertNotEqual(sac.eval_path, td3.eval_path)
        self.assertNotEqual(sac.rows[0]["flow_m3_min"], td3.rows[0]["flow_m3_min"])
        self.assertNotEqual(sac.rows[0]["sand_ratio_percent"], td3.rows[0]["sand_ratio_percent"])

    def test_raw_well_does_not_reuse_global_model_replay(self) -> None:
        registry = RegistryLoader(scenario_id="no_das_pressure_only", dataset_id="raw_fdbh1")
        loader = HMILoader(registry, model_id="td3")
        self.assertEqual(loader.rows, [])
        frames = build_replay_frames(registry, agent_model="td3")
        self.assertGreater(len(frames), 0)
        self.assertEqual(frames[0]["hmi_model_id"], "td3")
        self.assertIsNone(frames[0]["action_flow"])


if __name__ == "__main__":
    unittest.main()
