from __future__ import annotations

import unittest

from App.data.fsl_timeline_loader import FSLTimelineLoader
from App.data.registry_loader import RegistryLoader


class FSLTimelineTests(unittest.TestCase):
    def test_timeline_exposes_stages_from_independent_workbooks(self):
        loader = FSLTimelineLoader(RegistryLoader())
        self.assertEqual(loader.status, "ready")
        self.assertTrue(loader.source_paths)
        self.assertTrue(all("便签数据" not in path.name for path in loader.source_paths))
        stage_ids = loader.stage_ids()
        self.assertGreaterEqual(len(stage_ids), 2)
        self.assertEqual(len(stage_ids), len(set(stage_ids)))
        self.assertTrue(all(loader.stage(stage_id).get("sample_count", 0) > 0 for stage_id in stage_ids))
        # The source header contains two PL columns.  The measured flow must
        # survive that duplicate-header mapping instead of being overwritten
        # by the later empty annotation column.
        self.assertTrue(all(loader.stage(stage_id).get("flow_max") is not None for stage_id in stage_ids))
        # The same FDBH number can occur in separate source-well exports.
        # They must not be merged into a multi-year pseudo-stage.
        self.assertGreaterEqual(len(stage_ids), len(loader.source_paths))
        self.assertLess(max(loader.stage(stage_id)["duration_s"] for stage_id in stage_ids), 7 * 24 * 3600)
        self.assertTrue(any(stage_id.startswith("FDBH1 ·") for stage_id in stage_ids))
        self.assertIn("FDBH12", stage_ids)
        # Point-level pressure/flow predictions are exposed alongside the
        # measured arrays and carry explicit provenance/metrics metadata.
        point_ready = [loader.stage(stage_id) for stage_id in stage_ids if loader.stage(stage_id).get("point_prediction_count", 0) > 0]
        self.assertTrue(point_ready)
        first = point_ready[0]
        self.assertEqual(len(first["time_s"]), len(first["predicted_pressure_mpa"]))
        self.assertEqual(len(first["time_s"]), len(first["predicted_flow_m3_min"]))
        self.assertEqual(first["point_prediction_source"], "下一采样点逐点预测")
        self.assertIn("pressure", first["point_prediction_metrics"])


if __name__ == "__main__":
    unittest.main()
