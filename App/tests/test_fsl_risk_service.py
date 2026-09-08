from __future__ import annotations

import unittest

from App.services.fsl_risk_service import analyze_stage, model_catalog, model_options


class FSLRiskServiceTests(unittest.TestCase):
    @staticmethod
    def rows(count: int = 14):
        return [
            {
                "SGBY": 50.0 + index * 0.35,
                "PL": 12.0,
                "SB": 8.0,
                "LJYL": index * 2.0,
                "LJSL": index * 0.2,
                "BZJD": 1.0,
                "YTND": 1.0,
                "JDPL": 12.0,
                "JDSL": 0.2,
            }
            for index in range(count)
        ]

    def test_rule_engine_returns_causal_point_results(self):
        result = analyze_stage(
            [float(index * 10) for index in range(14)],
            self.rows(),
            model_id="rule_84",
        )
        self.assertEqual(result.model_status, "ready")
        self.assertEqual(len(result.probability), 14)
        self.assertIn("黄色", result.levels)
        self.assertTrue(result.rule_intervals)
        self.assertTrue(all(item["source"] == "8.4因果砂堵规则" for item in result.rule_intervals))

    def test_reviewed_model_weights_are_registered(self):
        catalog = model_catalog()
        self.assertIn("z6_source", catalog)
        self.assertIn("z7_transfer", catalog)
        option_ids = {model_id for model_id, _ in model_options()}
        self.assertTrue({"auto", "rule_84", "z6_source", "z7_transfer"}.issubset(option_ids))

    def test_z7_numpy_model_runs_without_pytorch_runtime(self):
        result = analyze_stage(
            [float(index * 10) for index in range(14)],
            self.rows(),
            model_id="z7_transfer",
        )
        self.assertEqual(result.model_status, "experimental")
        self.assertEqual(result.model_id, "z7_transfer")
        self.assertEqual(len(result.levels), 14)
        self.assertTrue(result.model_reason)
        self.assertTrue(any(value is not None for value in result.probability))


if __name__ == "__main__":
    unittest.main()
