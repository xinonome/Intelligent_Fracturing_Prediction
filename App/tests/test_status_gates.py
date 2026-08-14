from __future__ import annotations

import unittest

from App.core.status import evaluate_status


class StatusGateTests(unittest.TestCase):
    def test_demo_only_hmi_cannot_be_validated(self):
        result = evaluate_status("hmi", {"scientific_status": "demo_only", "quality_gate": {"passed": True}, "total_timesteps": 100000})
        self.assertEqual(result.status, "development_only")

    def test_missing_registered_file_is_not_available(self):
        result = evaluate_status("dt", {"metrics": {"validation_pass": True}}, required_exists=False)
        self.assertEqual(result.status, "not_available")

    def test_invalid_html_is_invalid(self):
        result = evaluate_status("dt", {"metrics": {"validation_pass": True}}, html_valid=False)
        self.assertEqual(result.status, "invalid")


if __name__ == "__main__":
    unittest.main()
