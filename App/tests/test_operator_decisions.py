from pathlib import Path
import tempfile
import unittest

from App.services.operator_decisions import append_decision, load_decisions


class OperatorDecisionLogTests(unittest.TestCase):
    def test_append_only_log_preserves_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            append_decision({"time_s": 12, "decision": "确认采用"}, path)
            append_decision({"time_s": 13, "decision": "撤销上次确认"}, path)
            rows = load_decisions(path)
            self.assertEqual([row["decision"] for row in rows], ["确认采用", "撤销上次确认"])
            self.assertTrue(all(row.get("recorded_at") for row in rows))


if __name__ == "__main__":
    unittest.main()
