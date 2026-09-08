from pathlib import Path
import tempfile
import unittest

from App.services.operator_decisions import append_decision, load_decisions, rollback_decision


class OperatorDecisionLogTests(unittest.TestCase):
    def test_append_only_log_preserves_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            append_decision({"time_s": 12, "decision": "确认采用"}, path)
            append_decision({"time_s": 13, "decision": "撤销上次确认"}, path)
            rows = load_decisions(path)
            self.assertEqual([row["decision"] for row in rows], ["确认采用", "撤销上次确认"])
            self.assertTrue(all(row.get("recorded_at") for row in rows))

    def test_modified_values_and_rollback_are_preserved_as_audit_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            saved = append_decision({
                "dataset_id": "raw_fdbh10",
                "model_id": "sac",
                "decision": "修改后采用",
                "recommended_flow_m3_min": 5.5,
                "recommended_sand_ratio_pct": 1.8,
                "flow_m3_min": 5.1,
                "sand_ratio_pct": 1.6,
            }, path)
            rollback = rollback_decision(
                saved["record_id"], dataset_id="raw_fdbh10", model_id="sac", path=path,
            )
            rows = load_decisions(path, limit=None)
            self.assertEqual(rows[0]["flow_m3_min"], 5.1)
            self.assertEqual(rows[0]["sand_ratio_pct"], 1.6)
            self.assertEqual(rollback["rollback_of"], saved["record_id"])
            self.assertEqual(rollback["execution_state"], "已撤销")


if __name__ == "__main__":
    unittest.main()
