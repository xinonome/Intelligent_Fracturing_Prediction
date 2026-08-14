from __future__ import annotations

import unittest

from App.data.snapshot_builder import build_replay_frames


class ReplayTests(unittest.TestCase):
    def test_all_sections_exist_in_each_frame(self):
        frames = build_replay_frames()
        if not frames:
            self.skipTest("authorized replay artifacts are not included in the public snapshot")
        self.assertGreaterEqual(len(frames), 240)
        self.assertTrue(all({"fsl", "dt", "hmi"}.issubset(frame) for frame in frames))
        options = {frame["hmi"]["high_level_action"] for frame in frames}
        self.assertTrue(options)
        self.assertTrue(options <= {"hold", "grow", "divert", "safe"})


if __name__ == "__main__":
    unittest.main()
