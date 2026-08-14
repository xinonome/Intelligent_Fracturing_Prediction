from __future__ import annotations

import unittest

from App.core.timeline import TimelineController, build_time_axis


class TimelineTests(unittest.TestCase):
    def test_axis_is_relative_and_one_based(self):
        self.assertEqual(build_time_axis([100, 101, 103]), [1.0, 2.0, 3.0, 4.0])

    def test_controller_emits_frame(self):
        frames = [{"time_s": 1}, {"time_s": 2}]
        controller = TimelineController(frames)
        seen = []
        controller.frameChanged.connect(seen.append)
        controller.set_index(1)
        self.assertEqual(seen[-1]["time_s"], 2)
        self.assertEqual(controller.set_time(1.1)["time_s"], 1)


if __name__ == "__main__":
    unittest.main()
