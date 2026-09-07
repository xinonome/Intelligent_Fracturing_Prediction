from __future__ import annotations

import unittest

from App.ui.widgets.chart_panel import _format_tick, _nice_axis_ticks


class ChartAxisTests(unittest.TestCase):
    def test_pressure_uses_round_engineering_ticks(self) -> None:
        low, high, ticks, step = _nice_axis_ticks([60.0, 137.0], y_min=0.0)
        self.assertEqual((low, high), (0.0, 150.0))
        self.assertEqual(ticks, [0.0, 50.0, 100.0, 150.0])
        self.assertEqual(step, 50.0)
        self.assertEqual([_format_tick(value, step) for value in ticks], ["0", "50", "100", "150"])

    def test_flow_and_sand_choose_their_own_scale(self) -> None:
        flow = _nice_axis_ticks([0.0, 14.9], y_min=0.0)
        sand = _nice_axis_ticks([0.0, 13.7], y_min=0.0)
        self.assertEqual(flow[1], 15.0)
        self.assertEqual(sand[1], 15.0)
        self.assertEqual(flow[3], 5.0)
        self.assertEqual(sand[3], 5.0)

    def test_axis_floor_does_not_become_negative(self) -> None:
        low, high, ticks, _step = _nice_axis_ticks([0.2, 0.8], y_min=0.0)
        self.assertEqual(low, 0.0)
        self.assertGreaterEqual(min(ticks), 0.0)
        self.assertGreaterEqual(high, 0.8)

    def test_scale_changes_with_data_range(self) -> None:
        low, high, ticks, step = _nice_axis_ticks([0.0, 260.0], y_min=0.0)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 300.0)
        self.assertEqual(step, 100.0)
        self.assertEqual(len(ticks), 4)


if __name__ == "__main__":
    unittest.main()
