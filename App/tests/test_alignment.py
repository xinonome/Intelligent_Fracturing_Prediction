from __future__ import annotations

import unittest

from App.core.timeline import align_categorical, align_numeric


class AlignmentTests(unittest.TestCase):
    def test_numeric_interpolation_keeps_provenance(self):
        value = align_numeric([1, 3], [10, 30], [2], "pressure")[0]
        self.assertEqual(value.value, 20)
        self.assertTrue(value.valid)
        self.assertEqual(value.interpolation, "linear_interpolation")

    def test_outside_numeric_range_is_missing_not_zero(self):
        value = align_numeric([2], [10], [1])[0]
        self.assertFalse(value.valid)
        self.assertIsNone(value.value)

    def test_category_uses_forward_fill(self):
        value = align_categorical([1, 3], ["grow", "hold"], [2])[0]
        self.assertEqual(value.value, "grow")
        self.assertEqual(value.interpolation, "forward_fill")


if __name__ == "__main__":
    unittest.main()
