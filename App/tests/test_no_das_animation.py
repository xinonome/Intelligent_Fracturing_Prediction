from __future__ import annotations

import unittest

from App.data.no_das_animation import find_no_das_gif


class NoDasAnimationTests(unittest.TestCase):
    def test_registered_single_stage_gif_is_resolved(self):
        path = find_no_das_gif("raw_fdbh15", stage_id="FDBH15")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_missing_dataset_does_not_borrow_another_stage(self):
        self.assertIsNone(find_no_das_gif("raw_unknown_stage", stage_id="unknown"))


if __name__ == "__main__":
    unittest.main()
