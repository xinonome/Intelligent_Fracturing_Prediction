from __future__ import annotations

import unittest

from App.data.registry_loader import RegistryLoader


class RegistryTests(unittest.TestCase):
    def test_dt_registers_one_frame_source_and_one_html(self):
        loader = RegistryLoader()
        dt = loader.module("dt")
        self.assertEqual(dt["frame_count"], 7331)
        self.assertTrue(loader.frame_source().exists())
        self.assertEqual(loader.html().name, "dt_realtime_3d.html")


if __name__ == "__main__":
    unittest.main()
