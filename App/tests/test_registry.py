from __future__ import annotations

import unittest

from App.data.registry_loader import RegistryLoader


class RegistryTests(unittest.TestCase):
    def test_public_registry_is_explicitly_unavailable_without_artifacts(self):
        loader = RegistryLoader()
        dt = loader.module("dt")
        self.assertEqual(dt["status"], "not_available")
        self.assertIsNone(loader.frame_source())
        self.assertFalse(loader.html().exists())


if __name__ == "__main__":
    unittest.main()
