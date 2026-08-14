from __future__ import annotations

import unittest

from App.core.paths import PATHS
from App.services.webengine_service import validate_html


class WebEngineTests(unittest.TestCase):
    def test_three_d_resource_is_local_and_embedded(self):
        result = validate_html(PATHS.dt_html)
        if not result["exists"]:
            self.skipTest("authorized 3D replay HTML is not included in the public snapshot")
        self.assertTrue(result["exists"])
        self.assertTrue(result["embedded"])
        self.assertFalse(result["external_browser"])


if __name__ == "__main__":
    unittest.main()
