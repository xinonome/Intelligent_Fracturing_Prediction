from __future__ import annotations

import unittest

from App.core.artifacts import ArtifactRegistry
from App.core.paths import PROJECT_ROOT


class KnowledgeGraphViewTests(unittest.TestCase):
    def test_app_uses_local_dependency_free_graph_view(self) -> None:
        module = ArtifactRegistry().module("fsl")
        path = PROJECT_ROOT / module["html"]
        self.assertTrue(path.exists())
        html = path.read_text(encoding="utf-8")
        self.assertIn("核心风险链", html)
        self.assertIn("逐步展开", html)
        self.assertIn('addEventListener("pointerdown"', html)
        self.assertIn('addEventListener("pointermove"', html)
        self.assertIn('addEventListener("wheel"', html)
        self.assertIn("zoomScale = 1", html)
        self.assertNotIn("unpkg.com/vis-network", html)
        self.assertNotIn("cdnjs.cloudflare.com", html)


if __name__ == "__main__":
    unittest.main()
