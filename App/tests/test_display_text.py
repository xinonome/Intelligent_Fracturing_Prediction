from __future__ import annotations

import unittest

from App.ui.display_text import display_text


class DisplayTextTests(unittest.TestCase):
    def test_visible_baseline_label_uses_english_spelling(self) -> None:
        self.assertEqual(display_text("概率基线"), "概率baseline")
        self.assertEqual(display_text("历史基准线"), "历史baseline")

    def test_machine_keys_are_not_changed_by_ui_helper(self) -> None:
        self.assertEqual(display_text("direct_baseline"), "direct_baseline")


if __name__ == "__main__":
    unittest.main()
