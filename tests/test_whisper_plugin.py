"""Tests for whisper/plugin.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the whisper plugin's plugin.py importable.
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "whisper"
sys.path.insert(0, str(PLUGIN_DIR))
import plugin  # noqa: E402


class ImportSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(plugin, "DEFAULT_CONFIG"))
        self.assertEqual(plugin.DEFAULT_CONFIG["model"], "large-v3-turbo")
        self.assertEqual(plugin.DEFAULT_CONFIG["language"], "auto")
        self.assertTrue(plugin.DEFAULT_CONFIG["skip_if_exists"])
        self.assertEqual(plugin.DEFAULT_CONFIG["compute_type"], "auto")


if __name__ == "__main__":
    unittest.main()
