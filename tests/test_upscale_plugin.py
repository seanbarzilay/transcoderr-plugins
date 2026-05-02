"""Tests for upscale/plugin.py."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "upscale"
sys.path.insert(0, str(PLUGIN_DIR))
import plugin  # noqa: E402


class ImportSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(plugin, "DEFAULT_CONFIG"))
        self.assertEqual(plugin.DEFAULT_CONFIG["model"], "realesr-animevideov3")
        self.assertEqual(plugin.DEFAULT_CONFIG["scale"], 4)
        self.assertEqual(plugin.DEFAULT_CONFIG["target_height"], 1080)
        self.assertEqual(plugin.DEFAULT_CONFIG["min_source_height"], 720)
        self.assertIsNone(plugin.DEFAULT_CONFIG["output_path"])
        self.assertEqual(plugin.DEFAULT_CONFIG["tile_size"], 0)

    def test_protocol_error_class_exists(self):
        self.assertTrue(issubclass(plugin.ProtocolError, Exception))


if __name__ == "__main__":
    unittest.main()
