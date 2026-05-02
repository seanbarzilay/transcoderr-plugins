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


class ComputeTargetHeightTests(unittest.TestCase):
    def test_proportional_downscale_preserves_aspect_ratio(self):
        # 1920x1080 source upscaled by model to 7680x4320, target 1080
        # → final must stay 1920x1080 (no-op for already-target dims is
        # the same as letting it through).
        w, h = plugin.compute_target_height(7680, 4320, 1080)
        self.assertEqual((w, h), (1920, 1080))

    def test_dvd_to_1080_uses_full_aspect_ratio(self):
        # 720x480 (DVD NTSC) × 4 = 2880x1920. Target 1080 → 1620x1080.
        w, h = plugin.compute_target_height(2880, 1920, 1080)
        self.assertEqual((w, h), (1620, 1080))

    def test_width_is_rounded_to_even(self):
        # Aspect ratios that produce odd widths get nudged to even (codec
        # requirement). Width 1621 → 1620.
        w, h = plugin.compute_target_height(2881, 1920, 1080)
        self.assertEqual((w, h), (1620, 1080))
        self.assertEqual(w % 2, 0)

    def test_target_zero_passes_through_unchanged(self):
        # target_height=0 disables the resize — caller decides what to
        # do (typically: skip the lanczos pass entirely).
        w, h = plugin.compute_target_height(2880, 1920, 0)
        self.assertEqual((w, h), (2880, 1920))


class ParseProgressLineTests(unittest.TestCase):
    def test_slash_separated_format(self):
        self.assertEqual(plugin.parse_progress_line("12345/123456"), (12345, 123456))

    def test_slash_format_with_extra_whitespace(self):
        self.assertEqual(plugin.parse_progress_line("  12345 / 123456  "), (12345, 123456))

    def test_slash_format_in_a_longer_line(self):
        # The binary sometimes prefixes with text; we just want the digits.
        self.assertEqual(
            plugin.parse_progress_line("frame: 7/100 done"),
            (7, 100),
        )

    def test_returns_none_for_unparseable(self):
        self.assertIsNone(plugin.parse_progress_line("hello world"))
        self.assertIsNone(plugin.parse_progress_line(""))
        self.assertIsNone(plugin.parse_progress_line("only one number 5"))

    def test_zero_total_returns_none(self):
        # Don't divide by zero downstream; treat 0/0 as no-progress.
        self.assertIsNone(plugin.parse_progress_line("0/0"))


if __name__ == "__main__":
    unittest.main()
