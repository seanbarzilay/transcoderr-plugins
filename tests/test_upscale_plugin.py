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


class ParseExecuteTests(unittest.TestCase):
    def test_full_message_with_explicit_config(self):
        line = json.dumps({
            "step_id": "upscale.video",
            "ctx": {"file": {"path": "/data/Movie.avi"}},
            "config": {
                "model": "realesr-general-x4v3",
                "scale": 4,
                "target_height": 720,
                "min_source_height": 480,
                "output_path": "/data/out.mkv",
                "denoise_strength": 0.3,
                "tile_size": 256,
            },
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["step_id"], "upscale.video")
        self.assertEqual(result["file_path"], "/data/Movie.avi")
        self.assertEqual(result["config"]["model"], "realesr-general-x4v3")
        self.assertEqual(result["config"]["target_height"], 720)
        self.assertEqual(result["config"]["output_path"], "/data/out.mkv")

    def test_missing_config_fills_in_all_defaults(self):
        line = json.dumps({
            "step_id": "upscale.video",
            "ctx": {"file": {"path": "/data/Movie.avi"}},
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"], plugin.DEFAULT_CONFIG)

    def test_partial_config_merges_with_defaults(self):
        line = json.dumps({
            "step_id": "upscale.video",
            "ctx": {"file": {"path": "/data/Movie.avi"}},
            "config": {"target_height": 720},
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"]["target_height"], 720)
        self.assertEqual(result["config"]["model"], "realesr-animevideov3")  # default
        self.assertEqual(result["config"]["scale"], 4)

    def test_missing_step_id_raises(self):
        line = json.dumps({"ctx": {"file": {"path": "/x"}}})
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("step_id", str(ctx.exception))

    def test_missing_file_path_raises(self):
        line = json.dumps({"step_id": "upscale.video", "ctx": {}})
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("file", str(ctx.exception).lower())

    def test_invalid_json_raises(self):
        with self.assertRaises(plugin.ProtocolError):
            plugin.parse_execute("{not json")


class StdoutWriterTests(unittest.TestCase):
    def test_emit_log(self):
        buf = io.StringIO()
        plugin.emit_log("hello", out=buf)
        self.assertEqual(buf.getvalue(), '{"event":"log","msg":"hello"}\n')

    def test_emit_progress(self):
        buf = io.StringIO()
        plugin.emit_progress(50, 100, out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {"event": "progress", "done": 50, "total": 100})

    def test_emit_context_set(self):
        buf = io.StringIO()
        plugin.emit_context_set("upscale", {"path": "/x"}, out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {
            "event": "context_set",
            "key": "upscale",
            "value": {"path": "/x"},
        })

    def test_emit_result_ok(self):
        buf = io.StringIO()
        plugin.emit_result_ok(out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {"event": "result", "status": "ok", "outputs": {}})

    def test_emit_result_err(self):
        buf = io.StringIO()
        plugin.emit_result_err("oops", out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {
            "event": "result",
            "status": "error",
            "error": {"msg": "oops"},
        })


if __name__ == "__main__":
    unittest.main()
