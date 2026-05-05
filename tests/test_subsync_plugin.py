"""Tests for subsync/plugin.py."""
from __future__ import annotations

import importlib.util
import io
import json
import unittest
from pathlib import Path

_PLUGIN_PATH = Path(__file__).resolve().parents[1] / "subsync" / "plugin.py"
_spec = importlib.util.spec_from_file_location("subsync_plugin_under_test", _PLUGIN_PATH)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


def _read_events(stdout: io.StringIO) -> list:
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]


class ImportSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(plugin, "DEFAULT_CONFIG"))
        self.assertEqual(plugin.DEFAULT_CONFIG["subtitle_path"], "")
        self.assertEqual(plugin.DEFAULT_CONFIG["max_offset_seconds"], 60.0)
        self.assertTrue(plugin.DEFAULT_CONFIG["framerate_correction"])
        self.assertFalse(plugin.DEFAULT_CONFIG["fail_on_no_match"])


class ProtocolSkeletonTests(unittest.TestCase):
    """Confirms the Task-1 no-op plugin acknowledges the host protocol."""

    def test_no_init_message(self):
        stdin = io.StringIO()  # empty
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("init", events[-1]["error"]["msg"])

    def test_unknown_step_id_returns_error(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "subsync.unknown",
            "ctx": {"file": {"path": "/x"}},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("unknown step_id", events[-1]["error"]["msg"])

    def test_happy_path_skeleton_emits_result_ok(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "subsync.align",
            "ctx": {"file": {"path": "/x"}},
            "config": {},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})
