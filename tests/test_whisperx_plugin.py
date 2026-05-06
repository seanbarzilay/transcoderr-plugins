"""Tests for whisperx/plugin.py."""
from __future__ import annotations

import importlib.util
import io
import json
import unittest
from pathlib import Path

_PLUGIN_PATH = Path(__file__).resolve().parents[1] / "whisperx" / "plugin.py"
_spec = importlib.util.spec_from_file_location("whisperx_plugin_under_test", _PLUGIN_PATH)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


def _read_events(stdout: io.StringIO) -> list:
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]


class ImportSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(plugin, "DEFAULT_ALIGN_CONFIG"))
        self.assertTrue(hasattr(plugin, "DEFAULT_TRANSCRIBE_ALIGNED_CONFIG"))
        self.assertEqual(plugin.DEFAULT_ALIGN_CONFIG["compute_type"], "auto")
        self.assertEqual(plugin.DEFAULT_TRANSCRIBE_ALIGNED_CONFIG["model"], "large-v3-turbo")
        self.assertEqual(plugin.HEARTBEAT_INTERVAL_SECS, 10.0)


class ProtocolSkeletonTests(unittest.TestCase):
    """Confirms the Task-1 no-op plugin acknowledges the host protocol
    and dispatches both step names."""

    def test_no_init_message(self):
        stdin = io.StringIO()
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
            "step_id": "whisperx.unknown",
            "ctx": {"file": {"path": "/x"}},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("unknown step_id", events[-1]["error"]["msg"])

    def test_align_step_id_skeleton_emits_result_ok(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "whisperx.align",
            "ctx": {"file": {"path": "/x"}},
            "config": {},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})

    def test_transcribe_aligned_step_id_skeleton_emits_result_ok(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "whisperx.transcribe_aligned",
            "ctx": {"file": {"path": "/x"}},
            "config": {},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        self.assertEqual(rc, 0)
        events = _read_events(stdout)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})
