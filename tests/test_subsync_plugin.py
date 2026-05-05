"""Tests for subsync/plugin.py."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class FindSubtitlePathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.2024.1080p.mkv"
        self.video.write_text("v")

    def tearDown(self):
        self._tmp.cleanup()

    def test_override_wins(self):
        # Even with a glob match present, the explicit override is preferred.
        (self.dir / "Movie.2024.1080p.en.srt").write_text("E")
        out = plugin.find_subtitle_path(
            {"subtitle_path": "/tmp/explicit.srt"},
            {"steps": {"whisper": {"subtitle_path": "/tmp/from_ctx.srt"}}},
            self.video,
        )
        self.assertEqual(out, Path("/tmp/explicit.srt"))

    def test_ctx_steps_walk_finds_whisper_output(self):
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {"whisper": {"subtitle_path": "/tmp/from_ctx.srt"}}},
            self.video,
        )
        self.assertEqual(out, Path("/tmp/from_ctx.srt"))

    def test_ctx_steps_walk_skips_non_dict_values(self):
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {"size_report": "not-a-dict",
                       "whisper": {"subtitle_path": "/tmp/x.srt"}}},
            self.video,
        )
        self.assertEqual(out, Path("/tmp/x.srt"))

    def test_glob_fallback_when_no_ctx_match(self):
        sidecar = self.dir / "Movie.2024.1080p.en.srt"
        sidecar.write_text("E")
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {}},
            self.video,
        )
        self.assertEqual(out, sidecar)

    def test_glob_picks_most_recently_modified(self):
        old = self.dir / "Movie.2024.1080p.en.srt"
        new = self.dir / "Movie.2024.1080p.de.srt"
        old.write_text("E")
        new.write_text("D")
        # Force `new` to be newer.
        import os as _os
        _os.utime(old, (1000.0, 1000.0))
        _os.utime(new, (2000.0, 2000.0))
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {}},
            self.video,
        )
        self.assertEqual(out, new)

    def test_returns_none_when_nothing_matches(self):
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {}},
            self.video,
        )
        self.assertIsNone(out)


class ParseOffsetFromStderrTests(unittest.TestCase):
    def test_positive_offset(self):
        s = "INFO:root:offset seconds: 1.234\nINFO:root:done.\n"
        self.assertEqual(plugin.parse_offset_from_stderr(s), 1.234)

    def test_negative_offset(self):
        s = "INFO:root:offset seconds: -0.5\n"
        self.assertEqual(plugin.parse_offset_from_stderr(s), -0.5)

    def test_zero_offset(self):
        s = "INFO:root:offset seconds: 0\n"
        self.assertEqual(plugin.parse_offset_from_stderr(s), 0.0)

    def test_no_offset_present_returns_none(self):
        s = "ERROR:root:could not align: no speech detected\n"
        self.assertIsNone(plugin.parse_offset_from_stderr(s))

    def test_handles_extra_whitespace(self):
        s = "INFO:root:offset    seconds:    2.0\n"
        self.assertEqual(plugin.parse_offset_from_stderr(s), 2.0)

    def test_framerate_corrected_true(self):
        s = "INFO:root:framerate scale factor: 1.04\n"
        self.assertTrue(plugin.parse_framerate_corrected_from_stderr(s))

    def test_framerate_corrected_false_when_unity(self):
        s = "INFO:root:framerate scale factor: 1.0\n"
        self.assertFalse(plugin.parse_framerate_corrected_from_stderr(s))

    def test_framerate_corrected_false_when_missing(self):
        self.assertFalse(plugin.parse_framerate_corrected_from_stderr(""))


class AtomicReplaceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_replaces_existing_target(self):
        target = self.dir / "Movie.en.srt"
        target.write_text("OLD")
        tmp = self.dir / "Movie.en.srt.subsync.tmp.srt"
        tmp.write_text("NEW")
        plugin.atomic_replace(tmp, target)
        self.assertEqual(target.read_text(), "NEW")
        self.assertFalse(tmp.exists())

    def test_creates_target_when_missing(self):
        target = self.dir / "Movie.en.srt"
        tmp = self.dir / "Movie.en.srt.subsync.tmp.srt"
        tmp.write_text("NEW")
        plugin.atomic_replace(tmp, target)
        self.assertEqual(target.read_text(), "NEW")
        self.assertFalse(tmp.exists())

    def test_raises_when_tmp_missing(self):
        target = self.dir / "Movie.en.srt"
        tmp = self.dir / "missing.tmp.srt"
        with self.assertRaises(FileNotFoundError):
            plugin.atomic_replace(tmp, target)


class _MockCompleted:
    def __init__(self, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


class RunFfsubsyncTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.mkv"
        self.video.write_text("v")
        self.srt = self.dir / "Movie.en.srt"
        self.srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n")
        self.tmp_out = self.dir / "Movie.en.srt.subsync.tmp.srt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_invokes_ffsubsync_with_expected_args(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _MockCompleted(0, "INFO:root:offset seconds: 1.0\n")

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run):
            rc, stderr = plugin.run_ffsubsync(
                self.video, self.srt, self.tmp_out,
                max_offset_seconds=60, framerate_correction=True,
            )
        self.assertEqual(rc, 0)
        self.assertIn("offset seconds", stderr)
        # Verify the args we hand-construct are correct.
        cmd = captured["cmd"]
        self.assertIn(str(self.video), cmd)
        self.assertIn("-i", cmd)
        self.assertIn(str(self.srt), cmd)
        self.assertIn("-o", cmd)
        self.assertIn(str(self.tmp_out), cmd)
        self.assertIn("--max-offset-seconds", cmd)
        self.assertIn("60", cmd)
        # framerate_correction=True ⇒ no --no-fix-framerate flag.
        self.assertNotIn("--no-fix-framerate", cmd)

    def test_no_fix_framerate_flag_added_when_disabled(self):
        with mock.patch.object(plugin.subprocess, "run",
                               return_value=_MockCompleted(0, "INFO:root:offset seconds: 0.0\n")) as m:
            plugin.run_ffsubsync(
                self.video, self.srt, self.tmp_out,
                max_offset_seconds=30, framerate_correction=False,
            )
        cmd = m.call_args[0][0]
        self.assertIn("--no-fix-framerate", cmd)

    def test_filenotfound_raises_protocolerror(self):
        with mock.patch.object(plugin.subprocess, "run",
                               side_effect=FileNotFoundError("ffsubsync")):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_ffsubsync(
                    self.video, self.srt, self.tmp_out,
                    max_offset_seconds=30, framerate_correction=True,
                )
            self.assertIn("ffsubsync", str(ctx.exception).lower())


class AlignSubtitleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.mkv"
        self.video.write_text("v")
        self.srt = self.dir / "Movie.en.srt"
        self.srt.write_text("ORIGINAL")

    def tearDown(self):
        self._tmp.cleanup()

    def _patch_subprocess(self, returncode: int, stderr: str, write_tmp: bool = True):
        """Patch subprocess.run to fake ffsubsync. If write_tmp, also
        materialise the synced output so atomic_replace can find it."""
        if write_tmp:
            tmp = self.dir / "Movie.en.srt.subsync.tmp.srt"
            tmp.write_text("SYNCED")
        return mock.patch.object(
            plugin.subprocess, "run",
            return_value=_MockCompleted(returncode, stderr),
        )

    def test_happy_path_replaces_srt_and_returns_metadata(self):
        with self._patch_subprocess(0, "INFO:root:offset seconds: 1.234\n"):
            stdout = io.StringIO()
            meta = plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=stdout,
            )
        self.assertEqual(meta["subtitle_path"], str(self.srt))
        self.assertEqual(meta["offset_seconds"], 1.234)
        self.assertFalse(meta["framerate_corrected"])
        self.assertEqual(self.srt.read_text(), "SYNCED")

    def test_framerate_corrected_when_scale_nonunity(self):
        stderr = ("INFO:root:offset seconds: 0.5\n"
                  "INFO:root:framerate scale factor: 1.04\n")
        with self._patch_subprocess(0, stderr):
            meta = plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=io.StringIO(),
            )
        self.assertTrue(meta["framerate_corrected"])

    def test_offset_exceeding_max_is_pass_through(self):
        with self._patch_subprocess(0, "INFO:root:offset seconds: 999.0\n"):
            stdout = io.StringIO()
            meta = plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=stdout,
            )
        self.assertIsNone(meta)
        self.assertEqual(self.srt.read_text(), "ORIGINAL")
        events = _read_events(stdout)
        self.assertTrue(any("exceeds max_offset_seconds" in e.get("msg", "") for e in events))

    def test_unparseable_offset_is_pass_through(self):
        with self._patch_subprocess(0, "INFO:root:done.\n"):
            stdout = io.StringIO()
            meta = plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=stdout,
            )
        self.assertIsNone(meta)
        self.assertEqual(self.srt.read_text(), "ORIGINAL")

    def test_nonzero_rc_warn_and_pass(self):
        with self._patch_subprocess(2, "ERROR:root:something broke\n",
                                    write_tmp=False):
            stdout = io.StringIO()
            meta = plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=stdout,
            )
        self.assertIsNone(meta)
        self.assertEqual(self.srt.read_text(), "ORIGINAL")

    def test_nonzero_rc_with_fail_on_no_match_raises(self):
        with self._patch_subprocess(2, "ERROR:root:nope\n", write_tmp=False):
            with self.assertRaises(plugin.ProtocolError):
                plugin.align_subtitle(
                    self.video, self.srt,
                    {"max_offset_seconds": 60, "framerate_correction": True,
                     "fail_on_no_match": True},
                    stdout=io.StringIO(),
                )

    def test_missing_video_raises_protocolerror(self):
        self.video.unlink()
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.align_subtitle(
                self.video, self.srt,
                {"max_offset_seconds": 60, "framerate_correction": True,
                 "fail_on_no_match": False},
                stdout=io.StringIO(),
            )
        self.assertIn("video", str(ctx.exception).lower())

    def test_missing_srt_is_benign_skip(self):
        self.srt.unlink()
        meta = plugin.align_subtitle(
            self.video, self.srt,
            {"max_offset_seconds": 60, "framerate_correction": True,
             "fail_on_no_match": False},
            stdout=io.StringIO(),
        )
        self.assertIsNone(meta)
