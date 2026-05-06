"""Tests for whisperx/plugin.py."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class FindSubtitlePathTests(unittest.TestCase):
    """Same priority logic as subsync — verbatim test shape."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.2024.1080p.mkv"
        self.video.write_text("v")

    def tearDown(self):
        self._tmp.cleanup()

    def test_override_wins(self):
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

    def test_glob_fallback_when_no_ctx_match(self):
        sidecar = self.dir / "Movie.2024.1080p.en.srt"
        sidecar.write_text("E")
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {}},
            self.video,
        )
        self.assertEqual(out, sidecar)

    def test_returns_none_when_nothing_matches(self):
        out = plugin.find_subtitle_path(
            {"subtitle_path": ""},
            {"steps": {}},
            self.video,
        )
        self.assertIsNone(out)


class ResolveLanguageTests(unittest.TestCase):
    def test_override_wins(self):
        out = plugin.resolve_language(
            {"language": "fr"},
            {"steps": {"whisper": {"language": "de"}}},
            Path("/x/Movie.es.srt"),
        )
        self.assertEqual(out, "fr")

    def test_override_auto_falls_through(self):
        out = plugin.resolve_language(
            {"language": "auto"},
            {"steps": {"whisper": {"language": "de"}}},
            Path("/x/Movie.es.srt"),
        )
        self.assertEqual(out, "de")

    def test_ctx_steps_walk_finds_whisper_language(self):
        out = plugin.resolve_language(
            {"language": ""},
            {"steps": {"whisper": {"language": "de"}}},
            Path("/x/Movie.es.srt"),
        )
        self.assertEqual(out, "de")

    def test_filename_parse_when_ctx_empty(self):
        out = plugin.resolve_language(
            {"language": ""},
            {"steps": {}},
            Path("/x/Movie.en.srt"),
        )
        self.assertEqual(out, "en")

    def test_default_en_when_nothing_else(self):
        stdout = io.StringIO()
        out = plugin.resolve_language(
            {"language": ""},
            {"steps": {}},
            None,
            stdout=stdout,
        )
        self.assertEqual(out, "en")
        events = _read_events(stdout)
        self.assertTrue(any("defaulting to 'en'" in e.get("msg", "") for e in events))

    def test_filename_parse_skips_when_no_lang_token(self):
        out = plugin.resolve_language(
            {"language": ""},
            {"steps": {}},
            Path("/x/Movie.srt"),
        )
        self.assertEqual(out, "en")


class SrtToSegmentsTests(unittest.TestCase):
    def test_single_cue(self):
        srt = (
            "1\n"
            "00:00:01,500 --> 00:00:02,750\n"
            "Hello world.\n"
        )
        out = plugin.srt_to_segments(srt)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["start"], 1.5, places=3)
        self.assertAlmostEqual(out[0]["end"], 2.75, places=3)
        self.assertEqual(out[0]["text"], "Hello world.")

    def test_multi_cue(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nFirst.\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nSecond.\n\n"
        )
        out = plugin.srt_to_segments(srt)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "First.")
        self.assertEqual(out[1]["text"], "Second.")

    def test_multi_line_cue_text_joined_with_spaces(self):
        srt = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "First line\n"
            "Second line\n"
        )
        out = plugin.srt_to_segments(srt)
        self.assertEqual(out[0]["text"], "First line Second line")

    def test_utf8_text(self):
        srt = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "こんにちは\n"
        )
        out = plugin.srt_to_segments(srt)
        self.assertEqual(out[0]["text"], "こんにちは")

    def test_empty_srt_returns_empty_list(self):
        self.assertEqual(plugin.srt_to_segments(""), [])
        self.assertEqual(plugin.srt_to_segments("   \n\n   "), [])

    def test_skip_empty_text_blocks(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\n\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nReal.\n\n"
        )
        out = plugin.srt_to_segments(srt)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "Real.")

    def test_handles_crlf_line_endings(self):
        srt = "1\r\n00:00:01,000 --> 00:00:02,000\r\nHi.\r\n\r\n"
        out = plugin.srt_to_segments(srt)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "Hi.")


class FormatSrtFromAlignedTests(unittest.TestCase):
    def test_uses_first_and_last_word_timestamps(self):
        segments = [{
            "start": 1.0,
            "end": 5.0,
            "text": "Hello world.",
            "words": [
                {"start": 1.2, "end": 1.5, "word": "Hello"},
                {"start": 1.6, "end": 2.0, "word": "world."},
            ],
        }]
        out = plugin.format_srt_from_aligned(segments)
        self.assertIn("00:00:01,200 --> 00:00:02,000", out)
        self.assertIn("Hello world.", out)

    def test_falls_back_to_segment_timestamps_when_no_words(self):
        segments = [{
            "start": 1.0,
            "end": 2.0,
            "text": "No alignment.",
            "words": [],
        }]
        out = plugin.format_srt_from_aligned(segments)
        self.assertIn("00:00:01,000 --> 00:00:02,000", out)

    def test_skips_words_missing_timestamps(self):
        segments = [{
            "start": 1.0,
            "end": 5.0,
            "text": "x",
            "words": [
                {"word": "x"},  # no start/end
            ],
        }]
        out = plugin.format_srt_from_aligned(segments)
        self.assertIn("00:00:01,000 --> 00:00:05,000", out)

    def test_empty_segments_returns_empty_string(self):
        self.assertEqual(plugin.format_srt_from_aligned([]), "")

    def test_skips_segments_with_empty_text(self):
        segments = [
            {"start": 1.0, "end": 2.0, "text": "  ", "words": []},
            {"start": 3.0, "end": 4.0, "text": "Real.", "words": []},
        ]
        out = plugin.format_srt_from_aligned(segments)
        self.assertEqual(out.count("-->"), 1)
        self.assertIn("Real.", out)
        self.assertTrue(out.startswith("1\n"))


class AtomicReplaceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_replaces_existing_target(self):
        target = self.dir / "Movie.en.srt"
        target.write_text("OLD")
        tmp = self.dir / "Movie.en.srt.whisperx.tmp.srt"
        tmp.write_text("NEW")
        plugin.atomic_replace(tmp, target)
        self.assertEqual(target.read_text(), "NEW")
        self.assertFalse(tmp.exists())

    def test_raises_when_tmp_missing(self):
        target = self.dir / "Movie.en.srt"
        tmp = self.dir / "missing.tmp.srt"
        with self.assertRaises(FileNotFoundError):
            plugin.atomic_replace(tmp, target)
