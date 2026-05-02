"""Tests for whisper/plugin.py."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

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


class FmtTsTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(plugin.fmt_ts(0.0), "00:00:00,000")

    def test_subsecond(self):
        self.assertEqual(plugin.fmt_ts(1.234), "00:00:01,234")

    def test_minutes_and_seconds(self):
        self.assertEqual(plugin.fmt_ts(61.5), "00:01:01,500")

    def test_hours(self):
        self.assertEqual(plugin.fmt_ts(3661.5), "01:01:01,500")

    def test_rounds_to_nearest_millisecond(self):
        # 0.0005 sits exactly on the boundary; banker's rounding is fine
        # — tests for the ms = round(secs * 1000) behaviour.
        self.assertEqual(plugin.fmt_ts(0.001), "00:00:00,001")
        self.assertEqual(plugin.fmt_ts(0.0014), "00:00:00,001")
        self.assertEqual(plugin.fmt_ts(0.0016), "00:00:00,002")


_Segment = namedtuple("_Segment", ["start", "end", "text"])


class FormatSrtTests(unittest.TestCase):
    def test_empty_iterable_returns_empty_string(self):
        self.assertEqual(plugin.format_srt([]), "")

    def test_single_segment(self):
        out = plugin.format_srt([_Segment(1.0, 2.0, "Hello world.")])
        self.assertEqual(
            out,
            "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n\n",
        )

    def test_two_segments_numbered_and_separated(self):
        segs = [
            _Segment(1.0, 2.0, "One."),
            _Segment(2.5, 3.5, "Two."),
        ]
        self.assertEqual(
            plugin.format_srt(segs),
            "1\n00:00:01,000 --> 00:00:02,000\nOne.\n\n"
            "2\n00:00:02,500 --> 00:00:03,500\nTwo.\n\n",
        )

    def test_text_is_stripped(self):
        out = plugin.format_srt([_Segment(0.0, 1.0, "  spaced  ")])
        self.assertIn("spaced\n\n", out)
        self.assertNotIn("  spaced  \n", out)


class WriteSrtAtomicallyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_exact_bytes(self):
        target = self.dir / "Movie.en.srt"
        plugin.write_srt_atomically(target, "1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n")
        self.assertEqual(
            target.read_text(),
            "1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n",
        )

    def test_no_tmp_left_behind_on_success(self):
        target = self.dir / "Movie.en.srt"
        plugin.write_srt_atomically(target, "x")
        self.assertFalse((self.dir / "Movie.en.srt.tmp").exists())

    def test_overwrites_existing_target(self):
        target = self.dir / "Movie.en.srt"
        target.write_text("OLD")
        plugin.write_srt_atomically(target, "NEW")
        self.assertEqual(target.read_text(), "NEW")

    def test_creates_parent_dir_if_missing(self):
        target = self.dir / "nested" / "Movie.en.srt"
        with self.assertRaises(FileNotFoundError):
            # We do NOT auto-create parents — the caller (transcribe) is
            # responsible. Confirm the contract.
            plugin.write_srt_atomically(target, "x")

    def test_writes_utf8_with_non_ascii_text(self):
        # SRT is defined as UTF-8. Path.write_text without an explicit
        # encoding falls back to the platform locale, which is ASCII on
        # minimal containers — that crashes the moment Whisper hands us a
        # transcript with non-ASCII characters.
        target = self.dir / "Movie.ja.srt"
        plugin.write_srt_atomically(target, "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n")
        self.assertEqual(target.read_bytes().decode("utf-8"),
                         "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n")


class FindExistingSidecarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.mkv"
        self.video.write_text("")  # empty placeholder

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_none_when_no_sidecar_exists(self):
        self.assertIsNone(plugin.find_existing_sidecar(self.video, "en"))
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_exact_match_when_language_fixed(self):
        sidecar = self.dir / "Movie.en.srt"
        sidecar.write_text("")
        self.assertEqual(plugin.find_existing_sidecar(self.video, "en"), sidecar)

    def test_no_match_when_language_fixed_but_only_other_lang_present(self):
        (self.dir / "Movie.fr.srt").write_text("")
        self.assertIsNone(plugin.find_existing_sidecar(self.video, "en"))

    def test_any_lang_match_when_language_is_none(self):
        sidecar = self.dir / "Movie.fr.srt"
        sidecar.write_text("")
        # find_existing_sidecar with None returns the FIRST match found.
        # The exact path (which one of several) is unimportant — any
        # truthy return value is treated as "skip".
        result = plugin.find_existing_sidecar(self.video, None)
        self.assertEqual(result, sidecar)

    def test_no_match_for_different_basename(self):
        (self.dir / "OtherMovie.en.srt").write_text("")
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_does_not_match_the_video_file_itself(self):
        # Movie.mkv exists but isn't a .srt — must not be confused for one.
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_preserves_dot_separated_quality_tags_in_filename(self):
        # Files like "Movie.2024.1080p.mkv" must look for sidecars at
        # "Movie.2024.1080p.<lang>.srt", not "Movie.2024.<lang>.srt".
        complex_video = self.dir / "Movie.2024.1080p.mkv"
        complex_video.write_text("")
        sidecar = self.dir / "Movie.2024.1080p.en.srt"
        sidecar.write_text("")
        # Wrong-prefix sidecar that must NOT be matched.
        (self.dir / "Movie.2024.en.srt").write_text("")
        self.assertEqual(
            plugin.find_existing_sidecar(complex_video, "en"),
            sidecar,
        )


class ParseExecuteTests(unittest.TestCase):
    """The host (transcoderr/src/plugins/subprocess.rs) sends the execute
    message as `{"method":"execute","params":{step_id, with, context}}`.
    `parse_execute` must accept that nested shape; for resilience, it
    also accepts a flat shape with the older `ctx` / `config` field
    names that early test fixtures used.
    """

    def test_host_protocol_full_message(self):
        # The exact shape transcoderr sends in production.
        line = json.dumps({
            "method": "execute",
            "params": {
                "step_id": "whisper.transcribe",
                "with": {
                    "model": "small",
                    "language": "en",
                    "skip_if_exists": False,
                    "compute_type": "int8",
                },
                "context": {"file": {"path": "/data/Movie.mkv"}},
            },
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["step_id"], "whisper.transcribe")
        self.assertEqual(result["file_path"], "/data/Movie.mkv")
        self.assertEqual(result["config"]["model"], "small")
        self.assertEqual(result["config"]["language"], "en")
        self.assertFalse(result["config"]["skip_if_exists"])
        self.assertEqual(result["config"]["compute_type"], "int8")

    def test_host_protocol_missing_with_fills_defaults(self):
        line = json.dumps({
            "method": "execute",
            "params": {
                "step_id": "whisper.transcribe",
                "context": {"file": {"path": "/data/Movie.mkv"}},
            },
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"], plugin.DEFAULT_CONFIG)

    def test_host_protocol_partial_with_merges_defaults(self):
        line = json.dumps({
            "method": "execute",
            "params": {
                "step_id": "whisper.transcribe",
                "with": {"model": "tiny"},
                "context": {"file": {"path": "/data/Movie.mkv"}},
            },
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"]["model"], "tiny")
        self.assertEqual(result["config"]["language"], "auto")  # default
        self.assertTrue(result["config"]["skip_if_exists"])
        self.assertEqual(result["config"]["compute_type"], "auto")

    def test_legacy_flat_shape_with_explicit_config(self):
        # Backward-compat: pre-fix test fixtures and any handwritten
        # tests that send a flat top-level message keep parsing.
        line = json.dumps({
            "step_id": "whisper.transcribe",
            "ctx": {"file": {"path": "/data/Movie.mkv"}},
            "config": {
                "model": "small",
                "language": "en",
                "skip_if_exists": False,
                "compute_type": "int8",
            },
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["step_id"], "whisper.transcribe")
        self.assertEqual(result["file_path"], "/data/Movie.mkv")
        self.assertEqual(result["config"]["model"], "small")

    def test_missing_step_id_raises(self):
        line = json.dumps({
            "method": "execute",
            "params": {"context": {"file": {"path": "/x"}}},
        })
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("step_id", str(ctx.exception))

    def test_missing_file_path_raises(self):
        line = json.dumps({
            "method": "execute",
            "params": {"step_id": "whisper.transcribe", "context": {}},
        })
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("file", str(ctx.exception).lower())

    def test_invalid_json_raises(self):
        with self.assertRaises(plugin.ProtocolError):
            plugin.parse_execute("{not json")


class ResolveComputeTypeTests(unittest.TestCase):
    def test_explicit_int8_passes_through(self):
        self.assertEqual(plugin.resolve_compute_type("int8"), "int8")

    def test_explicit_float16_passes_through(self):
        self.assertEqual(plugin.resolve_compute_type("float16"), "float16")

    def test_auto_with_gpu_returns_float16(self):
        with mock.patch.object(plugin, "cuda_available", return_value=True):
            self.assertEqual(plugin.resolve_compute_type("auto"), "float16")

    def test_auto_without_gpu_returns_int8(self):
        with mock.patch.object(plugin, "cuda_available", return_value=False):
            self.assertEqual(plugin.resolve_compute_type("auto"), "int8")


class StdoutWriterTests(unittest.TestCase):
    def test_emit_log(self):
        buf = io.StringIO()
        plugin.emit_log("hello", out=buf)
        self.assertEqual(buf.getvalue(), '{"event":"log","msg":"hello"}\n')

    def test_emit_log_escapes_quotes(self):
        buf = io.StringIO()
        plugin.emit_log('say "hi"', out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {"event": "log", "msg": 'say "hi"'})

    def test_emit_context_set(self):
        buf = io.StringIO()
        plugin.emit_context_set("whisper", {"language": "en"}, out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {
            "event": "context_set",
            "key": "whisper",
            "value": {"language": "en"},
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

    def test_emit_progress(self):
        buf = io.StringIO()
        plugin.emit_progress(42.5, out=buf)
        msg = json.loads(buf.getvalue().rstrip("\n"))
        self.assertEqual(msg, {"event": "progress", "pct": 42.5})


class HasAudioStreamTests(unittest.TestCase):
    def _make_completed(self, stdout: str, returncode: int = 0):
        cp = mock.MagicMock()
        cp.stdout = stdout
        cp.returncode = returncode
        return cp

    def test_true_when_ffprobe_finds_audio_stream(self):
        ffprobe_out = json.dumps({"streams": [{"index": 1}]})
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed(ffprobe_out)):
            self.assertTrue(plugin.has_audio_stream(Path("/x")))

    def test_false_when_ffprobe_finds_no_audio(self):
        ffprobe_out = json.dumps({"streams": []})
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed(ffprobe_out)):
            self.assertFalse(plugin.has_audio_stream(Path("/x")))

    def test_false_when_ffprobe_returns_no_streams_key(self):
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed("{}")):
            self.assertFalse(plugin.has_audio_stream(Path("/x")))

    def test_raises_when_ffprobe_not_found(self):
        with mock.patch.object(plugin.subprocess, "run", side_effect=FileNotFoundError("ffprobe")):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.has_audio_stream(Path("/x"))
            self.assertIn("ffprobe", str(ctx.exception).lower())


class _FakeInfo:
    def __init__(self, language: str, duration: float = 0.0):
        self.language = language
        self.duration = duration


class _FakeModel:
    def __init__(self, segments, language="en", duration: float = 0.0):
        self._segments = segments
        self._language = language
        self._duration = duration

    def transcribe(self, file_path, language=None, vad_filter=True):
        assert vad_filter is True, "vad_filter must be True"
        return iter(self._segments), _FakeInfo(self._language, self._duration)


def _read_events(stdout: io.StringIO) -> list:
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.mkv"
        self.video.write_text("video bytes")  # presence is enough

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, file_path=None, config=None, model=None, has_audio=True):
        file_path = file_path if file_path is not None else self.video
        config = config or {}
        model = model or _FakeModel(
            segments=[_Segment(0.0, 1.0, "Hello.")],
            language="en",
        )
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "whisper.transcribe",
            "ctx": {"file": {"path": str(file_path)}},
            "config": config,
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        with mock.patch.object(plugin, "load_model", return_value=model), \
             mock.patch.object(plugin, "has_audio_stream", return_value=has_audio):
            rc = plugin.main(stdin=stdin, stdout=stdout)
        return rc, _read_events(stdout)

    def test_happy_path_writes_srt_and_emits_context(self):
        rc, events = self._run()
        self.assertEqual(rc, 0)

        sidecar = self.dir / "Movie.en.srt"
        self.assertTrue(sidecar.exists())
        self.assertIn("Hello.", sidecar.read_text())

        kinds = [e["event"] for e in events]
        self.assertIn("context_set", kinds)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})

        ctx_set = next(e for e in events if e["event"] == "context_set")
        self.assertEqual(ctx_set["key"], "whisper")
        self.assertEqual(ctx_set["value"]["language"], "en")
        self.assertEqual(ctx_set["value"]["model"], "large-v3-turbo")
        self.assertEqual(ctx_set["value"]["subtitle_path"], str(sidecar))
        self.assertGreaterEqual(ctx_set["value"]["duration_sec"], 0.0)

    def test_missing_file_returns_error(self):
        rc, events = self._run(file_path=self.dir / "nope.mkv")
        self.assertEqual(rc, 0)  # main always exits 0; failure is in the result event
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("does not exist", events[-1]["error"]["msg"])

    def test_no_audio_stream_skips_with_log_and_ok(self):
        rc, events = self._run(has_audio=False)
        self.assertEqual(rc, 0)
        self.assertFalse((self.dir / "Movie.en.srt").exists())
        self.assertEqual(events[-1]["status"], "ok")
        kinds = [e["event"] for e in events]
        self.assertIn("log", kinds)
        self.assertNotIn("context_set", kinds)

    def test_skip_if_exists_with_existing_sidecar(self):
        (self.dir / "Movie.en.srt").write_text("existing")
        rc, events = self._run(config={"language": "en", "skip_if_exists": True})
        self.assertEqual(rc, 0)
        self.assertEqual((self.dir / "Movie.en.srt").read_text(), "existing")
        self.assertEqual(events[-1]["status"], "ok")
        kinds = [e["event"] for e in events]
        self.assertNotIn("context_set", kinds)

    def test_emits_progress_events_during_transcribe(self):
        # 100s of audio split into 4 segments. Progress should fire at
        # whole-percent jumps as each segment lands, plus a final 100.0.
        model = _FakeModel(
            segments=[
                _Segment(0.0, 30.0, "First chunk."),
                _Segment(30.0, 60.0, "Second chunk."),
                _Segment(60.0, 90.0, "Third chunk."),
                _Segment(90.0, 100.0, "Last chunk."),
            ],
            language="en",
            duration=100.0,
        )
        rc, events = self._run(model=model)
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})

        progress = [e for e in events if e["event"] == "progress"]
        self.assertGreaterEqual(len(progress), 4, f"expected >=4 progress events, got {len(progress)}")

        pcts = [p["pct"] for p in progress]
        # Monotonic non-decreasing.
        for prev, cur in zip(pcts, pcts[1:]):
            self.assertGreaterEqual(cur, prev, f"progress went backwards: {prev} -> {cur}")
        # Capped at 100.
        for p in pcts:
            self.assertLessEqual(p, 100.0)
        # Final emission is exactly 100.
        self.assertEqual(pcts[-1], 100.0)

    def test_no_progress_when_duration_unknown(self):
        # If faster-whisper's info object lacks a usable duration, the
        # plugin still completes the run but emits no progress events
        # (rather than emitting bogus pct values).
        model = _FakeModel(
            segments=[_Segment(0.0, 1.0, "Hello.")],
            language="en",
            duration=0.0,  # falsy -> skip per-segment emission
        )
        rc, events = self._run(model=model)
        self.assertEqual(rc, 0)
        progress = [e for e in events if e["event"] == "progress"]
        # The final "100.0 done" emission still fires (gives the run UI
        # a clean end state) — but no per-segment ones.
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["pct"], 100.0)

    def test_no_speech_detected_emits_ok_without_srt(self):
        empty_model = _FakeModel(segments=[], language="en")
        rc, events = self._run(model=empty_model)
        self.assertEqual(rc, 0)
        self.assertFalse((self.dir / "Movie.en.srt").exists())
        self.assertEqual(events[-1]["status"], "ok")
        kinds = [e["event"] for e in events]
        self.assertNotIn("context_set", kinds)

    def test_unknown_step_id_returns_error(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "whisper.unknown",
            "ctx": {"file": {"path": str(self.video)}},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        events = _read_events(stdout)
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("step_id", events[-1]["error"]["msg"])

    def test_oom_loading_model_returns_friendly_error(self):
        def _raise_oom(*args, **kwargs):
            raise MemoryError("not enough RAM")
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "whisper.transcribe",
            "ctx": {"file": {"path": str(self.video)}},
            "config": {"model": "large-v3"},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        with mock.patch.object(plugin, "load_model", side_effect=_raise_oom), \
             mock.patch.object(plugin, "has_audio_stream", return_value=True):
            plugin.main(stdin=stdin, stdout=stdout)
        events = _read_events(stdout)
        self.assertEqual(events[-1]["status"], "error")
        msg = events[-1]["error"]["msg"]
        self.assertIn("OOM", msg)
        self.assertIn("large-v3", msg)
        self.assertIn("smaller model", msg)
        self.assertIn("compute_type=int8", msg)


if __name__ == "__main__":
    unittest.main()
