"""Tests for upscale/plugin.py."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_PATH = Path(__file__).resolve().parents[1] / "upscale" / "plugin.py"
_spec = importlib.util.spec_from_file_location("upscale_plugin_under_test", _PLUGIN_PATH)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


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


class ProbeDimensionsTests(unittest.TestCase):
    def _make_completed(self, stdout_str: str, returncode: int = 0):
        cp = mock.MagicMock()
        cp.stdout = stdout_str
        cp.returncode = returncode
        return cp

    def test_returns_width_height_for_valid_video(self):
        ffprobe_out = json.dumps({"streams": [{"width": 720, "height": 480}]})
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed(ffprobe_out)):
            self.assertEqual(plugin.probe_dimensions(Path("/x")), (720, 480))

    def test_raises_when_no_video_stream(self):
        ffprobe_out = json.dumps({"streams": []})
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed(ffprobe_out)):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.probe_dimensions(Path("/x"))
            self.assertIn("video stream", str(ctx.exception).lower())

    def test_raises_when_ffprobe_not_found(self):
        with mock.patch.object(plugin.subprocess, "run", side_effect=FileNotFoundError("ffprobe")):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.probe_dimensions(Path("/x"))
            self.assertIn("ffprobe", str(ctx.exception).lower())

    def test_raises_when_ffprobe_returns_invalid_json(self):
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed("not json")):
            with self.assertRaises(plugin.ProtocolError):
                plugin.probe_dimensions(Path("/x"))

    def test_raises_when_stream_lacks_dimensions(self):
        # ffprobe can return a stream entry without width/height for
        # damaged containers or image-only streams. Map to ProtocolError
        # rather than letting the bare KeyError escape.
        ffprobe_out = json.dumps({"streams": [{"index": 0}]})
        with mock.patch.object(plugin.subprocess, "run", return_value=self._make_completed(ffprobe_out)):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.probe_dimensions(Path("/x"))
            self.assertIn("width", str(ctx.exception).lower())


class _FakePopen:
    """Stand-in for subprocess.Popen used by run_upscale_subprocess tests."""

    def __init__(self, stderr_lines: list, returncode: int = 0):
        self.stderr = io.StringIO("\n".join(stderr_lines) + ("\n" if stderr_lines else ""))
        self._returncode = returncode

    def wait(self):
        return self._returncode


class RunUpscaleSubprocessTests(unittest.TestCase):
    def test_argv_built_correctly(self):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _FakePopen(stderr_lines=[])

        out = io.StringIO()
        with mock.patch.object(plugin.subprocess, "Popen", side_effect=fake_popen):
            plugin.run_upscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"),
                model="realesr-animevideov3",
                scale=4,
                tile_size=0,
                stdout=out,
            )
        argv = captured["argv"]
        self.assertEqual(argv[0], "realesrgan-ncnn-vulkan")
        self.assertIn("-i", argv)
        self.assertEqual(argv[argv.index("-i") + 1], "/in.mkv")
        self.assertIn("-o", argv)
        self.assertEqual(argv[argv.index("-o") + 1], "/out.mkv")
        self.assertIn("-n", argv)
        self.assertEqual(argv[argv.index("-n") + 1], "realesr-animevideov3")
        self.assertIn("-s", argv)
        self.assertEqual(argv[argv.index("-s") + 1], "4")

    def test_tile_size_zero_omits_t_flag(self):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            return _FakePopen(stderr_lines=[])

        out = io.StringIO()
        with mock.patch.object(plugin.subprocess, "Popen", side_effect=fake_popen):
            plugin.run_upscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"),
                model="m", scale=4, tile_size=0, stdout=out,
            )
        self.assertNotIn("-t", captured["argv"])

    def test_tile_size_nonzero_adds_t_flag(self):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            return _FakePopen(stderr_lines=[])

        out = io.StringIO()
        with mock.patch.object(plugin.subprocess, "Popen", side_effect=fake_popen):
            plugin.run_upscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"),
                model="m", scale=4, tile_size=256, stdout=out,
            )
        argv = captured["argv"]
        self.assertIn("-t", argv)
        self.assertEqual(argv[argv.index("-t") + 1], "256")

    def test_progress_lines_become_progress_events(self):
        stderr = ["1/100", "50/100", "100/100", "done"]
        out = io.StringIO()
        with mock.patch.object(
            plugin.subprocess, "Popen",
            return_value=_FakePopen(stderr_lines=stderr),
        ):
            plugin.run_upscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"),
                model="m", scale=4, tile_size=0, stdout=out,
            )
        events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
        progress = [e for e in events if e["event"] == "progress"]
        self.assertEqual(len(progress), 3)
        self.assertEqual(progress[0], {"event": "progress", "done": 1, "total": 100})
        self.assertEqual(progress[-1], {"event": "progress", "done": 100, "total": 100})

    def test_nonzero_exit_raises_with_stderr_in_message(self):
        stderr = ["model not found"]
        out = io.StringIO()
        with mock.patch.object(
            plugin.subprocess, "Popen",
            return_value=_FakePopen(stderr_lines=stderr, returncode=1),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_upscale_subprocess(
                    Path("/in.mkv"), Path("/out.mkv"),
                    model="bogus", scale=4, tile_size=0, stdout=out,
                )
            self.assertIn("realesrgan failed", str(ctx.exception))
            self.assertIn("model not found", str(ctx.exception))

    def test_oom_stderr_gets_actionable_message(self):
        stderr = ["vkAllocateMemory failed: out of device memory"]
        out = io.StringIO()
        with mock.patch.object(
            plugin.subprocess, "Popen",
            return_value=_FakePopen(stderr_lines=stderr, returncode=1),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_upscale_subprocess(
                    Path("/in.mkv"), Path("/out.mkv"),
                    model="m", scale=4, tile_size=0, stdout=out,
                )
            self.assertIn("OOM", str(ctx.exception))
            self.assertIn("tile_size", str(ctx.exception))

    def test_argv_does_not_include_bogus_f_format_flag(self):
        # The -f flag in realesrgan-ncnn-vulkan is for image output
        # format (jpg/png/webp), not video container — passing -f mkv
        # is semantically wrong and should not appear in the argv.
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            return _FakePopen(stderr_lines=[])

        out = io.StringIO()
        with mock.patch.object(plugin.subprocess, "Popen", side_effect=fake_popen):
            plugin.run_upscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"),
                model="m", scale=4, tile_size=0, stdout=out,
            )
        self.assertNotIn("-f", captured["argv"])

    def test_binary_not_on_path_raises(self):
        out = io.StringIO()
        with mock.patch.object(
            plugin.subprocess, "Popen",
            side_effect=FileNotFoundError("realesrgan-ncnn-vulkan"),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_upscale_subprocess(
                    Path("/in.mkv"), Path("/out.mkv"),
                    model="m", scale=4, tile_size=0, stdout=out,
                )
            self.assertIn("realesrgan-ncnn-vulkan", str(ctx.exception).lower())
            self.assertIn("path", str(ctx.exception).lower())


class RunDownscaleSubprocessTests(unittest.TestCase):
    def _make_completed(self, returncode: int = 0, stderr: str = ""):
        cp = mock.MagicMock()
        cp.returncode = returncode
        cp.stderr = stderr
        return cp

    def test_argv_uses_lanczos_and_target_height(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return self._make_completed()

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run):
            plugin.run_downscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"), target_height=1080,
            )
        argv = captured["argv"]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertIn("-vf", argv)
        vf = argv[argv.index("-vf") + 1]
        self.assertIn("scale=-2:1080", vf)
        self.assertIn("flags=lanczos", vf)
        self.assertIn("libx264", argv)
        self.assertIn("ultrafast", argv)
        self.assertIn("18", argv)  # CRF
        self.assertEqual(argv[-1], "/out.mkv")

    def test_overwrites_existing_output(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return self._make_completed()

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run):
            plugin.run_downscale_subprocess(
                Path("/in.mkv"), Path("/out.mkv"), target_height=1080,
            )
        # -y forces overwrite without prompting.
        self.assertIn("-y", captured["argv"])

    def test_nonzero_exit_raises_with_stderr(self):
        with mock.patch.object(
            plugin.subprocess, "run",
            return_value=self._make_completed(returncode=1, stderr="bad codec"),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_downscale_subprocess(
                    Path("/in.mkv"), Path("/out.mkv"), target_height=1080,
                )
            self.assertIn("downscale", str(ctx.exception).lower())
            self.assertIn("bad codec", str(ctx.exception))

    def test_ffmpeg_not_found_raises(self):
        with mock.patch.object(
            plugin.subprocess, "run",
            side_effect=FileNotFoundError("ffmpeg"),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_downscale_subprocess(
                    Path("/in.mkv"), Path("/out.mkv"), target_height=1080,
                )
            self.assertIn("ffmpeg", str(ctx.exception).lower())
            self.assertIn("path", str(ctx.exception).lower())


class RunMuxSubprocessTests(unittest.TestCase):
    def _make_completed(self, returncode: int = 0, stderr: str = ""):
        cp = mock.MagicMock()
        cp.returncode = returncode
        cp.stderr = stderr
        return cp

    def test_argv_maps_video_from_first_input_audio_subs_from_second(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return self._make_completed()

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run):
            plugin.run_mux_subprocess(
                Path("/video_only.mkv"),
                Path("/original.mkv"),
                Path("/out.mkv"),
            )
        argv = captured["argv"]
        self.assertEqual(argv[0], "ffmpeg")
        # Two -i inputs, in order: video_only then original
        i_indices = [i for i, x in enumerate(argv) if x == "-i"]
        self.assertEqual(len(i_indices), 2)
        self.assertEqual(argv[i_indices[0] + 1], "/video_only.mkv")
        self.assertEqual(argv[i_indices[1] + 1], "/original.mkv")
        # Maps: 0:v:0, 1:a?, 1:s?
        self.assertIn("-map", argv)
        map_values = [argv[i + 1] for i, x in enumerate(argv) if x == "-map"]
        self.assertIn("0:v:0", map_values)
        self.assertIn("1:a?", map_values)
        self.assertIn("1:s?", map_values)
        # -c copy
        c_indices = [i for i, x in enumerate(argv) if x == "-c"]
        self.assertTrue(any(argv[i + 1] == "copy" for i in c_indices))
        # Output last
        self.assertEqual(argv[-1], "/out.mkv")

    def test_overwrites_existing_output(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return self._make_completed()

        with mock.patch.object(plugin.subprocess, "run", side_effect=fake_run):
            plugin.run_mux_subprocess(
                Path("/v.mkv"), Path("/o.mkv"), Path("/out.mkv"),
            )
        self.assertIn("-y", captured["argv"])

    def test_nonzero_exit_raises(self):
        with mock.patch.object(
            plugin.subprocess, "run",
            return_value=self._make_completed(returncode=1, stderr="mux failed"),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_mux_subprocess(
                    Path("/v.mkv"), Path("/o.mkv"), Path("/out.mkv"),
                )
            self.assertIn("mux", str(ctx.exception).lower())
            self.assertIn("mux failed", str(ctx.exception))

    def test_ffmpeg_not_found_raises(self):
        with mock.patch.object(
            plugin.subprocess, "run",
            side_effect=FileNotFoundError("ffmpeg"),
        ):
            with self.assertRaises(plugin.ProtocolError) as ctx:
                plugin.run_mux_subprocess(
                    Path("/v.mkv"), Path("/o.mkv"), Path("/out.mkv"),
                )
            self.assertIn("ffmpeg", str(ctx.exception).lower())
            self.assertIn("path", str(ctx.exception).lower())


def _read_events(stdout: io.StringIO) -> list:
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.avi"
        self.video.write_text("video bytes")  # presence is enough

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, file_path=None, config=None,
             probe_dims=(720, 480),
             upscale_side_effect=None,
             downscale_side_effect=None,
             mux_side_effect=None):
        file_path = file_path if file_path is not None else self.video
        config = config or {}
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "upscale.video",
            "ctx": {"file": {"path": str(file_path)}},
            "config": config,
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()

        def default_upscale(input_path, output_path, **kwargs):
            Path(output_path).write_text("upscaled")

        def default_downscale(input_path, output_path, **kwargs):
            Path(output_path).write_text("downscaled")

        def default_mux(video_only, original, output_path, **kwargs):
            Path(output_path).write_text("muxed")

        # Build a probe_dimensions side_effect: first call returns the
        # source dims (probe_dims); second call returns the arithmetic
        # upscaled dims (src * scale) as a stand-in for what ncnn-vulkan
        # would normally produce.  This keeps existing tests green after
        # the fix that probes the upscaled file instead of computing
        # src*scale arithmetically.
        _scale = {**plugin.DEFAULT_CONFIG, **config}.get("scale", 4)
        _probe_seq = iter([probe_dims, (probe_dims[0] * _scale, probe_dims[1] * _scale)])

        def _probe_side_effect(path):
            return next(_probe_seq)

        with mock.patch.object(plugin, "probe_dimensions", side_effect=_probe_side_effect), \
             mock.patch.object(plugin, "run_upscale_subprocess",
                               side_effect=upscale_side_effect or default_upscale), \
             mock.patch.object(plugin, "run_downscale_subprocess",
                               side_effect=downscale_side_effect or default_downscale), \
             mock.patch.object(plugin, "run_mux_subprocess",
                               side_effect=mux_side_effect or default_mux):
            rc = plugin.main(stdin=stdin, stdout=stdout)
        return rc, _read_events(stdout)

    def test_happy_path_dvd_to_1080_emits_context_and_writes_file(self):
        rc, events = self._run(probe_dims=(720, 480))
        self.assertEqual(rc, 0)
        out = self.dir / "Movie.upscaled.mkv"
        self.assertTrue(out.exists())

        kinds = [e["event"] for e in events]
        self.assertIn("context_set", kinds)
        self.assertEqual(events[-1], {"event": "result", "status": "ok", "outputs": {}})

        ctx_set = next(e for e in events if e["event"] == "context_set")
        self.assertEqual(ctx_set["key"], "upscale")
        self.assertEqual(ctx_set["value"]["from"], "720x480")
        # 720x480 → x4 → 2880x1920 → downscaled to 1620x1080
        self.assertEqual(ctx_set["value"]["to"], "1620x1080")
        self.assertEqual(ctx_set["value"]["model"], "realesr-animevideov3")
        self.assertEqual(ctx_set["value"]["path"], str(out))

    def test_self_gate_skips_when_already_hd(self):
        rc, events = self._run(probe_dims=(1920, 1080))
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "ok")
        kinds = [e["event"] for e in events]
        self.assertIn("log", kinds)
        self.assertNotIn("context_set", kinds)

    def test_target_zero_skips_downscale(self):
        rc, events = self._run(
            probe_dims=(720, 480),
            config={"target_height": 0},
        )
        self.assertEqual(rc, 0)
        ctx_set = next(e for e in events if e["event"] == "context_set")
        self.assertEqual(ctx_set["value"]["to"], "2880x1920")

    def test_skips_downscale_when_model_output_is_below_target(self):
        # 240x160 × 4 = 960x640. Target 1080 — but the model already
        # undershoots; the orchestrator must NOT lanczos-upscale further.
        # ctx.to should reflect the raw model output, not the target.
        rc, events = self._run(
            probe_dims=(240, 160),
            config={"min_source_height": 720, "target_height": 1080},
        )
        self.assertEqual(rc, 0)
        ctx_set = next(e for e in events if e["event"] == "context_set")
        self.assertEqual(ctx_set["value"]["to"], "960x640")

    def test_uses_probed_dims_of_upscaled_file_not_arithmetic(self):
        # The orchestrator must probe the upscaled file for its actual
        # dims rather than computing src*scale. Real ncnn-vulkan output
        # may differ from the arithmetic prediction by a few pixels in
        # edge cases (internal padding). Use a probe sequence that
        # returns different dims for the source vs the upscaled file
        # and verify ctx.to reflects the PROBED upscaled dims.

        # Probe sequence: first call (source) returns 720x480; second
        # call (upscaled file) returns 2884x1924 (slightly different
        # from the arithmetic 2880x1920 due to imagined padding).
        probe_results = iter([(720, 480), (2884, 1924)])

        def fake_probe(path):
            return next(probe_results)

        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "upscale.video",
            "ctx": {"file": {"path": str(self.video)}},
            "config": {"target_height": 0},  # disable downscale to keep ctx.to == probed dims
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()

        def default_upscale(input_path, output_path, **kwargs):
            Path(output_path).write_text("upscaled")

        def default_mux(video_only, original, output_path, **kwargs):
            Path(output_path).write_text("muxed")

        with mock.patch.object(plugin, "probe_dimensions", side_effect=fake_probe), \
             mock.patch.object(plugin, "run_upscale_subprocess", side_effect=default_upscale), \
             mock.patch.object(plugin, "run_mux_subprocess", side_effect=default_mux):
            plugin.main(stdin=stdin, stdout=stdout)

        events = _read_events(stdout)
        ctx_set = next(e for e in events if e["event"] == "context_set")
        # If the orchestrator probed the upscaled file, ctx.to is "2884x1924".
        # If it used arithmetic (720*4 x 480*4), ctx.to is "2880x1920" — wrong.
        self.assertEqual(ctx_set["value"]["to"], "2884x1924")

    def test_explicit_output_path_is_honored(self):
        out = self.dir / "custom-out.mkv"
        rc, events = self._run(
            probe_dims=(720, 480),
            config={"output_path": str(out)},
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())
        ctx_set = next(e for e in events if e["event"] == "context_set")
        self.assertEqual(ctx_set["value"]["path"], str(out))

    def test_missing_file_returns_error(self):
        rc, events = self._run(file_path=self.dir / "nope.mkv")
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("does not exist", events[-1]["error"]["msg"])

    def test_upscale_subprocess_oom_returns_error(self):
        def raise_oom(*a, **kw):
            raise plugin.ProtocolError("OOM during upscale; try a smaller tile_size")

        rc, events = self._run(
            probe_dims=(720, 480),
            upscale_side_effect=raise_oom,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("OOM", events[-1]["error"]["msg"])

    def test_unknown_step_id_returns_error(self):
        stdin = io.StringIO()
        stdin.write('{"event":"init"}\n')
        stdin.write(json.dumps({
            "step_id": "upscale.unknown",
            "ctx": {"file": {"path": str(self.video)}},
        }) + "\n")
        stdin.seek(0)
        stdout = io.StringIO()
        rc = plugin.main(stdin=stdin, stdout=stdout)
        events = _read_events(stdout)
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("step_id", events[-1]["error"]["msg"])


if __name__ == "__main__":
    unittest.main()
