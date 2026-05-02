# Upscale Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `upscale` plugin to this catalog. Step `upscale.video` runs `realesrgan-ncnn-vulkan` against the input video, lanczos-downscales the model output to a target height if it overshoots, and re-muxes with the original audio + subtitle streams.

**Architecture:** Single Python module `upscale/plugin.py` (pure stdlib — no `deps`/venv) with pure helpers, JSON event writers, and three subprocess wrappers (`probe_dimensions` / `run_upscale_subprocess` / `run_downscale_subprocess` / `run_mux_subprocess`) orchestrated by `upscale_video` and `main`. Subprocess wrappers are indirected so tests can mock the binaries. `bin/run` is a 3-line POSIX shell wrapper.

**Tech Stack:** Python 3.11+ stdlib (`json`, `subprocess`, `re`, `tempfile`, `shutil`, `pathlib`, `os`, `sys`), `unittest` for tests, external binaries (`ffprobe`, `ffmpeg`, `realesrgan-ncnn-vulkan`) declared via `runtimes`.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `upscale/manifest.toml` | Plugin manifest. Declares the four runtimes; no `deps`. | Create |
| `upscale/bin/run` | 3-line POSIX wrapper that execs `plugin.py` via `python3`. | Create |
| `upscale/plugin.py` | All logic — pure helpers, stdout writers, subprocess wrappers, orchestrator. Importable for tests. | Create |
| `upscale/schema.json` | JSON Schema for per-step config. | Create |
| `upscale/README.md` | User-facing docs: step name, config, ctx output, example flow. | Create |
| `tests/test_upscale_plugin.py` | Stdlib-`unittest` tests at the repo root (kept out of the tarball). | Create |

---

### Task 1: Plugin skeleton (manifest, wrapper, stubs)

**Files:**
- Create: `upscale/manifest.toml`
- Create: `upscale/bin/run`
- Create: `upscale/plugin.py`
- Create: `upscale/schema.json`
- Create: `upscale/README.md`
- Create: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Create `upscale/manifest.toml`**

```toml
name = "upscale"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["upscale.video"]
summary = "Upscales the video stream with Real-ESRGAN (ncnn-vulkan). Designed for SD-source upgrades."
min_transcoderr_version = "0.27.0"
runtimes = ["python3", "ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan"]
```

- [ ] **Step 2: Create `upscale/schema.json`**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "model": {
      "type": "string",
      "default": "realesr-animevideov3",
      "description": "ncnn model name (realesr-animevideov3, realesrgan-x4plus, realesrgan-x4plus-anime, realesr-general-x4v3)"
    },
    "scale": {
      "type": "integer",
      "default": 4,
      "enum": [2, 3, 4]
    },
    "target_height": {
      "type": "integer",
      "default": 1080,
      "description": "Final output height. Lanczos-downscales the model output if it overshoots. 0 disables the post-resize."
    },
    "min_source_height": {
      "type": "integer",
      "default": 720,
      "description": "Self-gate: skip if probe.streams[0].height >= this."
    },
    "output_path": {
      "type": ["string", "null"],
      "default": null,
      "description": "Absolute output path. Null => sibling <basename>.upscaled.mkv."
    },
    "denoise_strength": {
      "type": "number",
      "default": 0.5,
      "description": "Only respected by realesr-general-x4v3."
    },
    "tile_size": {
      "type": "integer",
      "default": 0,
      "description": "0 = auto. Smaller values trade speed for less VRAM."
    }
  }
}
```

- [ ] **Step 3: Create `upscale/plugin.py` (stub)**

```python
#!/usr/bin/env python3
"""upscale.video — AI-upscale via Real-ESRGAN (ncnn-vulkan).

Pure-stdlib orchestrator that drives ffprobe + realesrgan-ncnn-vulkan +
ffmpeg subprocesses. Tests import this module directly; production runs
via bin/run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

DEFAULT_CONFIG = {
    "model": "realesr-animevideov3",
    "scale": 4,
    "target_height": 1080,
    "min_source_height": 720,
    "output_path": None,
    "denoise_strength": 0.5,
    "tile_size": 0,
}


class ProtocolError(Exception):
    """Raised when the JSON-RPC execute message or a subprocess invocation fails."""


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `upscale/bin/run`**

```bash
mkdir -p upscale/bin
```

Then create `upscale/bin/run`:

```sh
#!/bin/sh
# upscale plugin entrypoint. Pure-stdlib Python; no per-plugin venv.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec python3 "$HERE/plugin.py"
```

```bash
chmod +x upscale/bin/run
```

- [ ] **Step 5: Create `upscale/README.md` (stub — full content lands in Task 10)**

```markdown
# upscale

Upscales the video stream with Real-ESRGAN (ncnn-vulkan). Designed for
SD-source upgrades.

(Full README written in Task 10.)
```

- [ ] **Step 6: Create `tests/test_upscale_plugin.py`**

```python
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
```

- [ ] **Step 7: Run tests**

```bash
python3 -m unittest tests.test_upscale_plugin -v
```

Expected: 2 tests pass.

- [ ] **Step 8: Confirm existing test suites still pass**

```bash
python3 -m unittest tests.test_publish tests.test_whisper_plugin 2>&1 | tail -3
```

Expected: existing tests pass.

- [ ] **Step 9: Commit**

```bash
git add upscale tests/test_upscale_plugin.py
git commit -m "upscale: scaffold plugin (manifest, schema, stubs)"
```

---

### Task 2: `compute_target_height` and `parse_progress_line` (pure helpers, TDD)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py` before `if __name__ == "__main__":`:

```python
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
```

- [ ] **Step 2: Run, verify they fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 9 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement both helpers**

Add to `upscale/plugin.py` after `DEFAULT_CONFIG` and `ProtocolError`, before `def main`:

```python
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def compute_target_height(
    source_w: int, source_h: int, target_h: int
) -> tuple[int, int]:
    """Compute the final (width, height) after a lanczos-downscale.

    `target_h == 0` disables the resize: returns the input unchanged.
    Otherwise computes the new width preserving aspect ratio, rounded
    to even (codec requirement).
    """
    if target_h <= 0:
        return source_w, source_h
    new_w = round(source_w * (target_h / source_h))
    if new_w % 2 != 0:
        new_w -= 1
    return new_w, target_h


def parse_progress_line(line: str) -> tuple[int, int] | None:
    """Pull (done, total) frame counts from a ncnn-vulkan stderr line.

    Returns None for lines that don't match or have a zero total.
    """
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    if total == 0:
        return None
    return done, total
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add compute_target_height and parse_progress_line"
```

---

### Task 3: `parse_execute` (TDD)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 6 new tests fail.

- [ ] **Step 3: Implement `parse_execute`**

Add to `upscale/plugin.py` after `parse_progress_line`:

```python
def parse_execute(line: str) -> dict:
    """Parse a JSON-RPC execute line into step_id, file_path, and config.

    Config defaults are filled in for any missing keys.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"execute message is not valid JSON: {exc}") from exc

    step_id = msg.get("step_id")
    if not step_id:
        raise ProtocolError("execute message missing step_id")

    ctx = msg.get("ctx") or {}
    file_path = (ctx.get("file") or {}).get("path")
    if not file_path:
        raise ProtocolError("execute message missing ctx.file.path")

    user_config = msg.get("config") or {}
    config = {**DEFAULT_CONFIG, **user_config}

    return {
        "step_id": step_id,
        "file_path": file_path,
        "config": config,
    }
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add parse_execute"
```

---

### Task 4: stdout protocol writers (TDD)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 5 new tests fail.

- [ ] **Step 3: Implement the writers**

Add to `upscale/plugin.py` after `parse_execute`:

```python
def emit_log(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "log", "msg": msg}, separators=(",", ":")) + "\n")


def emit_progress(done: int, total: int, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "progress", "done": done, "total": total},
        separators=(",", ":"),
    ) + "\n")


def emit_context_set(key: str, value: dict, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "context_set", "key": key, "value": value},
        separators=(",", ":"),
    ) + "\n")


def emit_result_ok(out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "result", "status": "ok", "outputs": {}},
        separators=(",", ":"),
    ) + "\n")


def emit_result_err(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "result", "status": "error", "error": {"msg": msg}},
        separators=(",", ":"),
    ) + "\n")
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 22 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add stdout protocol writers"
```

---

### Task 5: `probe_dimensions` (TDD with mocked subprocess.run)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 4 new tests fail.

- [ ] **Step 3: Implement `probe_dimensions`**

Add to `upscale/plugin.py` after the stdout writers:

```python
def probe_dimensions(file_path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream in file_path."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(file_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProtocolError("ffprobe not on PATH") from exc

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise ProtocolError("no video stream found")
    s = streams[0]
    return int(s["width"]), int(s["height"])
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 26 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add probe_dimensions via ffprobe"
```

---

### Task 6: `run_upscale_subprocess` with progress streaming (TDD)

The trickiest subprocess wrapper — it must stream stderr in real time and emit progress events.

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
        # -i / -o pair present
        self.assertIn("-i", argv)
        self.assertEqual(argv[argv.index("-i") + 1], "/in.mkv")
        self.assertIn("-o", argv)
        self.assertEqual(argv[argv.index("-o") + 1], "/out.mkv")
        # -n model and -s scale
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 7 new tests fail.

- [ ] **Step 3: Implement `run_upscale_subprocess`**

Add to `upscale/plugin.py` after `probe_dimensions`:

```python
def run_upscale_subprocess(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    scale: int,
    tile_size: int,
    stdout,
) -> None:
    """Invoke realesrgan-ncnn-vulkan, stream progress events to stdout.

    Raises ProtocolError on non-zero exit or missing binary.
    """
    argv = [
        "realesrgan-ncnn-vulkan",
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model,
        "-s", str(scale),
        "-f", "mkv",
    ]
    if tile_size and tile_size > 0:
        argv.extend(["-t", str(tile_size)])

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProtocolError("realesrgan-ncnn-vulkan not on PATH") from exc

    stderr_lines: list[str] = []
    if proc.stderr is not None:
        for line in proc.stderr:
            line = line.rstrip("\n")
            stderr_lines.append(line)
            parsed = parse_progress_line(line)
            if parsed is not None:
                done, total = parsed
                emit_progress(done, total, out=stdout)

    rc = proc.wait()
    if rc != 0:
        stderr_blob = "\n".join(stderr_lines).strip()
        if "vkAllocateMemory" in stderr_blob or "out of device memory" in stderr_blob.lower():
            raise ProtocolError(
                "OOM during upscale; try a smaller tile_size"
            )
        raise ProtocolError(f"realesrgan failed: {stderr_blob}")
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 33 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add run_upscale_subprocess with progress streaming"
```

---

### Task 7: `run_downscale_subprocess` (TDD)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 4 new tests fail.

- [ ] **Step 3: Implement `run_downscale_subprocess`**

Add to `upscale/plugin.py` after `run_upscale_subprocess`:

```python
def run_downscale_subprocess(
    input_path: Path, output_path: Path, *, target_height: int
) -> None:
    """Lanczos-downscale input_path to (-2:target_height) and re-encode.

    Intermediate codec is libx264 ultrafast crf=18 — file is transient
    (consumed by the next mux step via -c copy). Raises ProtocolError on
    non-zero exit.
    """
    argv = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale=-2:{target_height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        str(output_path),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ProtocolError("ffmpeg not on PATH") from exc

    if result.returncode != 0:
        raise ProtocolError(
            f"downscale ffmpeg failed: {(result.stderr or '').strip()}"
        )
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 37 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add run_downscale_subprocess (lanczos)"
```

---

### Task 8: `run_mux_subprocess` (TDD)

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
        # All -map values
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 3 new tests fail.

- [ ] **Step 3: Implement `run_mux_subprocess`**

Add to `upscale/plugin.py` after `run_downscale_subprocess`:

```python
def run_mux_subprocess(
    video_only_path: Path,
    original_input: Path,
    output_path: Path,
) -> None:
    """Mux upscaled video stream from video_only_path with audio + subs from original_input.

    Streams are stream-copied (no re-encoding). Audio and subtitle maps
    use `?` suffix so missing streams don't cause ffmpeg to abort.
    """
    argv = [
        "ffmpeg", "-y",
        "-i", str(video_only_path),
        "-i", str(original_input),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-c", "copy",
        str(output_path),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ProtocolError("ffmpeg not on PATH") from exc

    if result.returncode != 0:
        raise ProtocolError(
            f"mux ffmpeg failed: {(result.stderr or '').strip()}"
        )
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 40 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: add run_mux_subprocess (copy-only mux)"
```

---

### Task 9: `upscale_video` orchestrator + `main` (TDD, end-to-end)

The headline task. Wires everything together.

**Files:**
- Modify: `upscale/plugin.py`
- Modify: `tests/test_upscale_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_upscale_plugin.py`:

```python
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
            # Touch the expected output so subsequent steps can probe.
            Path(output_path).write_text("upscaled")

        def default_downscale(input_path, output_path, **kwargs):
            Path(output_path).write_text("downscaled")

        def default_mux(video_only, original, output_path, **kwargs):
            Path(output_path).write_text("muxed")

        with mock.patch.object(plugin, "probe_dimensions", return_value=probe_dims), \
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
        # Default output path is sibling .upscaled.mkv next to input.
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
        rc, events = self._run(probe_dims=(1920, 1080))  # already at 1080
        self.assertEqual(rc, 0)
        self.assertEqual(events[-1]["status"], "ok")
        kinds = [e["event"] for e in events]
        self.assertIn("log", kinds)
        self.assertNotIn("context_set", kinds)

    def test_target_zero_skips_downscale(self):
        # When target_height=0 the downscale step is bypassed; ctx still
        # reports the model output dims.
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
        self.assertEqual(rc, 0)  # main returns 0; failure is in result event
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 8 new tests fail (main is `NotImplementedError`).

- [ ] **Step 3: Implement `upscale_video` and `main`**

Replace the `main` stub in `upscale/plugin.py` and add `upscale_video` before it. Insert this block AFTER `run_mux_subprocess` (the last subprocess wrapper from Task 8):

```python
def upscale_video(
    file_path: Path, config: dict, *, stdout
) -> dict | None:
    """Run the full upscale pipeline. Returns ctx dict on success or None on benign skip.

    Benign skip (source already at/above min_source_height) emits a log
    event but no context_set. Errors raise ProtocolError; the caller
    maps them to result events.
    """
    if not file_path.exists():
        raise ProtocolError("file does not exist")

    src_w, src_h = probe_dimensions(file_path)
    if src_h >= config["min_source_height"]:
        emit_log(
            f"source already at {src_h}p (>= {config['min_source_height']}), skipping",
            out=stdout,
        )
        return None

    output_path = (
        Path(config["output_path"])
        if config.get("output_path")
        else file_path.with_name(f"{file_path.stem}.upscaled.mkv")
    )

    started = time.monotonic()
    work_dir = Path(tempfile.mkdtemp(prefix=f"transcoderr-upscale-{os.getpid()}-"))
    try:
        upscaled = work_dir / "upscaled.mkv"
        run_upscale_subprocess(
            file_path, upscaled,
            model=config["model"],
            scale=config["scale"],
            tile_size=config["tile_size"],
            stdout=stdout,
        )

        # After model: native scale = src × scale.
        model_w = src_w * config["scale"]
        model_h = src_h * config["scale"]

        target_h = config["target_height"]
        # Only lanczos-downscale if the model OVERSHOOT target height.
        # If the model output is at or below target (small source +
        # modest scale), we accept the model output as-is rather than
        # lanczos-UPSCALING further — that would defeat the purpose of
        # using the AI model as the sole upscaler.
        if target_h and target_h > 0 and model_h > target_h:
            resized = work_dir / "resized.mkv"
            run_downscale_subprocess(upscaled, resized, target_height=target_h)
            video_only = resized
            final_w, final_h = compute_target_height(model_w, model_h, target_h)
        else:
            video_only = upscaled
            final_w, final_h = model_w, model_h

        run_mux_subprocess(video_only, file_path, output_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.monotonic() - started
    return {
        "path": str(output_path),
        "from": f"{src_w}x{src_h}",
        "to": f"{final_w}x{final_h}",
        "model": config["model"],
        "duration_sec": round(elapsed, 3),
    }


def main(stdin=None, stdout=None) -> int:
    """Read init+execute from stdin, drive upscale_video, emit events to stdout."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    try:
        _init_line = stdin.readline()
        if not _init_line:
            emit_result_err("no init message on stdin", out=stdout)
            return 0
        exec_line = stdin.readline()
        if not exec_line:
            emit_result_err("no execute message on stdin", out=stdout)
            return 0

        try:
            parsed = parse_execute(exec_line)
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0

        if parsed["step_id"] != "upscale.video":
            emit_result_err(f"unknown step_id: {parsed['step_id']}", out=stdout)
            return 0

        try:
            ctx_value = upscale_video(
                Path(parsed["file_path"]),
                parsed["config"],
                stdout=stdout,
            )
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0

        if ctx_value is not None:
            emit_context_set("upscale", ctx_value, out=stdout)
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_upscale_plugin -v`
Expected: 48 tests pass.

- [ ] **Step 5: Commit**

```bash
git add upscale/plugin.py tests/test_upscale_plugin.py
git commit -m "upscale: implement upscale_video + main protocol orchestration"
```

---

### Task 10: Plugin README

**Files:**
- Modify: `upscale/README.md`

- [ ] **Step 1: Replace the stub README with the full content**

Overwrite `upscale/README.md` with:

```markdown
# upscale

Upscales the video stream with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
via the [`realesrgan-ncnn-vulkan`](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
binary. Designed for SD-source upgrades — DVD rips, VHS, anything where
the source is well below 720p and the operator wants a clean 1080p
output. Vulkan-based, so it runs on AMD, Intel, NVIDIA, and Apple
Silicon GPUs without a CUDA stack.

## Step

`upscale.video` — produces a sibling `<basename>.upscaled.mkv` (or a
configurable `output_path`) with the upscaled video stream re-muxed
together with the original audio + subtitles.

## Manifest dependencies

- `runtimes = ["python3", "ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan"]`
  — all four must be on `$PATH` at install time.
- No `deps` line — the plugin is pure stdlib Python; it orchestrates
  the binaries via subprocess.

The operator is responsible for installing
`realesrgan-ncnn-vulkan` on the host (e.g. unpacking a release zip from
the project's GitHub).

## Config (per-step)

| Key | Default | Description |
|---|---|---|
| `model` | `realesr-animevideov3` | ncnn model name. Other valid names: `realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-general-x4v3` (denoising-friendly for grainy DVDs) |
| `scale` | `4` | model scale factor (`2`, `3`, or `4` depending on model) |
| `target_height` | `1080` | final output height after the model. If the model overshoots, lanczos-downscales to this. `0` disables the post-resize. |
| `min_source_height` | `720` | self-gate: skip if `probe.streams[0].height >= this` |
| `output_path` | `null` | absolute output path. `null` ⇒ sibling `<basename>.upscaled.mkv` |
| `denoise_strength` | `0.5` | only respected by `realesr-general-x4v3` |
| `tile_size` | `0` | `0` = auto. Smaller (e.g. `256`) trades speed for less VRAM on small GPUs. |

## Output

- A new file at `<output_path>` (defaults to a sibling `.upscaled.mkv`).
- `ctx.steps.upscale` is populated with:

  ```json
  {
    "path": "/data/movies/Movie.upscaled.mkv",
    "from": "720x480",
    "to": "1920x1080",
    "model": "realesr-animevideov3",
    "frames": 129600,
    "duration_sec": 4218.3
  }
  ```

  Use `{{ steps.upscale.path }}` in subsequent flow steps.

## Behavior on edge cases

| Case | What happens |
|---|---|
| Input file missing | Step fails with "file does not exist" |
| Source already at or above `min_source_height` | Step succeeds (no-op), no output file written |
| Source has no video stream (audio-only) | Step fails with "no video stream found" from ffprobe |
| `realesrgan-ncnn-vulkan` missing | Step fails — but the install-time `runtimes` check should prevent this |
| Model name unknown to ncnn-vulkan | Step fails with "realesrgan failed: model not found" |
| GPU OOM during upscale | Step fails with "OOM during upscale; try a smaller tile_size" |

## Example flow snippet

```yaml
name: dvd-upscale-normalize
triggers:
  - radarr: [downloaded]
match:
  expr: file.size_gb > 0.001 && file.path.startsWith("/dvd-rips/")
steps:
  - id: probe
    use: probe
  - id: do-upscale
    use: upscale.video
    with:
      model: realesr-general-x4v3
      target_height: 1080
      min_source_height: 720
  - id: probe-upscaled
    use: probe
    with: { path: "{{ steps.upscale.path }}" }
  # ... rest of an encode pipeline operates on steps.upscale.path ...
```

Two `probe` steps because the upscale changes the dimensions; the
downstream encode plan needs to see the new ones.
```

- [ ] **Step 2: Verify the file renders cleanly**

```bash
head -10 upscale/README.md
```

Expected: starts with `# upscale`, no leftover stub text.

```bash
python3 -c "
text = open('upscale/README.md').read()
fences = text.count('\`\`\`')
print(f'fence count: {fences}; even = {fences % 2 == 0}')
"
```

Expected: even count.

- [ ] **Step 3: Commit**

```bash
git add upscale/README.md
git commit -m "upscale: full plugin README"
```

---

### Task 11: Local sanity check (no commit)

Confirms the plugin is publishable end-to-end against the real catalog scripts. Discards local artifacts after.

**Files:** none modified durably.

- [ ] **Step 1: Confirm clean baseline**

`git status`
Expected: working tree clean, on branch `feat/upscale-plugin`.

- [ ] **Step 2: Run publish.py**

```bash
python3 scripts/publish.py upscale
```

Expected: prints `upscale: (new) -> 0.1.0 (<sha>)` and exits 0.

- [ ] **Step 3: Inspect index entry**

```bash
python3 -c "
import json
e = next(p for p in json.load(open('index.json'))['plugins'] if p['name'] == 'upscale')
print('version:', e['version'])
print('runtimes:', e['runtimes'])
print('deps:', e.get('deps'))
print('sha:', e['tarball_sha256'])
print('min_v:', e['min_transcoderr_version'])
"
```

Expected:
- `version: 0.1.0`
- `runtimes: ['python3', 'ffmpeg', 'ffprobe', 'realesrgan-ncnn-vulkan']`
- `deps: None`
- `sha:` matches `shasum -a 256 tarballs/upscale-0.1.0.tar.gz`

- [ ] **Step 4: Inspect tarball contents**

```bash
tar -tzf tarballs/upscale-0.1.0.tar.gz | sort
```

Expected (no test files, no `__pycache__`):

```
upscale/
upscale/README.md
upscale/bin/
upscale/bin/run
upscale/manifest.toml
upscale/plugin.py
upscale/schema.json
```

- [ ] **Step 5: Reproducibility check**

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import publish, hashlib, pathlib
p = pathlib.Path('upscale')
a = publish.build_tarball_bytes(p)
b = publish.build_tarball_bytes(p)
print('identical:', a == b)
print('sha:', hashlib.sha256(a).hexdigest())
"
```

Expected: `identical: True` and a sha matching the index entry.

- [ ] **Step 6: Discard the local publish artifacts**

```bash
git checkout -- index.json tarballs/
rm -f tarballs/upscale-0.1.0.tar.gz
rm -rf upscale/__pycache__ tests/__pycache__ scripts/__pycache__
git status
```

Expected: working tree clean.

---

### Task 12: Final sweep + push + PR

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

```bash
python3 -m unittest tests.test_publish tests.test_whisper_plugin tests.test_upscale_plugin 2>&1 | tail -3
```

Expected: existing tests + 48 new = total. All pass.

- [ ] **Step 2: Confirm git state**

```bash
git status
git log --oneline main..HEAD
```

Expected:
- working tree clean
- 11 commits on the branch (one per task that committed):

```
upscale: full plugin README
upscale: implement upscale_video + main protocol orchestration
upscale: add run_mux_subprocess (copy-only mux)
upscale: add run_downscale_subprocess (lanczos)
upscale: add run_upscale_subprocess with progress streaming
upscale: add probe_dimensions via ffprobe
upscale: add stdout protocol writers
upscale: add parse_execute
upscale: add compute_target_height and parse_progress_line
upscale: scaffold plugin (manifest, schema, stubs)
docs: spec for Real-ESRGAN upscale plugin
```

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/upscale-plugin
gh pr create --base main --head feat/upscale-plugin --title "Add upscale subtitle plugin" --body "$(cat <<'EOF'
## Summary

- New \`upscale/\` plugin: step \`upscale.video\` runs Real-ESRGAN (ncnn-vulkan) against the input video, lanczos-downscales the model output to a target height if it overshoots, and re-muxes with the original audio + subtitle streams.
- Pure-stdlib Python; no \`deps\`/venv. Declares \`runtimes = ["python3", "ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan"]\` — operator installs the realesrgan binary themselves.
- 48 unit + integration tests covering pure helpers, every subprocess wrapper (with mocked binaries), and the full end-to-end protocol.

Spec: \`docs/superpowers/specs/2026-05-03-upscale-plugin-design.md\`
Plan: \`docs/superpowers/plans/2026-05-03-upscale-plugin.md\`

## Test plan

- [x] \`python3 -m unittest tests.test_publish tests.test_whisper_plugin tests.test_upscale_plugin -v\` — all pass
- [x] Local \`python3 scripts/publish.py upscale\` produces a deterministic, clean tarball (verified)
- [ ] After merge: dispatch \`Publish plugin\` workflow with \`plugin=upscale\`, confirm a PR is opened with \`upscale-0.1.0.tar.gz\` and the new index entry
- [ ] After publish: install on a host with \`realesrgan-ncnn-vulkan\` available, run a flow with \`upscale.video\` against an SD source, confirm the upscaled \`.mkv\` lands at the expected path

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
