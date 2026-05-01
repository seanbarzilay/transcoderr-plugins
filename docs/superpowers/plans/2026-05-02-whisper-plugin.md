# Whisper Subtitle Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `whisper` plugin to this catalog that produces a sidecar `.srt` for the post-transcode output file using `faster-whisper`, with GPU auto-detection.

**Architecture:** A single Python module (`whisper/plugin.py`) with the protocol logic and pure helpers, plus a thin executable wrapper (`whisper/bin/run`). All `faster_whisper` imports are lazy and indirected through a `load_model()` function so tests can monkeypatch a stub. Per-plugin deps install to `whisper/libs/` via the new `deps` manifest field.

**Tech Stack:** Python 3.11+ stdlib (`json`, `subprocess`, `glob`, `io`, `pathlib`, `time`, `os`, `sys`), `faster-whisper` (loaded lazily, installed by the host at boot), `unittest` for tests.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `whisper/manifest.toml` | Plugin manifest including new `runtimes` and `deps` fields. | Create |
| `whisper/bin/run` | 3-line executable wrapper: shebang, sys.path bootstrap for `libs/`, calls `plugin.main()`. | Create |
| `whisper/plugin.py` | All real logic — pure helpers, stdout writers, the main orchestrator. Importable for tests. | Create |
| `whisper/schema.json` | JSON Schema for per-step config (`model`, `language`, `skip_if_exists`, `compute_type`). | Create |
| `whisper/README.md` | User-facing docs: step name, config, ctx output, example flow. | Create |
| `tests/test_whisper_plugin.py` | Stdlib-`unittest` tests for `plugin.py`. Lives at the repo root, **not** inside `whisper/`, so test code isn't shipped in the tarball. | Create |

`whisper/plugin.py` is the only code file with real logic. `bin/run` is a stub. The split lets tests `import plugin` directly while the on-disk entrypoint stays minimal and predictable.

---

### Task 1: Plugin skeleton (manifest, wrapper, stubs)

This task lays down every file with just enough scaffolding for the rest of the plan to build on. No business logic yet — but everything is in place so subsequent TDD tasks can grow it cleanly.

**Files:**
- Create: `whisper/manifest.toml`
- Create: `whisper/bin/run`
- Create: `whisper/plugin.py`
- Create: `whisper/schema.json`
- Create: `whisper/README.md`
- Create: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Create `whisper/manifest.toml`**

```toml
name = "whisper"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["whisper.transcribe"]
summary = "Generates a sidecar .srt for the output file using faster-whisper. GPU auto-detected."
min_transcoderr_version = "0.27.0"
runtimes = ["python3", "ffprobe"]
deps = "pip install --target ./libs faster-whisper nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"
```

- [ ] **Step 2: Create `whisper/schema.json`**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "model": {
      "type": "string",
      "default": "large-v3-turbo",
      "description": "faster-whisper model name (e.g. tiny, base, small, medium, large-v3, large-v3-turbo)"
    },
    "language": {
      "type": "string",
      "default": "auto",
      "description": "ISO 639-1 language code, or 'auto' for autodetect"
    },
    "skip_if_exists": {
      "type": "boolean",
      "default": true,
      "description": "Skip if a sidecar .srt already exists for this file"
    },
    "compute_type": {
      "type": "string",
      "default": "auto",
      "description": "CTranslate2 compute type. 'auto' picks float16 on GPU, int8 on CPU."
    }
  }
}
```

- [ ] **Step 3: Create `whisper/plugin.py` (stub)**

```python
#!/usr/bin/env python3
"""whisper.transcribe — sidecar .srt via faster-whisper.

Pure helpers and protocol orchestration for the whisper plugin. Tests
import this module directly; production runs via bin/run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG = {
    "model": "large-v3-turbo",
    "language": "auto",
    "skip_if_exists": True,
    "compute_type": "auto",
}


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `whisper/bin/run`**

Make sure the parent directory exists first.

```bash
mkdir -p whisper/bin
```

Then create `whisper/bin/run` with this content:

```bash
#!/bin/sh
# whisper plugin entrypoint. Adds the per-plugin libs/ dir (populated
# by manifest's `deps` line) to PYTHONPATH and execs plugin.py.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$HERE/libs:${PYTHONPATH:-}"
exec python3 "$HERE/plugin.py"
```

Make it executable:

```bash
chmod +x whisper/bin/run
```

- [ ] **Step 5: Create `whisper/README.md`**

```markdown
# whisper

Generates a sidecar `.srt` for the post-transcode output file using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). GPU is
auto-detected; falls back to CPU.

## Step

`whisper.transcribe` — runs after `output` in a flow.

## Config

| Key | Default | Description |
|---|---|---|
| `model` | `large-v3-turbo` | faster-whisper model name |
| `language` | `auto` | ISO 639-1 or `auto` |
| `skip_if_exists` | `true` | Bail if a sidecar already exists |
| `compute_type` | `auto` | CTranslate2 compute type; auto picks `float16` on GPU, `int8` on CPU |

## Output

Writes `<basename>.<lang>.srt` next to the input file. Sets
`steps.whisper` in ctx:

```json
{
  "subtitle_path": "/data/movies/Movie.en.srt",
  "language": "en",
  "model": "large-v3-turbo",
  "duration_sec": 42.7
}
```

(Filled out fully in Task 10.)
```

- [ ] **Step 6: Create `tests/test_whisper_plugin.py` with an import smoke test**

```python
"""Tests for whisper/plugin.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 7: Run the tests**

```bash
python3 -m unittest tests.test_whisper_plugin -v
```

Expected: 1 test, passes.

- [ ] **Step 8: Confirm publish-pipeline tests still pass**

```bash
python3 -m unittest tests.test_publish 2>&1 | tail -3
```

Expected: existing 41 tests pass.

- [ ] **Step 9: Commit**

```bash
git add whisper tests/test_whisper_plugin.py
git commit -m "whisper: scaffold plugin (manifest, schema, README, stubs)"
```

---

### Task 2: SRT formatting (`fmt_ts` + `format_srt`)

The plugin's only job at the end of the day is to write a correct SRT file. These two pure functions are the heart of that.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_whisper_plugin.py` before `if __name__ == "__main__":`:

```python
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


from collections import namedtuple

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
```

- [ ] **Step 2: Run the tests, verify they fail**

```bash
python3 -m unittest tests.test_whisper_plugin -v
```

Expected: `AttributeError: module 'plugin' has no attribute 'fmt_ts'` (and `format_srt`). Existing import smoke test still passes.

- [ ] **Step 3: Implement `fmt_ts` and `format_srt`**

Add to `whisper/plugin.py`, after `DEFAULT_CONFIG` and before `def main`:

```python
def fmt_ts(secs: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    ms_total = int(round(secs * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(segments: Iterable) -> str:
    """Format an iterable of (start, end, text) segments as SRT text.

    `segments` may be any iterable of objects with `.start`, `.end`, and
    `.text` attributes — typically faster-whisper Segments. Text is
    stripped. Cues are 1-indexed and separated by a blank line. Empty
    iterable returns the empty string (no trailing newline).
    """
    parts = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        parts.append(
            f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n"
        )
    return "".join(parts)
```

- [ ] **Step 4: Run, verify pass**

```bash
python3 -m unittest tests.test_whisper_plugin -v
```

Expected: all tests pass (10 total: 1 import + 5 fmt_ts + 4 format_srt).

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add fmt_ts and format_srt"
```

---

### Task 3: `write_srt_atomically`

Atomic write — `<final>.tmp` + `os.replace`. Avoids leaving a half-written sidecar if the process is killed.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
import tempfile


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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 4 new tests fail with AttributeError.

- [ ] **Step 3: Implement `write_srt_atomically`**

Add to `whisper/plugin.py` after `format_srt`:

```python
def write_srt_atomically(path: Path, srt_text: str) -> None:
    """Write srt_text to path atomically (write to .tmp + os.replace).

    The caller is responsible for ensuring the parent directory exists.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(srt_text)
    os.replace(tmp, path)
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add write_srt_atomically"
```

---

### Task 4: `find_existing_sidecar`

Implements the language-aware skip check: exact match when language is fixed, glob match when language is `auto`.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 7 new tests fail with AttributeError.

- [ ] **Step 3: Implement `find_existing_sidecar`**

Add to `whisper/plugin.py` after `write_srt_atomically`:

```python
def find_existing_sidecar(video_path: Path, language: str | None) -> Path | None:
    """Return an existing sidecar .srt for video_path, or None.

    With `language` set to an ISO code, checks for `<basename>.<lang>.srt`
    exactly. With `language=None` (autodetect), returns the first
    `<basename>.*.srt` that exists. `<basename>` here means the filename
    with only its rightmost extension stripped — `Movie.2024.1080p.mkv`
    yields a base of `Movie.2024.1080p` so dot-separated quality tags
    are preserved.
    """
    if language is not None:
        candidate = video_path.with_name(f"{video_path.stem}.{language}.srt")
        return candidate if candidate.exists() else None
    pattern = str(video_path.with_suffix("")) + ".*.srt"
    matches = sorted(glob(pattern))
    return Path(matches[0]) if matches else None
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 21 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add find_existing_sidecar with lang-aware matching"
```

---

### Task 5: `parse_execute`

Pulls `step_id`, `file_path`, and resolved config from a JSON-RPC execute line. Handles missing config keys by filling in defaults from `DEFAULT_CONFIG`.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
import json


class ParseExecuteTests(unittest.TestCase):
    def test_full_message_with_explicit_config(self):
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
        self.assertEqual(result["config"]["language"], "en")
        self.assertFalse(result["config"]["skip_if_exists"])
        self.assertEqual(result["config"]["compute_type"], "int8")

    def test_missing_config_fills_in_all_defaults(self):
        line = json.dumps({
            "step_id": "whisper.transcribe",
            "ctx": {"file": {"path": "/data/Movie.mkv"}},
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"], plugin.DEFAULT_CONFIG)

    def test_partial_config_merges_with_defaults(self):
        line = json.dumps({
            "step_id": "whisper.transcribe",
            "ctx": {"file": {"path": "/data/Movie.mkv"}},
            "config": {"model": "tiny"},
        })
        result = plugin.parse_execute(line)
        self.assertEqual(result["config"]["model"], "tiny")
        self.assertEqual(result["config"]["language"], "auto")  # default
        self.assertTrue(result["config"]["skip_if_exists"])
        self.assertEqual(result["config"]["compute_type"], "auto")

    def test_missing_step_id_raises(self):
        line = json.dumps({"ctx": {"file": {"path": "/x"}}})
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("step_id", str(ctx.exception))

    def test_missing_file_path_raises(self):
        line = json.dumps({"step_id": "whisper.transcribe", "ctx": {}})
        with self.assertRaises(plugin.ProtocolError) as ctx:
            plugin.parse_execute(line)
        self.assertIn("file", str(ctx.exception).lower())

    def test_invalid_json_raises(self):
        with self.assertRaises(plugin.ProtocolError):
            plugin.parse_execute("{not json")
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 6 new tests fail with AttributeError on `plugin.parse_execute` or `plugin.ProtocolError`.

- [ ] **Step 3: Implement `parse_execute` and `ProtocolError`**

Add to `whisper/plugin.py` near the top, after `DEFAULT_CONFIG`:

```python
class ProtocolError(Exception):
    """Raised when the JSON-RPC execute message is missing required fields."""
```

Add to `whisper/plugin.py` after `find_existing_sidecar`:

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

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 27 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add parse_execute and ProtocolError"
```

---

### Task 6: `resolve_compute_type`

Maps the user-facing `compute_type` config to a CTranslate2 type string, with `"auto"` selecting based on CUDA availability. CUDA detection is indirected through a separate function so tests can monkeypatch it.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
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
```

You'll also need `from unittest import mock` near the top of the test file — add it next to the other imports if it's not already there.

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 4 new tests fail with `AttributeError: module 'plugin' has no attribute 'resolve_compute_type'`.

- [ ] **Step 3: Implement `resolve_compute_type` and `cuda_available`**

Add to `whisper/plugin.py` after `parse_execute`:

```python
def cuda_available() -> bool:
    """Return True if a CUDA device looks usable.

    Indirected through this function so tests can monkeypatch it.
    Production: probe ctranslate2's device list lazily — a missing import
    or a no-CUDA-build returns False rather than raising.
    """
    try:
        import ctranslate2  # type: ignore
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_compute_type(user_value: str) -> str:
    """Resolve 'auto' to float16 (GPU) or int8 (CPU); pass others through."""
    if user_value != "auto":
        return user_value
    return "float16" if cuda_available() else "int8"
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 31 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add resolve_compute_type with cuda probe"
```

---

### Task 7: stdout protocol writers

Four small functions for emitting JSON event lines. They take an optional `out` stream so tests can capture without redirecting global stdout.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
import io


class StdoutWriterTests(unittest.TestCase):
    def test_emit_log(self):
        buf = io.StringIO()
        plugin.emit_log("hello", out=buf)
        self.assertEqual(buf.getvalue(), '{"event":"log","msg":"hello"}\n')

    def test_emit_log_escapes_quotes(self):
        buf = io.StringIO()
        plugin.emit_log('say "hi"', out=buf)
        # json.dumps handles the escaping for us.
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 5 new tests fail with AttributeError.

- [ ] **Step 3: Implement the writers**

Add to `whisper/plugin.py` after `resolve_compute_type`:

```python
def emit_log(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "log", "msg": msg}) + "\n")


def emit_context_set(key: str, value: dict, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "context_set", "key": key, "value": value}) + "\n")


def emit_result_ok(out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "result", "status": "ok", "outputs": {}}) + "\n")


def emit_result_err(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "result", "status": "error", "error": {"msg": msg}}) + "\n")
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 36 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add stdout protocol writers"
```

---

### Task 8: `has_audio_stream`

Wraps a single `ffprobe` invocation. Tests mock `subprocess.run`.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 4 new tests fail.

- [ ] **Step 3: Implement `has_audio_stream`**

Add to `whisper/plugin.py` after the stdout writers:

```python
def has_audio_stream(file_path: Path) -> bool:
    """Return True if ffprobe sees at least one audio stream in file_path."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
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
    except json.JSONDecodeError:
        return False
    return bool(data.get("streams"))
```

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 40 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: add has_audio_stream via ffprobe"
```

---

### Task 9: `transcribe` + `main` orchestration (end-to-end)

The headline task. Wires every helper together. Tests stub `load_model` to inject a fake `WhisperModel` and run the full stdin → stdout protocol.

**Files:**
- Modify: `whisper/plugin.py`
- Modify: `tests/test_whisper_plugin.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_whisper_plugin.py`:

```python
class _FakeInfo:
    def __init__(self, language: str):
        self.language = language


class _FakeModel:
    def __init__(self, segments, language="en"):
        self._segments = segments
        self._language = language

    def transcribe(self, file_path, language=None, vad_filter=True):
        return iter(self._segments), _FakeInfo(self._language)


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
```

- [ ] **Step 2: Run, verify fail**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 6 new tests fail because `main` is `NotImplementedError` from Task 1.

- [ ] **Step 3: Implement `load_model`, `transcribe`, and `main`**

Replace the existing `main` stub in `whisper/plugin.py` and add `load_model` + `transcribe` before it. Final layout (everything after the stdout writers from Task 7, replacing the Task 1 stub):

```python
def load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper model. Indirected so tests can monkeypatch."""
    from faster_whisper import WhisperModel  # noqa: WPS433 (lazy import)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(file_path: Path, config: dict, *, stdout) -> dict | None:
    """Transcribe file_path. Returns ctx dict on success, or None on benign skip.

    Benign skips (no-audio, sidecar-already-exists, no-speech) emit a
    log event but no context_set. Errors raise; the caller maps them to
    result events.
    """
    if not file_path.exists():
        raise ProtocolError("file does not exist")

    if not has_audio_stream(file_path):
        emit_log("no audio stream, skipping", out=stdout)
        return None

    skip_lang = None if config["language"] == "auto" else config["language"]
    if config["skip_if_exists"]:
        existing = find_existing_sidecar(file_path, skip_lang)
        if existing is not None:
            emit_log(f"sidecar already exists at {existing}, skipping", out=stdout)
            return None

    compute_type = resolve_compute_type(config["compute_type"])
    model = load_model(config["model"], device="auto", compute_type=compute_type)

    started = time.monotonic()
    segments_iter, info = model.transcribe(
        str(file_path),
        language=skip_lang,
        vad_filter=True,
    )
    segments = list(segments_iter)

    if not segments:
        emit_log("no speech detected, skipping", out=stdout)
        return None

    srt_text = format_srt(segments)
    sidecar = file_path.with_suffix(f".{info.language}.srt")
    write_srt_atomically(sidecar, srt_text)

    elapsed = time.monotonic() - started
    return {
        "subtitle_path": str(sidecar),
        "language": info.language,
        "model": config["model"],
        "duration_sec": round(elapsed, 3),
    }


def main(stdin=None, stdout=None) -> int:
    """Read init+execute from stdin, drive transcribe, emit events to stdout."""
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

        if parsed["step_id"] != "whisper.transcribe":
            emit_result_err(f"unknown step_id: {parsed['step_id']}", out=stdout)
            return 0

        try:
            ctx_value = transcribe(
                Path(parsed["file_path"]),
                parsed["config"],
                stdout=stdout,
            )
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0

        if ctx_value is not None:
            emit_context_set("whisper", ctx_value, out=stdout)
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0
```

Note: `Path.with_suffix(".en.srt")` replaces only the rightmost extension. So `/data/Movie.2024.1080p.mkv` becomes `/data/Movie.2024.1080p.en.srt` — the existing dot-separated quality tags in the basename are preserved.

- [ ] **Step 4: Run, verify pass**

`python3 -m unittest tests.test_whisper_plugin -v`
Expected: 46 tests pass.

- [ ] **Step 5: Commit**

```bash
git add whisper/plugin.py tests/test_whisper_plugin.py
git commit -m "whisper: implement transcribe + main protocol orchestration"
```

---

### Task 10: Plugin README

Replace the stub README from Task 1 with a full user-facing doc — installation note, step config, ctx output, example flow snippet.

**Files:**
- Modify: `whisper/README.md`

- [ ] **Step 1: Replace the README**

Overwrite `whisper/README.md` with:

```markdown
# whisper

Generates a sidecar `.srt` for the post-transcode output file using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). The plugin
is GPU-aware: if CUDA is available it picks `float16` automatically;
otherwise it falls back to `int8` on CPU.

## Step

`whisper.transcribe` — runs after the `output` step in a flow.

## Manifest dependencies

- `runtimes = ["python3", "ffprobe"]` — both must be on `$PATH` at install time.
- `deps = "pip install --target ./libs faster-whisper nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"`
  — runs at install + boot. The `--target ./libs` keeps wheels inside
  the plugin directory so the host's global Python isn't touched.
- The CUDA libs are installed regardless of host hardware (~300MB on
  disk). They're loaded only if `device="auto"` finds a CUDA device,
  so CPU-only hosts pay disk but no runtime cost.

## Config (per-step)

| Key | Default | Description |
|---|---|---|
| `model` | `large-v3-turbo` | Any faster-whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `language` | `auto` | ISO 639-1 code (e.g. `en`, `ja`); `auto` lets Whisper detect |
| `skip_if_exists` | `true` | If a sidecar already exists for this file, skip without transcribing |
| `compute_type` | `auto` | CTranslate2 compute type. `auto` → `float16` on GPU, `int8` on CPU |

## Output

- A new file `<basename>.<lang>.srt` next to the input — Plex/Jellyfin
  pick this up automatically.
- `ctx.steps.whisper` is populated with:

  ```json
  {
    "subtitle_path": "/data/movies/Movie.en.srt",
    "language": "en",
    "model": "large-v3-turbo",
    "duration_sec": 42.7
  }
  ```

  Use `{{ steps.whisper.subtitle_path }}` in notify templates or in
  later flow steps.

## Behavior on edge cases

| Case | What happens |
|---|---|
| Input file missing | Step fails with "file does not exist" |
| Input has no audio stream | Step succeeds (no-op), no `.srt` written |
| Sidecar already exists and `skip_if_exists=true` | Step succeeds (no-op), no work done |
| Audio is fully silent (no speech) | Step succeeds (no-op), no empty `.srt` |
| `ffprobe` missing | Step fails — but the install-time `runtimes` check should prevent this from reaching production |

## Example flow snippet

```yaml
- step: output
- step: whisper.transcribe
  config:
    model: small         # smaller model for faster runs
    language: en         # skip autodetect
- step: notify
  config:
    body: |
      Transcoded {{ file.path }}.
      Subtitles: {{ steps.whisper.subtitle_path }}
```
```

- [ ] **Step 2: Verify the file renders cleanly**

```bash
head -20 whisper/README.md
```

Expected: starts with `# whisper`, no leftover stub text.

- [ ] **Step 3: Commit**

```bash
git add whisper/README.md
git commit -m "whisper: full plugin README"
```

---

### Task 11: Local sanity check (no commit)

Confirms the plugin is publishable end-to-end against the real catalog scripts. Runs `publish.py` against the new plugin, verifies the tarball + index update, then reverts so the actual publish happens via the workflow.

**Files:** none modified durably.

- [ ] **Step 1: Confirm baseline is clean**

`git status`
Expected: working tree clean, on branch `feat/whisper-plugin`.

- [ ] **Step 2: Run the publish script**

```bash
python3 scripts/publish.py whisper
```

Expected: prints `whisper: (new) -> 0.1.0 (<sha256>)` and exits 0. Creates `tarballs/whisper-0.1.0.tar.gz` and adds an entry to `index.json`.

- [ ] **Step 3: Inspect the resulting index entry**

```bash
python3 -c "
import json
e = next(p for p in json.load(open('index.json'))['plugins'] if p['name'] == 'whisper')
print('version:', e['version'])
print('runtimes:', e['runtimes'])
print('deps:', e['deps'])
print('sha:', e['tarball_sha256'])
"
```

Expected:
- `version: 0.1.0`
- `runtimes: ['python3', 'ffprobe']`
- `deps:` the pip install line from the manifest
- `sha:` matches `shasum -a 256 tarballs/whisper-0.1.0.tar.gz`

- [ ] **Step 4: Sanity check the tarball contents**

```bash
tar -tzf tarballs/whisper-0.1.0.tar.gz | sort
```

Expected output (in any order, but these exact paths must be present and nothing test-related):

```
whisper/
whisper/README.md
whisper/bin/
whisper/bin/run
whisper/manifest.toml
whisper/plugin.py
whisper/schema.json
```

`tests/test_whisper_plugin.py` must NOT be in the listing — that's why we kept the test file at the repo root.

- [ ] **Step 5: Verify reproducibility**

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import publish, hashlib, pathlib
p = pathlib.Path('whisper')
a = publish.build_tarball_bytes(p)
b = publish.build_tarball_bytes(p)
print('identical:', a == b)
print('sha:', hashlib.sha256(a).hexdigest())
"
```

Expected: `identical: True` and a sha that matches the entry in `index.json`.

- [ ] **Step 6: Discard the local publish artifacts**

The actual publishing happens via the GitHub workflow once this branch lands. Discard local changes:

```bash
git checkout -- index.json tarballs/
rm -f tarballs/whisper-0.1.0.tar.gz
git status
```

Expected: working tree clean. The tarball is gone, `index.json` is back to what's on the branch.

---

### Task 12: Final test sweep before shipping

**Files:** none modified.

- [ ] **Step 1: Run the full test suite (both files)**

```bash
python3 -m unittest discover -v 2>&1 | tail -10
```

Expected: 87 tests total — 41 from `test_publish.py` + 46 from `test_whisper_plugin.py` — all pass.

If `discover` doesn't pick up both files automatically, run them explicitly:

```bash
python3 -m unittest tests.test_publish tests.test_whisper_plugin -v 2>&1 | tail -5
```

Same expected output.

- [ ] **Step 2: Confirm git state**

```bash
git status
git log --oneline main..HEAD
```

Expected:
- working tree clean
- 11 commits on the branch (one per task that committed):

```
whisper: full plugin README
whisper: implement transcribe + main protocol orchestration
whisper: add has_audio_stream via ffprobe
whisper: add stdout protocol writers
whisper: add resolve_compute_type with cuda probe
whisper: add parse_execute and ProtocolError
whisper: add find_existing_sidecar with lang-aware matching
whisper: add write_srt_atomically
whisper: add fmt_ts and format_srt
whisper: scaffold plugin (manifest, schema, README, stubs)
docs: spec for whisper subtitle plugin
```

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/whisper-plugin
gh pr create --base main --head feat/whisper-plugin --title "Add whisper subtitle plugin" --body "$(cat <<'EOF'
## Summary

- New `whisper/` plugin: `whisper.transcribe` step, runs after `output` and writes a sidecar `<basename>.<lang>.srt` via `faster-whisper`.
- Uses both new optional manifest fields: `runtimes = ["python3", "ffprobe"]` and `deps = "pip install --target ./libs faster-whisper nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"`.
- GPU-aware: `device="auto"` + `compute_type="auto"` resolves to `float16`+CUDA when available, else `int8`+CPU.
- 45 unit + integration tests covering pure helpers and the full stdin/stdout protocol with a stub `WhisperModel`.

Spec: `docs/superpowers/specs/2026-05-02-whisper-plugin-design.md`
Plan: `docs/superpowers/plans/2026-05-02-whisper-plugin.md`

## Test plan

- [x] `python3 -m unittest tests.test_publish tests.test_whisper_plugin -v` — 87/87 pass
- [x] Local `python3 scripts/publish.py whisper` produces a deterministic tarball and a valid index entry; reproducibility verified
- [ ] After merge: dispatch `Publish plugin` workflow with `plugin=whisper`, confirm a PR is opened with `whisper-0.1.0.tar.gz` and the new index entry
- [ ] After publish: install the plugin on a host, confirm `runtimes` PATH check passes, `deps` runs, and a flow with `whisper.transcribe` produces a `.srt`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
