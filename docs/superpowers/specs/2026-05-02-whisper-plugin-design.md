# Whisper subtitle plugin

Date: 2026-05-02

## Goal

A new plugin in this catalog that runs after the `output` step in a flow,
takes the transcoded file, and produces a sidecar `.srt` next to it
using `faster-whisper`. The first plugin in the catalog to exercise the
new optional manifest fields (`runtimes`, `deps`).

## Why

Existing `size-report` is shell-only and minimal. A Whisper plugin is the
natural showcase for the publishing pipeline because it needs both
language-runtime awareness (`runtimes = ["python3", "ffprobe"]` so the
server refuses to install on a host without them) and a per-plugin deps
install (`pip install --target ./libs faster-whisper …` so the wheels
land in the plugin's own directory rather than the global site-packages).

## Plugin layout

```
whisper/
├── manifest.toml
├── bin/run             # Python entrypoint, executable bit set
├── schema.json         # per-step config schema
└── README.md
```

## Manifest

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

- `runtimes`: PR #60 enforces both at install. `ffprobe` is part of
  ffmpeg and effectively always present on a transcoderr host, but
  declaring it is honest.
- `deps`: PR #63 runs this via `/bin/sh -c` from the plugin directory at
  install + boot. `--target ./libs` keeps the install self-contained;
  the entrypoint adds `./libs` to `sys.path` at startup.
- CUDA libs are included in `deps` regardless of host hardware. CPU-only
  hosts pay ~300MB of disk for libs they never load — much less than a
  full torch+CUDA stack. The plugin's runtime device choice
  (`device="auto"` in faster-whisper) decides at run time.
- `min_transcoderr_version = "0.27.0"` because the plugin requires both
  the `runtimes` and `deps` fields, which land in 0.27.

## Step semantics

One step name: `whisper.transcribe`. Designed to run after `output` in a
flow.

**Inputs from `ctx`:**

- `ctx.file.path` — the post-transcode output file.

**Per-step config** (validated by `schema.json`, all optional with
defaults):

| Key | Default | Meaning |
|---|---|---|
| `model` | `"large-v3-turbo"` | faster-whisper model name (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`) |
| `language` | `"auto"` | ISO 639-1 code (`"en"`, `"ja"`, …) or `"auto"` for Whisper's autodetect |
| `skip_if_exists` | `true` | If a sidecar SRT for this file already exists, skip without transcribing |
| `compute_type` | `"auto"` | CTranslate2 compute type. `"auto"` resolves to `"float16"` on CUDA, `"int8"` on CPU |

**Sidecar path.** Plex/Jellyfin convention is `<basename>.<lang>.srt`,
so `Movie.mkv` becomes `Movie.en.srt`. Two cases:

- `language` set to a specific code → sidecar path is known up-front.
  `skip_if_exists` checks that exact file.
- `language: "auto"` → final path isn't known until faster-whisper
  reports the detected language. `skip_if_exists` does a glob match for
  any `<basename>.*.srt` already on disk and skips if any exists.

**Behavior:**

1. Parse `step_id`, `ctx.file.path`, and the per-step config from the
   `execute` JSON-RPC line. Apply defaults for any missing config keys.
2. Validate `step_id == "whisper.transcribe"`; otherwise
   `result error: "unknown step_id"`.
3. If `ctx.file.path` doesn't exist on disk, `result error: "file does
   not exist"`.
4. Probe the file for an audio stream via
   `ffprobe -v error -select_streams a:0 -show_entries stream=index
   -of json <path>`. If the response has no audio stream, log `"no
   audio stream, skipping"` and `result ok` without ctx.
5. If `skip_if_exists` and a matching sidecar exists, log `"sidecar
   already exists, skipping"` and `result ok` without ctx.
6. Lazy-load the model:
   `WhisperModel(model, device="auto", compute_type=resolved_compute_type)`.
   Resolve `compute_type="auto"` to `"float16"` if CUDA is available,
   else `"int8"`.
7. Transcribe:
   `segments, info = model.transcribe(file_path,
   language=lang_or_None, vad_filter=True)`.
   - `vad_filter=True` is the recommended setting for long-form audio:
     it gates the model on a voice-activity detector so silence stretches
     don't get filled with hallucinated text.
   - `language=None` triggers Whisper's autodetect; an ISO code skips
     detection.
8. Stream segments into an SRT-formatted string (see SRT format
   below). Write to `<basename>.<info.language>.srt.tmp`, then
   `os.replace` to the final path. (Atomic on POSIX when source and
   destination are in the same directory.)
9. If the model returns zero segments (e.g. fully-silent audio), do
   **not** write an empty `.srt`. Log `"no speech detected"` and
   `result ok` without ctx.
10. Emit `context_set` for key `whisper`:
    ```json
    {
      "subtitle_path": "<absolute path to the .srt>",
      "language": "<info.language>",
      "model": "<model name actually used>",
      "duration_sec": <float, wall-clock seconds spent transcribing>
    }
    ```
11. `result ok`.

`{{ steps.whisper.subtitle_path }}` then becomes available to subsequent
steps and notify templates.

## Plugin entrypoint (`bin/run`)

Python file with `#!/usr/bin/env python3` shebang, executable mode bit
set in the tarball (the catalog's deterministic-tarball builder
preserves the executable bit, so `chmod +x` in the source tree is the
durable signal).

Skeleton:

```python
#!/usr/bin/env python3
"""whisper.transcribe — sidecar .srt via faster-whisper."""
import json
import os
import sys
import time
from pathlib import Path

# Make per-plugin deps installed by `deps` in manifest.toml importable.
HERE = Path(__file__).resolve().parent.parent  # plugin root
sys.path.insert(0, str(HERE / "libs"))

from faster_whisper import WhisperModel  # noqa: E402
```

The protocol mirrors `size-report` exactly:

- Read two JSON-RPC lines from stdin: `init`, then `execute`.
- Emit JSON event lines to stdout: any number of `log` and
  `context_set` events, then exactly one terminating `result` line, then
  exit 0.

The script is split into helpers:

- `emit_log(msg)`, `emit_context_set(key, value)`,
  `emit_result_ok()`, `emit_result_err(msg)` — JSON line writers.
- `parse_execute(line)` — pulls `step_id`, `file_path`, and `config`
  out of the execute JSON.
- `has_audio_stream(file_path)` — runs ffprobe and returns bool.
- `find_existing_sidecar(file_path, lang_or_none)` — implements the
  language-aware skip check (exact path or glob).
- `resolve_compute_type(user_value)` — maps `"auto"` to the right
  CTranslate2 type given device availability.
- `format_srt(segments)` — joins `Segment` objects into the full SRT
  text.
- `write_srt_atomically(path, srt_text)` — `.tmp` write + `os.replace`.
- `transcribe(...)` — the main work; returns the populated ctx dict
  or `None` for benign-skip cases.

`main()` orchestrates: protocol read → step-id check → file/audio gates
→ skip-if-exists check → transcribe → write → context_set → result.

## SRT format

Pure stdlib formatter. faster-whisper yields `Segment` namedtuples with
float `start`, float `end`, and str `text`. SRT cue format:

```
<1-based index>
HH:MM:SS,mmm --> HH:MM:SS,mmm
<text>

```

Timestamp formatter:

```python
def fmt_ts(secs: float) -> str:
    ms = int(round(secs * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

Cue separator is a blank line; trailing newline at end of file.

## Schema

`schema.json`:

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

## Edge cases

| Case | Behavior |
|---|---|
| File missing on disk | `result error: "file does not exist"` |
| File has no audio stream | log + `result ok`, no `.srt`, no ctx |
| `skip_if_exists` + sidecar present | log + `result ok`, no work, no ctx |
| `ffprobe` not on PATH at runtime | `result error: "ffprobe not on PATH"` (belt-and-braces — `runtimes` install-time check should already prevent this) |
| Model download fails | `result error: <stderr from faster-whisper>` |
| OOM loading model | `result error: "OOM loading model <name>; try a smaller model or compute_type=int8"` |
| Empty transcription (silent audio) | log `"no speech detected"`, `result ok`, no `.srt`, no ctx |
| Unknown step ID | `result error: "unknown step_id"` |

## Testing strategy

The plugin can't run faster-whisper end-to-end in CI (no GPU, no model
weights). Test what we can:

1. **`format_srt` + `fmt_ts`** — pure-function unit tests with synthetic
   `Segment`-like tuples. Round-trip a few cues, verify exact byte
   output (timestamps, cue numbering, blank-line separators, final
   newline).
2. **`find_existing_sidecar`** — tmpdir tests covering
   exact-path-with-language and glob-when-auto. Both true and false
   cases.
3. **`parse_execute`** — sample JSON-RPC payloads (with and without
   `config` set), verify defaults are applied correctly.
4. **`write_srt_atomically`** — verify the `.tmp` is gone after success
   and the final file has the exact bytes.
5. **End-to-end with a fake `WhisperModel`** — replace the import with a
   stub fixture that yields known segments. Verify the full
   stdin/stdout protocol: feed init+execute, capture stdout, assert the
   stream of `log`/`context_set`/`result` lines.

Tests live in `tests/test_whisper_plugin.py` at the repo root — **not**
inside `whisper/`. The deterministic tarball builder includes everything
under the plugin directory, so tests inside it would ship to consumers
unnecessarily. Keeping them at the repo root mirrors how the
catalog-level `tests/test_publish.py` is laid out, and the existing
sys.path bootstrap pattern (`sys.path.insert(0, .../bin)` for the
plugin entrypoint) translates cleanly.

Stdlib `unittest`, no extra deps. The `tests/` directory is already at
the repo root, so adding another file there fits the existing layout.
Catalog-level tests in `tests/test_publish.py` keep exercising the
publishing pipeline; they don't need to know about this plugin.

## Out of scope (v0.1.0)

- Embedding the generated `.srt` into the container as a subtitle
  stream. Use a follow-up step or a separate plugin.
- Translating or transcribing to a different language than detected.
- Custom prompt engineering / initial-prompt config.
- Resuming a partial transcription.
- Multiple-language outputs from a single run (e.g. detect + translate
  to English in one pass).
- Using whisper's word-level timestamps. The current SRT cues are
  per-segment, which is the standard granularity for human-readable
  subtitles.

## Publishing

Once the plugin is in `whisper/`, the existing publish-plugin workflow
handles tarball + index update:

1. Bump version on a branch (or land the initial 0.1.0 directly on
   `main`).
2. Run **Actions → Publish plugin** with `plugin=whisper`.
3. Workflow opens a PR adding `tarballs/whisper-0.1.0.tar.gz` and the
   `index.json` entry.

The plugin's `runtimes` and `deps` flow into `index.json` automatically
since both manifest fields are now passed through by `build_entry`.
