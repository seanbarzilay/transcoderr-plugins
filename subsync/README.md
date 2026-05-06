# subsync

`subsync.align` — re-times an out-of-sync `.srt` against the audio of a
video file using [ffsubsync](https://github.com/smacke/ffsubsync) (VAD-
based speech-onset detection). Designed to drop in after
`whisper.transcribe` to clean up timing drift in the whisper-generated
sidecar.

## Known limitation: Python version

ffsubsync transitively depends on `webrtcvad`, which ships only an
sdist on PyPI — no wheels. The transcoderr container doesn't include
gcc, so a naive `pip install ffsubsync` fails at build time with
`x86_64-linux-gnu-gcc: No such file or directory`. The manifest
works around this by installing the `webrtcvad-wheels` fork (same
`import webrtcvad` API, prebuilt binaries) first, then pulling
ffsubsync with `--no-deps`.

`webrtcvad-wheels` ships wheels for cp310..cp313 on glibc + musl
linux (x86_64 / aarch64 / ppc64le / i686), macOS (intel + arm64),
and Windows. Python 3.14 has no wheel yet — operators on 3.14 will
need to either drop back to 3.13 or `apt install gcc` in the
container.

## Flow snippet

```yaml
- id: transcribe
  use: whisper.transcribe
- id: sync
  use: subsync.align
```

The `transcribe` step writes a sidecar; `sync` re-times its cues
against the audio in place. The synced sidecar lands beside the video
as `<basename>.<lang>.srt` (the same path whisper wrote it to).

## Configuration

```yaml
- id: sync
  use: subsync.align
  with:
    # All keys optional; defaults shown.
    subtitle_path: ""             # Templated override; empty = auto-discover.
    max_offset_seconds: 60        # Refuse to apply offsets above this.
    framerate_correction: true    # Allow ffsubsync to also fit a linear scale.
    fail_on_no_match: false       # Hard-fail vs. warn-and-pass.
```

Auto-discovery walks `ctx.steps.*` for the first entry with a
`subtitle_path` field. This means `subsync.align` works after
`whisper.transcribe` without the operator wiring an explicit reference.

## Output

On a successful sync, the plugin emits a `context_set` event with key
`subsync` and the computed metadata:

```json
{
  "subtitle_path": "/movies/X/X.en.srt",
  "offset_seconds": 1.23,
  "framerate_corrected": false
}
```

Notify templates can reference these fields, e.g.
`{{ steps.sync.offset_seconds }}`.

## Failure modes

| Condition | Result | Notes |
|---|---|---|
| Subtitle file not found by any priority | `ok` (no-op) | Logged warning. |
| Video file not found | `error` | Real flow problem. |
| ffsubsync exits non-zero | `error` if `fail_on_no_match: true`, else `ok` (warn + leave original) | Default warns and passes through. |
| Offset exceeds `max_offset_seconds` | `ok` (warn + leave original) | Sanity check — large offsets usually mean ffsubsync misidentified the speech. |

## Runtime requirements

- `python3` (the per-plugin venv installs ffsubsync via pip)
- `ffmpeg` on the host's `$PATH` (ffsubsync uses it for audio extraction)

The plugin runs on the coordinator only — it's CPU-bound and finishes
in 5–15s for a 90-min file, so the operational simplicity of skipping
plugin-push to remote workers outweighs any locality benefit.
