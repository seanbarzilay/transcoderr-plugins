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
