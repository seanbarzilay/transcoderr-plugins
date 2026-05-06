# whisperx

Forced phoneme alignment for whisper-generated subtitles via
[WhisperX](https://github.com/m-bain/whisperX). Provides two steps:

- **`whisperx.align`** — drop-in after `whisper.transcribe`. Reads the
  existing `.srt` and the audio, runs wav2vec2 forced alignment to
  produce per-word-accurate timestamps, and rewrites the `.srt` in
  place. Mirrors the chaining model of the `subsync` plugin (which
  handles global drift) but with much higher precision.
- **`whisperx.transcribe_aligned`** — full pipeline standalone.
  Doesn't need whisper to have run first; runs faster-whisper
  internally for transcription and wav2vec2 for alignment in one
  shot. Writes a fresh `<basename>.<lang>.srt` next to the video.

## Known limitation: GPU strongly recommended

WhisperX is GPU-heavy. CPU fallback works (`compute_type=int8`) but
takes 5-10× longer than GPU. On a Tesla P4 (8GB VRAM) the alignment
step for a 90-minute file takes ~2-5 minutes; on CPU the same step
takes 15-30 minutes. Both step names declare `executor = "any-worker"`
so the dispatcher will route them to GPU-equipped remote workers.

## Known limitation: install footprint

The `deps` line pulls torch / torchaudio / transformers /
faster-whisper transitively via `whisperx`. Total install footprint
is ~5–7 GB including model weights downloaded on first use. First
install takes 3–5 minutes.

## Known limitation: GPU architecture

The plugin pins `torch==2.3.1` because torch 2.4+ dropped Pascal
(sm_61) — Tesla P4, GTX 10-series, and Quadro P-series cards otherwise
fail at wav2vec2 forward time with `"CUDA error: no kernel image is
available for execution on the device"`. Volta and newer (sm_70+) work
on either pin, so the 2.3.1 ceiling is the broadest-compatibility
choice. If you need a newer torch (e.g., for Hopper-only features) and
have a sm_70+ GPU, edit the manifest's `deps` line and rebuild the
plugin venv.

This pin chain has knock-on effects:

- `whisperx==3.1.6` (the last release whose torch floor is loose enough
  to coexist with the 2.3.x line) hard-pins `pyannote.audio==3.1.1`,
  and that pyannote release calls `torchaudio.set_audio_backend(...)`
  at import time — an API removed in torchaudio 2.1. The manifest
  installs `pyannote.audio==3.3.2` first (the release that dropped
  the call) and then installs whisperx with `--no-deps` to bypass
  the hard pin.
- `faster-whisper==1.0.0` pulls `av>=10` (PyAV). Recent `av` releases
  drop wheels for some Python/arch combos and fall back to building
  from source, which fails with `"pkg-config is required for building
  PyAV"` unless `pkg-config` and the `libav*-dev` headers are present
  on the host. The manifest pins `av==12.3.0` (broad wheel coverage:
  cp38..cp312 × linux x86_64/aarch64/i686 + macOS + Windows) and
  installs with `--only-binary=av` so the install fails fast on
  unsupported Python versions instead of hanging on a confusing
  build-from-source error.

If you upgrade past torch 2.3.x, you can drop the `--no-deps`
workaround and let pip resolve a fresh whisperx + pyannote chain.

## Flow snippets

After `whisper.transcribe` (most common):

```yaml
- id: transcribe
  use: whisper.transcribe
- id: align
  use: whisperx.align
```

The `transcribe` step writes a sidecar; `align` re-times its cues
in place. Auto-discovers the `.srt` and the language from the whisper
plugin's `context_set` output.

Standalone (replaces `whisper.transcribe`):

```yaml
- id: transcribe-aligned
  use: whisperx.transcribe_aligned
```

## Configuration

### `whisperx.align`

```yaml
- id: align
  use: whisperx.align
  with:
    # All keys optional; defaults shown.
    subtitle_path: ""           # Templated override; empty = auto-discover.
    language: ""                # Override; empty = ctx.steps walk → filename → "en".
    alignment_model: ""         # Override wav2vec2 model name (HuggingFace ID).
    fail_on_no_match: false     # Hard-fail vs. warn-and-pass when no .srt found.
```

### `whisperx.transcribe_aligned`

```yaml
- id: transcribe-aligned
  use: whisperx.transcribe_aligned
  with:
    # All keys optional; defaults shown.
    model: "large-v3-turbo"     # whisper transcription model.
    language: "auto"            # whisper language detection ("auto" or ISO code).
    skip_if_exists: true        # Skip if a sidecar .srt already exists.
    compute_type: "auto"        # auto | float16 | int8 | float32
    batch_size: 16              # whisper transcription batch size.
    alignment_model: ""         # Override wav2vec2 model.
```

## Output

On success, both steps emit a `context_set` event with key `whisperx`:

```json
{
  "subtitle_path": "/movies/X/X.en.srt",
  "language": "en",
  "alignment_model": "torchaudio",
  "n_words": 1234,
  "duration_sec": 45.6
}
```

`whisperx.transcribe_aligned` adds two extra fields:
`transcription_model` and `n_segments`. Notify templates can reference
`{{ steps.align.duration_sec }}`, etc.

## Failure modes

| Condition | Result | Notes |
|---|---|---|
| `whisperx.align`: no .srt found | `ok` (no-op + log) by default; `error` if `fail_on_no_match: true` | Same default as subsync. |
| `whisperx.transcribe_aligned`: sidecar exists | `ok` (no-op + log) | Skip-if-exists default true. |
| Video has no audio stream | `ok` (no-op + log) | Same as whisper. |
| Wav2vec2 / whisper model download fails | `error` | Network problem on first install. |
| Language not supported by WhisperX | `error` | WhisperX ships a per-language model registry; unknown languages fail loudly. |
| GPU OOM | `error` | Dial down `batch_size` (transcribe_aligned only) or run on a worker with more VRAM. |
| ffmpeg/ffprobe not on PATH | `error` | WhisperX uses ffmpeg internally for audio extraction. |

## Runtime requirements

- `python3` (3.11–3.13 supported; 3.14 currently broken because
  faster-whisper transitively pulls webrtcvad, which still imports
  the removed `pkg_resources`).
- `ffmpeg` on the host's `$PATH`.
- A CUDA-capable GPU is strongly recommended; CPU fallback works
  but is slow.

## Coexistence with whisper and subsync

- **whisper.transcribe → whisperx.align:** the canonical chain. whisper
  produces a coarse-but-fast .srt; whisperx.align refines the timing
  with per-word accuracy.
- **whisper.transcribe → subsync.align → whisperx.align:** belt and
  braces. subsync handles global drift first (cheap), then whisperx
  refines per-cue (expensive). The outputs compose cleanly because
  subsync rewrites the .srt before whisperx reads it.
- **whisperx.transcribe_aligned alone:** the simplest setup, but uses
  more GPU than whisper alone (loads two models: faster-whisper plus
  wav2vec2). Good for fresh installs without an existing whisper flow.
