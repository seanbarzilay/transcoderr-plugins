# Upscale plugin (Real-ESRGAN ncnn-vulkan)

Date: 2026-05-03

## Goal

A new plugin in this catalog called `upscale`. Step `upscale.video`
takes the input video file, runs `realesrgan-ncnn-vulkan` against it
(with a configurable model and scale), then re-muxes the upscaled video
stream with the original audio + subtitle streams. Designed for
SD-source upgrades — DVD rips, VHS, anything where the source is well
below 720p and the operator wants a clean 1080p output.

## Why

Existing flow steps cover encoding (`plan.video.encode`), audio
(`plan.audio.ensure`), container hygiene (`plan.streams.*`). What's
missing is a way to actually grow the resolution of small sources before
encoding. Doing this with a generic ffmpeg upscale (`scale=...:lanczos`)
is sharp but visibly artifact-y on grainy DVDs. Real-ESRGAN's
ncnn-vulkan build runs on any GPU (AMD, Intel, NVIDIA, Apple) without a
torch + CUDA stack, and it has a built-in video mode — making it the
sweet spot for a small, portable plugin.

## Plugin layout

```
upscale/
├── manifest.toml
├── bin/run             # tiny shell wrapper, executable
├── plugin.py           # all real logic, importable for tests
├── schema.json         # per-step config schema
└── README.md
```

## Manifest

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

No `deps` line — the plugin is pure stdlib Python; it orchestrates the
binaries (`ffprobe`, `ffmpeg`, `realesrgan-ncnn-vulkan`) via
`subprocess`. The runtimes check at install time fails fast if the
operator hasn't installed the realesrgan binary yet — same trust model
as ffmpeg.

## Step semantics

One step name: `upscale.video`. Inputs from `ctx`:

- `ctx.file.path` — the input video.

Per-step config (validated by `schema.json`, all optional with
defaults):

| Key | Default | Meaning |
|---|---|---|
| `model` | `"realesr-animevideov3"` | ncnn model name. Other valid: `realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-general-x4v3` (denoising-friendly for grainy DVDs) |
| `scale` | `4` | model scale factor (`2`, `3`, or `4` depending on model) |
| `target_height` | `1080` | final output height after the model. If the model's native output overshoots, the plugin lanczos-downscales to this. `0` disables the post-resize. |
| `min_source_height` | `720` | self-gate: if `probe.streams[0].height >= this`, skip with `result ok` and no work. Lets one flow handle mixed-resolution libraries. |
| `output_path` | `null` | absolute output path. `null` ⇒ produce a sibling `<basename>.upscaled.mkv` next to the input (always `.mkv` regardless of source extension, since the mux step always emits matroska). |
| `denoise_strength` | `0.5` | only respected by models that support it (`realesr-general-x4v3`); ignored by others. Forwarded as the binary's `-x` flag where applicable. |
| `tile_size` | `0` | `0` = auto. Smaller (e.g. `256`) trades speed for VRAM on small GPUs. |

Context written under key `upscale` on success:

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

`{{ steps.upscale.path }}` is then available to subsequent steps and
notify templates. The original file is not modified — the operator's
own `output: replace` step (or a manual cleanup) decides whether to
swap.

## Pipeline architecture

`realesrgan-ncnn-vulkan` does not have a clean raw-frame stdin/stdout
mode; it owns its own internal ffmpeg invocation when given a video
file. Trying to pipe raw frames around it fights the binary. Instead,
the plugin runs three sequential subprocesses:

1. **AI upscale (ncnn-vulkan).**
   ```
   realesrgan-ncnn-vulkan -i <input> -o <tmp/upscaled.mkv> -n <model> \
     -s <scale> [-t <tile_size>] [-f mkv]
   ```
   Output is a video-only file at the model's native scale (e.g. 480p ×
   4 = 1920p). No audio, no subs.

2. **Lanczos downscale (ffmpeg).** Optional. Only if `target_height >
   0` and the upscaled height differs:
   ```
   ffmpeg -i <tmp/upscaled.mkv> -vf "scale=-2:<target_height>:flags=lanczos" \
     -c:v libx264 -preset ultrafast -crf 18 <tmp/resized.mkv>
   ```
   The `-2` keeps width even and aspect-ratio-correct. The intermediate
   codec is hardcoded to `libx264 -preset ultrafast -crf 18` — the file
   is transient (lives in `/tmp/` for at most a few seconds before step
   3 copies its video stream by `-c copy`) so we optimize for fastest
   encode, not smallest file. The operator's downstream flow is what
   produces the final codec choice; this pass is just a clean
   intermediate so step 3 has a properly-sized stream to copy.

3. **Mux (ffmpeg).** Combine the upscaled video stream with the
   original audio + subtitle streams:
   ```
   ffmpeg -i <tmp/resized.mkv> -i <original_input> \
     -map 0:v:0 -map 1:a? -map 1:s? -c copy <output_path>
   ```
   No re-encoding here — copy the upscaled video and copy every audio
   + subtitle stream from the original input.

If `target_height == 0` (post-resize disabled), step 2 is skipped and
step 3 reads from `tmp/upscaled.mkv` directly.

The temp work dir lives under `/tmp/transcoderr-upscale-<pid>/` and is
removed on completion (success or failure) via a `try/finally`.

## Plugin runtime (`plugin.py`)

Top of file: stdlib imports only, no third-party deps. The protocol
mirrors whisper:

- Read 2 JSON-RPC lines from stdin (`init`, `execute`).
- Emit JSON event lines to stdout: `log` and `progress` and
  `context_set`, then a single terminating `result`.
- Exit 0 regardless of success/failure (failures surface in the
  `result` event payload, not exit code).

Functions:

- Stdout writers (`emit_log`, `emit_progress`, `emit_context_set`,
  `emit_result_ok`, `emit_result_err`) — JSON line writers with
  `separators=(",",":")` and an injectable `out` arg for tests.
- `parse_execute(line)` — pull `step_id`, `file_path`, and resolved
  config (defaults filled in).
- `probe_dimensions(path)` — runs ffprobe, returns `(width, height)`.
- `compute_target_height(source_h, source_w, target_h)` — returns the
  final `(width, height)` after model + downscale, ensuring even
  numbers for codec compatibility (matches the `-2` flag in step 2).
- `parse_progress_line(line)` — turns ncnn-vulkan stderr lines like
  `frame: 12345/123456` into a `(done, total)` tuple. Returns `None`
  for lines that don't match.
- `run_upscale_subprocess(input_path, output_path, model, scale, tile_size, stdout)`
  — invokes the ncnn-vulkan binary with `subprocess.Popen`, streams
  stderr into `parse_progress_line`, emits `progress` events as it
  goes.
- `run_downscale_subprocess(input_path, output_path, target_height)`
  — single ffmpeg invocation with the lanczos filter.
- `run_mux_subprocess(video_only_path, original_input, output_path)`
  — single ffmpeg invocation that maps streams and copies.
- `transcribe_video(file_path, config, *, stdout) -> dict | None` —
  the orchestrator. Returns the ctx dict on success, or `None` for the
  benign skip case (source already at or above `min_source_height`).
- `main(stdin, stdout)` — drives the protocol.

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
      "description": "Absolute output path. Null ⇒ sibling <basename>.upscaled.<ext>."
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

## Edge cases

| Case | Behavior |
|---|---|
| Input file missing | `result error: "file does not exist"` |
| `height >= min_source_height` | log `"source already at NNNp, skipping"`, `result ok`, no ctx |
| `realesrgan-ncnn-vulkan` not on PATH | `result error: "realesrgan-ncnn-vulkan not on PATH"` (install-time `runtimes` check should already prevent this from reaching production) |
| Model name unknown to ncnn-vulkan | binary returns non-zero with stderr `model not found`; propagate as `result error: "realesrgan failed: model not found"` |
| Tile size too small / OOM on GPU | binary returns non-zero with stderr `vkAllocateMemory failed`; `result error: "OOM during upscale; try a smaller tile_size"` |
| Output already exists at `output_path` | overwrite (matches `output: replace` semantics elsewhere in transcoderr) |
| Cancel mid-upscale | the engine's cancellation kills the realesrgan child; the temp work dir is cleaned up in the orchestrator's `try/finally` |

## Testing strategy

The plugin can't run realesrgan-ncnn-vulkan end-to-end in CI (no GPU,
no model weights, no test video files). Tests cover what we can
without binaries:

1. **Pure helpers** — `compute_target_height` (aspect ratio +
   even-number rounding), `parse_progress_line` (multiple ncnn-vulkan
   stderr formats, including malformed lines), `parse_execute`
   (config-defaults merging).
2. **Stdout writers** — same pattern as whisper: byte-exact check on
   one writer, JSON-decode roundtrip on the rest.
3. **Subprocess wrappers** — `mock.patch.object(plugin.subprocess,
   "run"/"Popen", ...)` for `ffprobe`/`ffmpeg`/`realesrgan-ncnn-vulkan`
   invocations. Assert argv shape; simulate non-zero exits and check
   the error mapping.
4. **End-to-end protocol** — stub all three subprocess wrappers,
   drive `main()` via stdin/stdout, assert: happy path emits log +
   progress + context_set + result_ok; missing-file emits result_err;
   self-gate (already HD) emits log + result_ok with no ctx; OOM
   stderr from upscale subprocess maps to result_err with
   "smaller tile_size" guidance.

Tests live in `tests/test_upscale_plugin.py` at the repo root (not
inside `upscale/`) so test code isn't shipped in the tarball — same
layout as whisper.

Stdlib `unittest`, no extra deps.

## Documentation

The README inside `upscale/` describes the manifest dependencies, the
config knobs, the ctx output, the edge cases, and a worked example
flow:

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

The catalog-level README's "Plugins" section gets a new `upscale`
entry once the plugin ships.

## Out of scope (v0.1.0)

- Frame interpolation (RIFE, IFRNet) — different plugin.
- Multi-GPU pipelining.
- In-pipeline integration with `plan.execute` (would require a custom
  ffmpeg build with DNN backends; spec'd as alternative #3 during
  brainstorming, rejected).
- Resuming a partial upscale.
- Anime-specific colorspace handling (the model handles BT.709 fine
  for live-action; anime models output what they get).
- Bundling the realesrgan binary in the plugin tarball — multi-arch
  fanout, trust issues, version-pinning headaches.

## Publishing

Once the plugin lives in `upscale/`, the existing publish-plugin
workflow handles tarball + index update:

1. Land the initial 0.1.0 directly on `main`.
2. Run **Actions → Publish plugin** with `plugin=upscale`.
3. Workflow opens a PR adding `tarballs/upscale-0.1.0.tar.gz` and the
   `index.json` entry (with `runtimes` flowing through automatically
   per the existing `build_entry` mapping).
