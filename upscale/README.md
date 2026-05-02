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
