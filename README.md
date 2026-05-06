# transcoderr-plugins

Official plugin catalog for [transcoderr](https://github.com/seanbarzilay/transcoderr).

A transcoderr server fetches `index.json` from this repo and surfaces the
listed plugins under **Plugins → Browse** in the web UI. Click **Install**
and the server downloads the tarball, verifies its sha256, atomically
swaps it into `{data_dir}/plugins/<name>/`, and live-rebuilds the in-memory
step registry — no restart needed.

## What's here

```
.
├── index.json              # the catalog the server fetches
├── tarballs/               # pre-built plugin tarballs (sha256-pinned in index.json)
│   ├── size-report-0.1.2.tar.gz
│   ├── upscale-0.1.0.tar.gz
│   └── whisper-0.1.4.tar.gz
├── size-report/            # plugin source (mirror of what's in the tarball)
│   ├── manifest.toml
│   ├── bin/run
│   ├── schema.json
│   └── README.md
├── subsync/                # Python plugin: re-time .srt to audio via ffsubsync
│   ├── manifest.toml
│   ├── bin/run
│   ├── plugin.py
│   ├── schema.json
│   └── README.md
├── upscale/                # Python plugin: AI upscaling via Real-ESRGAN (ncnn-vulkan)
│   ├── manifest.toml
│   ├── bin/run
│   ├── plugin.py
│   ├── schema.json
│   └── README.md
├── whisper/                # Python plugin: sidecar .srt via faster-whisper
│   ├── manifest.toml
│   ├── bin/run
│   ├── plugin.py
│   ├── schema.json
│   └── README.md
└── whisperx/               # Python plugin: forced phoneme alignment via WhisperX (wav2vec2)
    ├── manifest.toml
    ├── bin/run
    ├── plugin.py
    ├── schema.json
    └── README.md
```

## Plugins

### [`size-report`](size-report/)

Records the input and output size of every transcoded file plus the
compression ratio, so notify templates can render lines like:

```
✓ /mnt/movies/Foo (2024)/Foo.mkv — saved 38.4% (12433551104 → 7659011840)
```

Pure POSIX shell + `awk` + `wc`. Provides two step names:
`size.report.before` (run early) and `size.report.after` (run after `output`).

### [`subsync`](subsync/)

Re-times an out-of-sync `.srt` against the audio of a video file using
[ffsubsync](https://github.com/smacke/ffsubsync) (VAD-based). Designed
to drop in after `whisper.transcribe` to clean up timing drift in the
whisper-generated sidecar. Provides one step: `subsync.align`.

### [`upscale`](upscale/)

AI-upscales the video stream with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
via the [`realesrgan-ncnn-vulkan`](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
binary. Designed for SD-source upgrades — DVD rips, VHS, anything well
below 720p that the operator wants at clean 1080p. Vulkan-based, so it
runs on AMD, Intel, NVIDIA, and Apple Silicon GPUs without a CUDA stack.

Provides one step: `upscale.video` (run after `output`). Declares
`runtimes = ["python3", "ffmpeg", "ffprobe", "realesrgan-ncnn-vulkan"]`.
Per-step config lets you pick the model, target height, and gate by
source resolution.

### [`whisper`](whisper/)

Generates a sidecar `<basename>.<lang>.srt` for the post-transcode
output file using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). GPU is
auto-detected (`float16` on CUDA, `int8` on CPU). Provides one step:
`whisper.transcribe` (run after `output`).

Declares `runtimes = ["python3", "ffprobe"]` and a `deps` line that
pip-installs faster-whisper + CUDA libs into the plugin's own directory
at install + boot. Per-step config lets you pick the model
(`large-v3-turbo` by default), pin a language, or skip when a sidecar
already exists.

### [`whisperx`](whisperx/)

Forced phoneme alignment for whisper-generated subtitles via
[WhisperX](https://github.com/m-bain/whisperX) (wav2vec2 + faster-whisper).
Produces per-word-accurate timestamps, much tighter than whisper's
own per-cue inference. Provides two steps: `whisperx.align` (drops in
after `whisper.transcribe`) and `whisperx.transcribe_aligned` (full
pipeline standalone). GPU-heavy; remote-eligible.

## Adding a plugin

1. Drop a directory under the repo root: `<plugin-name>/manifest.toml`
   plus `bin/run` (or whatever the manifest's `entrypoint` points at).
2. Fill in the manifest. Required fields:

   ```toml
   name = "your-plugin"
   version = "0.1.0"
   kind = "subprocess"
   entrypoint = "bin/run"
   provides_steps = ["your.step.name"]
   summary = "One-line description shown in the catalog."
   min_transcoderr_version = "0.19.0"
   ```

   Optional: declare `runtimes` if your `bin/run` shells out to anything
   beyond POSIX shell + coreutils. The transcoderr server checks each
   listed executable is on `$PATH` before allowing install:

   ```toml
   runtimes = ["python3"]      # or ["node"], ["bash"], etc.
   ```

   Omit or leave empty if the plugin only needs `sh`/`awk`/`wc`-style
   tools that every supported transcoderr image already ships.

   Optional: declare `deps` to run a setup command at install time and
   on every server boot — typically to fetch language-level dependencies
   into the plugin's own directory:

   ```toml
   deps = "pip install --target ./libs -r requirements.txt"
   ```

   Runs via `/bin/sh -c` from the plugin's directory. A non-zero exit at
   install fails the install (with rollback). Prefer install-into-plugin
   patterns (`pip install --target ./libs`, `npm install` writing to
   local `node_modules`) so dependencies don't leak outside the plugin.

3. Run the **Publish plugin** workflow (Actions → Publish plugin → Run
   workflow), with the plugin directory name as the input. The workflow
   builds a deterministic tarball, updates `index.json`, and opens a PR
   for review.

To re-publish an existing plugin, bump `version` in `manifest.toml` and
run the workflow again. The script refuses to re-use an already-listed
version.

You can also run the script locally to validate before opening a PR
(requires Python 3.11+ for `tomllib`):

```bash
python3 scripts/publish.py <plugin-name>
```

## Hosting your own catalog

You don't have to submit plugins here. transcoderr lets operators add
additional catalog URLs in **Plugins → Catalogs**, so an internal/private
catalog can serve its own `index.json` (HTTP basic / bearer auth header
optionally) alongside this one. The server merges entries from every
configured catalog into one Browse view.

## License

(Per-plugin licenses live inside each plugin directory.)
