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
│   └── size-report-0.1.0.tar.gz
└── size-report/            # plugin source (mirror of what's in the tarball)
    ├── manifest.toml
    ├── bin/run
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

3. Run the **Publish plugin** workflow (Actions → Publish plugin → Run
   workflow), with the plugin directory name as the input. The workflow
   builds a deterministic tarball, updates `index.json`, and opens a PR
   for review.

To re-publish an existing plugin, bump `version` in `manifest.toml` and
run the workflow again. The script refuses to re-use an already-listed
version.

You can also run the script locally to validate before opening a PR:

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
