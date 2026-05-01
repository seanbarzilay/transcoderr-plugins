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
2. Build a tarball: `tar -czf tarballs/<name>-<version>.tar.gz <plugin-name>`.
3. Compute sha256: `shasum -a 256 tarballs/<name>-<version>.tar.gz`.
4. Add an entry to `index.json` with the tarball URL, sha256, and the
   step names the plugin provides.
5. Open a PR.

The minimum manifest:

```toml
name = "your-plugin"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["your.step.name"]
```

## Hosting your own catalog

You don't have to submit plugins here. transcoderr lets operators add
additional catalog URLs in **Plugins → Catalogs**, so an internal/private
catalog can serve its own `index.json` (HTTP basic / bearer auth header
optionally) alongside this one. The server merges entries from every
configured catalog into one Browse view.

## License

(Per-plugin licenses live inside each plugin directory.)
