# Publish-plugin GitHub workflow

Date: 2026-05-01

## Goal

Add a manual GitHub Actions workflow that re-publishes a plugin from this
catalog repo. The user enters a plugin directory name; the workflow
rebuilds a deterministic tarball, updates `index.json`, deletes the prior
tarball, and opens a pull request with the change. Works for both
already-listed plugins (update) and brand-new ones (first publish).

## Why a workflow

Today the README documents a manual sequence: `tar -czf …`, `shasum -a
256`, hand-edit `index.json`, open a PR. That is bug-prone (sha mismatch,
forgetting to bump `min_transcoderr_version`, leaving stale tarballs)
and version-conflict-prone if two contributors land overlapping edits.
A scripted publish removes all of that.

## Inputs

A single `workflow_dispatch` input:

- `plugin` — the plugin's directory name at the repo root (e.g.
  `size-report`). Required.

No version input. Version is read from `manifest.toml`.

## Files added / changed

- `.github/workflows/publish-plugin.yml` — new, thin workflow.
- `scripts/publish.py` — new, all real logic; runnable locally as
  `python3 scripts/publish.py <plugin>`.
- `<plugin>/manifest.toml` — extended with two new fields (`summary`,
  `min_transcoderr_version`). Existing `size-report/manifest.toml` is
  updated as a separate prep commit before the workflow lands, so the
  workflow PR is purely additive.
- `README.md` — replace the "Adding a plugin" steps with: edit
  `manifest.toml`, then run the workflow.

No new dependencies. Script uses only Python stdlib (`tomllib`,
`tarfile`, `gzip`, `hashlib`, `json`, `pathlib`, `argparse`,
`subprocess`).

## Manifest schema (extended)

```toml
name = "size-report"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["size.report.before", "size.report.after"]
summary = "Records before/after byte counts and the compression ratio so notify templates can render saved-percentage stats."
min_transcoderr_version = "0.19.0"
```

`summary` and `min_transcoderr_version` are required by the publish
script. `homepage` and `tarball_url` are not in the manifest — they are
derived from the repo's git origin URL.

## `index.json` field mapping

For each plugin entry the script writes:

| Index field | Source |
|---|---|
| `name` | `manifest.name` |
| `version` | `manifest.version` |
| `summary` | `manifest.summary` |
| `tarball_url` | `https://raw.githubusercontent.com/<owner>/<repo>/main/tarballs/<name>-<version>.tar.gz` |
| `tarball_sha256` | sha256 of the tarball bytes (lowercase hex) |
| `homepage` | `https://github.com/<owner>/<repo>/tree/main/<name>` |
| `min_transcoderr_version` | `manifest.min_transcoderr_version` |
| `kind` | `manifest.kind` |
| `provides_steps` | `manifest.provides_steps` |

`<owner>/<repo>` is parsed from `git remote get-url origin`, supporting
both `git@github.com:owner/repo(.git)` and
`https://github.com/owner/repo(.git)` forms. This avoids hardcoding
`seanbarzilay/transcoderr-plugins` and keeps the workflow usable in
forks.

The top-level `schema_version`, `catalog_name`, and `catalog_url` keys
are preserved verbatim. Plugins are sorted by `name` on write for stable
diffs. The file is written with `indent=2` and a trailing newline.

## Script behavior

`python3 scripts/publish.py <plugin>` runs these steps in order:

1. Resolve repo root via `git rev-parse --show-toplevel`. Resolve plugin
   dir = `<root>/<plugin>`. Fail if it does not exist.
2. Parse `<plugin>/manifest.toml` with `tomllib`. Require all of:
   `name`, `version`, `kind`, `entrypoint`, `provides_steps`, `summary`,
   `min_transcoderr_version`. Fail if any are missing. Fail if
   `manifest.name` does not equal the input.
3. Read `<root>/index.json`. Look up the existing entry by `name`.
4. **Version-conflict gate.** If an entry exists and its `version`
   equals the manifest's `version`, exit non-zero with: *"<name>
   <version> is already published. Bump version in manifest.toml
   first."*
5. Build a deterministic tarball at
   `tarballs/<name>-<version>.tar.gz`. Compute sha256 of the bytes.
6. If a previous entry existed, delete its
   `tarballs/<name>-<old-version>.tar.gz` from disk. Silent no-op if the
   file is missing.
7. Build the new index entry per the field-mapping table above.
8. Write `index.json`: replace existing entry by `name`, or append if
   new. Sort plugins by `name`.
9. Print one summary line: `<plugin>: <old-version-or-new> -> <version>
   (<sha256>)`.

## Deterministic tarball

Use Python's `tarfile` and `gzip` modules directly so we do not depend
on the host's `tar` flag dialect (BSD vs GNU diverge on `--mtime`,
`--owner=0:0`, etc.).

Knobs (full implementation lives in `scripts/publish.py`):

- Walk the plugin dir with sorted entries (`sorted(plugin_dir.rglob("*"))`),
  filesystem-iteration order is not stable across runs.
- Per-entry `TarInfo` filter: `mtime=0`, `uid=0`, `gid=0`, `uname=""`,
  `gname=""`, mode normalized to `0o755` for executables and dirs,
  `0o644` otherwise.
- `tarfile.open(..., format=tarfile.USTAR_FORMAT)` to avoid PAX extended
  headers (which embed paths and extras non-deterministically).
- `gzip.GzipFile(fileobj=..., mode="wb", mtime=0, compresslevel=9)` so
  the gzip member header has no embedded timestamp.

Result: same source tree → identical bytes → identical sha256, regardless
of when or where the workflow runs.

## Workflow

`.github/workflows/publish-plugin.yml`:

```yaml
name: Publish plugin

on:
  workflow_dispatch:
    inputs:
      plugin:
        description: "Plugin directory name (e.g. size-report)"
        required: true
        type: string

permissions:
  contents: write
  pull-requests: write

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Run publish script
        run: python3 scripts/publish.py "${{ inputs.plugin }}"

      - name: Open PR
        uses: peter-evans/create-pull-request@v6
        with:
          branch: publish/${{ inputs.plugin }}
          title: "publish: ${{ inputs.plugin }}"
          body: |
            Automated publish run for `${{ inputs.plugin }}`.

            Triggered by @${{ github.actor }} via `workflow_dispatch`.
          commit-message: "publish: ${{ inputs.plugin }}"
          add-paths: |
            index.json
            tarballs/
```

Notes:

- `contents: write` and `pull-requests: write` are required for
  `peter-evans/create-pull-request` to push the branch and open the PR
  via the default `GITHUB_TOKEN`.
- Branch name `publish/<plugin>` — re-running for the same plugin
  updates the existing PR rather than creating a second one.
- `add-paths` keeps the PR scoped to catalog files only; nothing else
  the script touches accidentally lands in the PR.
- Python 3.12 picks up `tomllib` from the standard library (3.11+).
- A PR opened by `GITHUB_TOKEN` does *not* trigger downstream
  `pull_request` workflows. No CI exists today so this is fine; if CI is
  added later, switch to a PAT or `pull_request_target`.

## Error handling

The script exits non-zero with a clear, single-line message in these
cases:

1. Plugin dir does not exist → *"plugin directory `<name>/` not found"*.
2. `manifest.toml` missing or unparseable → propagate `tomllib` error
   prefixed with the path.
3. Required manifest fields missing → *"manifest.toml is missing
   required fields: summary, min_transcoderr_version"* (lists all
   missing fields).
4. `manifest.name` ≠ input plugin name → *"manifest.name 'foo' doesn't
   match plugin dir 'bar'"*.
5. Version already published → *"<name> <version> is already published.
   Bump version in manifest.toml first."*
6. Owner/repo not parseable from origin → *"unable to parse owner/repo
   from origin URL"*.

Silent / OK cases:

- Plugin not in `index.json` yet → first publish, append the entry.
- No prior tarball to delete on first publish → skip silently.
- Re-running the workflow for an in-progress publish PR →
  `create-pull-request` updates the existing branch / PR. The
  version-conflict gate does not trip because `index.json` on `main`
  has not changed.

## Out of scope

- CI on PRs (lint, schema-validate `index.json`).
- Auto-version-bump (Q4-C). Versions are bumped by the contributor.
- Removing a plugin from the catalog. That stays a manual PR.
- Cross-plugin batch publish (one plugin per workflow run is enough).
