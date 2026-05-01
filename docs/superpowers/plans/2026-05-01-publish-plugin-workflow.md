# Publish-plugin Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual GitHub workflow that re-builds a plugin's tarball, updates `index.json`, and opens a PR — driven by a single Python script that's also runnable locally.

**Architecture:** Thin `workflow_dispatch` workflow → single Python entry script (`scripts/publish.py`) using only the standard library → `peter-evans/create-pull-request` to open the PR. Manifest fields (`summary`, `min_transcoderr_version`) become the source of truth for catalog metadata. Tarballs are bit-for-bit reproducible (fixed mtime/uid/gid, sorted entries, USTAR format, gzip mtime=0).

**Tech Stack:** Python 3.12 stdlib (`tomllib`, `tarfile`, `gzip`, `hashlib`, `json`, `subprocess`, `pathlib`, `argparse`, `re`), `unittest` for tests, GitHub Actions, `peter-evans/create-pull-request@v6`.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `size-report/manifest.toml` | Add `summary` + `min_transcoderr_version` (catalog metadata moves into the manifest as the single source of truth). | Modify |
| `scripts/publish.py` | All publish logic: parse manifest, build deterministic tarball, mutate `index.json`, delete old tarball. Importable for tests. | Create |
| `tests/test_publish.py` | Unit + integration tests for `scripts/publish.py`. Pure stdlib `unittest`, no install needed. | Create |
| `.github/workflows/publish-plugin.yml` | Thin `workflow_dispatch` runner: checkout → run script → open PR. | Create |
| `README.md` | Replace the manual "Adding a plugin" steps with the workflow approach. | Modify |

The script is split into small pure functions so each can be unit-tested in isolation; only `publish()` (the orchestrator) and `main()` touch the filesystem and `git`. Tests use a tmpdir-based fake repo (with `git init` + an origin remote) so they never touch the real working copy.

---

### Task 1: Add catalog fields to `size-report/manifest.toml`

This is a prep commit. The manifest gains the two fields that the publish script will read as the source of truth. Values mirror what's already in `index.json` so nothing user-visible changes.

**Files:**
- Modify: `size-report/manifest.toml`

- [ ] **Step 1: Read the current manifest**

Run: `cat size-report/manifest.toml`
Expected:
```
name = "size-report"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["size.report.before", "size.report.after"]
```

- [ ] **Step 2: Append the two new fields**

Edit `size-report/manifest.toml` so it reads:

```toml
name = "size-report"
version = "0.1.0"
kind = "subprocess"
entrypoint = "bin/run"
provides_steps = ["size.report.before", "size.report.after"]
summary = "Records before/after byte counts and the compression ratio so notify templates can render saved-percentage stats."
min_transcoderr_version = "0.19.0"
```

- [ ] **Step 3: Verify it parses**

Run: `python3 -c "import tomllib; print(tomllib.load(open('size-report/manifest.toml','rb')))"`
Expected: a dict with `summary` and `min_transcoderr_version` keys whose values match `index.json`.

- [ ] **Step 4: Commit**

```bash
git add size-report/manifest.toml
git commit -m "size-report: add summary and min_transcoderr_version to manifest"
```

---

### Task 2: Create `scripts/publish.py` skeleton + `parse_owner_repo` (TDD)

**Files:**
- Create: `scripts/publish.py`
- Create: `tests/test_publish.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish.py`:

```python
"""Unit + integration tests for scripts/publish.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish as pub  # noqa: E402


class ParseOwnerRepoTests(unittest.TestCase):
    def test_https_form(self):
        self.assertEqual(
            pub.parse_owner_repo("https://github.com/foo/bar.git"),
            ("foo", "bar"),
        )

    def test_https_form_no_dot_git(self):
        self.assertEqual(
            pub.parse_owner_repo("https://github.com/foo/bar"),
            ("foo", "bar"),
        )

    def test_ssh_form(self):
        self.assertEqual(
            pub.parse_owner_repo("git@github.com:foo/bar.git"),
            ("foo", "bar"),
        )

    def test_ssh_form_no_dot_git(self):
        self.assertEqual(
            pub.parse_owner_repo("git@github.com:foo/bar"),
            ("foo", "bar"),
        )

    def test_trailing_whitespace_tolerated(self):
        self.assertEqual(
            pub.parse_owner_repo("  https://github.com/foo/bar.git\n"),
            ("foo", "bar"),
        )

    def test_unparseable_raises(self):
        with self.assertRaises(pub.PublishError):
            pub.parse_owner_repo("not a url")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_publish -v`
Expected: `ModuleNotFoundError: No module named 'publish'` — the script doesn't exist yet.

- [ ] **Step 3: Create the skeleton script**

Create `scripts/publish.py`:

```python
#!/usr/bin/env python3
"""Publish a plugin from this catalog: pack tarball, update index.json."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

REQUIRED_MANIFEST_FIELDS = (
    "name",
    "version",
    "kind",
    "entrypoint",
    "provides_steps",
    "summary",
    "min_transcoderr_version",
)


class PublishError(Exception):
    """Raised for any condition that should fail the publish run."""


def parse_owner_repo(origin_url: str) -> tuple[str, str]:
    """Parse 'foo/bar' from a GitHub origin URL (https or ssh form)."""
    pattern = r"^(?:https://github\.com/|git@github\.com:)([^/]+)/(.+?)(?:\.git)?/?$"
    m = re.match(pattern, origin_url.strip())
    if not m:
        raise PublishError(
            f"unable to parse owner/repo from origin URL: {origin_url!r}"
        )
    return m.group(1), m.group(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", help="plugin directory name")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: scaffold script and parse_owner_repo helper"
```

---

### Task 3: `load_manifest` (TDD)

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_publish.py` (above `if __name__ == "__main__":`):

```python
import tempfile


def _write_manifest(plugin_dir: Path, **overrides) -> None:
    fields = {
        "name": "demo",
        "version": "0.1.0",
        "kind": "subprocess",
        "entrypoint": "bin/run",
        "provides_steps": ["demo.step"],
        "summary": "A demo plugin.",
        "min_transcoderr_version": "0.1.0",
    }
    fields.update(overrides)
    lines = []
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, list):
            inner = ", ".join(f'"{x}"' for x in v)
            lines.append(f"{k} = [{inner}]")
        else:
            lines.append(f'{k} = "{v}"')
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "manifest.toml").write_text("\n".join(lines) + "\n")


class LoadManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.plugin = self.tmp / "demo"

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_valid_manifest(self):
        _write_manifest(self.plugin)
        m = pub.load_manifest(self.plugin, "demo")
        self.assertEqual(m["name"], "demo")
        self.assertEqual(m["version"], "0.1.0")
        self.assertEqual(m["provides_steps"], ["demo.step"])

    def test_missing_required_field_raises_listing_all(self):
        _write_manifest(self.plugin, summary=None, min_transcoderr_version=None)
        with self.assertRaises(pub.PublishError) as ctx:
            pub.load_manifest(self.plugin, "demo")
        msg = str(ctx.exception)
        self.assertIn("summary", msg)
        self.assertIn("min_transcoderr_version", msg)

    def test_name_mismatch_raises(self):
        _write_manifest(self.plugin, name="other")
        with self.assertRaises(pub.PublishError) as ctx:
            pub.load_manifest(self.plugin, "demo")
        self.assertIn("doesn't match", str(ctx.exception))

    def test_missing_manifest_file_raises(self):
        self.plugin.mkdir()
        with self.assertRaises(pub.PublishError) as ctx:
            pub.load_manifest(self.plugin, "demo")
        self.assertIn("manifest.toml", str(ctx.exception))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 4 new tests fail with `AttributeError: module 'publish' has no attribute 'load_manifest'`.

- [ ] **Step 3: Implement `load_manifest`**

Add to `scripts/publish.py` after `parse_owner_repo`:

```python
def load_manifest(plugin_dir: Path, expected_name: str) -> dict:
    """Parse manifest.toml from plugin_dir and validate its required fields."""
    manifest_path = plugin_dir / "manifest.toml"
    if not manifest_path.exists():
        raise PublishError(f"manifest.toml not found at {manifest_path}")
    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)
    missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise PublishError(
            f"manifest.toml is missing required fields: {', '.join(missing)}"
        )
    if manifest["name"] != expected_name:
        raise PublishError(
            f"manifest.name {manifest['name']!r} doesn't match plugin dir "
            f"{expected_name!r}"
        )
    return manifest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (10 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: add load_manifest with required-field validation"
```

---

### Task 4: `find_entry` and `check_version_conflict` (TDD)

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_publish.py`:

```python
class IndexLookupTests(unittest.TestCase):
    def test_find_entry_returns_match(self):
        index = {"plugins": [{"name": "a", "version": "1"}, {"name": "b", "version": "2"}]}
        self.assertEqual(pub.find_entry(index, "b"), {"name": "b", "version": "2"})

    def test_find_entry_returns_none_when_missing(self):
        index = {"plugins": [{"name": "a", "version": "1"}]}
        self.assertIsNone(pub.find_entry(index, "z"))

    def test_find_entry_handles_no_plugins_key(self):
        self.assertIsNone(pub.find_entry({}, "z"))


class VersionConflictTests(unittest.TestCase):
    def test_no_existing_entry_passes(self):
        pub.check_version_conflict(None, "0.1.0", "demo")  # no raise

    def test_different_version_passes(self):
        pub.check_version_conflict({"version": "0.1.0"}, "0.1.1", "demo")  # no raise

    def test_same_version_raises(self):
        with self.assertRaises(pub.PublishError) as ctx:
            pub.check_version_conflict({"version": "0.1.0"}, "0.1.0", "demo")
        self.assertIn("already published", str(ctx.exception))
        self.assertIn("0.1.0", str(ctx.exception))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 6 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement both helpers**

Add to `scripts/publish.py` after `load_manifest`:

```python
def find_entry(index: dict, name: str) -> dict | None:
    """Return the plugin entry with this name from index, or None."""
    for entry in index.get("plugins", []):
        if entry.get("name") == name:
            return entry
    return None


def check_version_conflict(
    existing: dict | None, new_version: str, name: str
) -> None:
    """Fail if existing entry already publishes this exact version."""
    if existing is not None and existing.get("version") == new_version:
        raise PublishError(
            f"{name} {new_version} is already published. "
            f"Bump version in manifest.toml first."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (16 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: add find_entry and version-conflict gate"
```

---

### Task 5: Deterministic tarball builder (TDD)

The key property under test is **bit-identical output for identical input** — that's what makes the sha256 a stable identity rather than a function of build time.

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_publish.py`:

```python
import tarfile as _tarfile_mod


class BuildTarballTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.plugin = self.tmp / "demo"
        _write_manifest(self.plugin)
        bin_dir = self.plugin / "bin"
        bin_dir.mkdir()
        (bin_dir / "run").write_text("#!/bin/sh\necho hi\n")
        (bin_dir / "run").chmod(0o755)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_builds_are_byte_identical(self):
        a = pub.build_tarball_bytes(self.plugin)
        b = pub.build_tarball_bytes(self.plugin)
        self.assertEqual(a, b)

    def test_tarball_contains_plugin_files(self):
        data = pub.build_tarball_bytes(self.plugin)
        with _tarfile_mod.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            names = sorted(m.name for m in tf.getmembers())
        self.assertIn("demo/manifest.toml", names)
        self.assertIn("demo/bin/run", names)

    def test_tarball_entries_have_zero_mtime_and_owner(self):
        data = pub.build_tarball_bytes(self.plugin)
        with _tarfile_mod.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                self.assertEqual(m.mtime, 0, m.name)
                self.assertEqual(m.uid, 0, m.name)
                self.assertEqual(m.gid, 0, m.name)
                self.assertEqual(m.uname, "", m.name)
                self.assertEqual(m.gname, "", m.name)

    def test_executable_bit_preserved_for_bin_run(self):
        data = pub.build_tarball_bytes(self.plugin)
        with _tarfile_mod.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            run = tf.getmember("demo/bin/run")
            self.assertTrue(run.mode & 0o100, oct(run.mode))
```

You'll also need `import io` near the top of the test file if it's not already imported — add it at the top with the other imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 4 new tests fail with `AttributeError: module 'publish' has no attribute 'build_tarball_bytes'`.

- [ ] **Step 3: Implement `build_tarball_bytes`**

Add to `scripts/publish.py` after `check_version_conflict`:

```python
def build_tarball_bytes(plugin_dir: Path) -> bytes:
    """Build a deterministic .tar.gz of plugin_dir, returned as raw bytes.

    Determinism requires: zero mtime everywhere (per-entry and the gzip
    header), zero uid/gid with empty uname/gname, normalized modes,
    sorted iteration order, and USTAR format (no PAX extended headers).
    """

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        if info.isdir():
            info.mode = 0o755
        elif info.isfile():
            info.mode = 0o755 if (info.mode & 0o100) else 0o644
        return info

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            for path in sorted(plugin_dir.rglob("*")):
                arcname = path.relative_to(plugin_dir.parent).as_posix()
                tf.add(path, arcname=arcname, recursive=False, filter=_filter)
    return buf.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (20 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: deterministic tarball builder"
```

---

### Task 6: `build_entry` (index entry construction) (TDD)

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_publish.py`:

```python
class BuildEntryTests(unittest.TestCase):
    def test_field_mapping(self):
        manifest = {
            "name": "demo",
            "version": "0.2.0",
            "kind": "subprocess",
            "entrypoint": "bin/run",
            "provides_steps": ["demo.before", "demo.after"],
            "summary": "demo summary",
            "min_transcoderr_version": "0.19.0",
        }
        entry = pub.build_entry(manifest, "foo", "bar", "deadbeef" * 8)
        self.assertEqual(entry, {
            "name": "demo",
            "version": "0.2.0",
            "summary": "demo summary",
            "tarball_url": "https://raw.githubusercontent.com/foo/bar/main/tarballs/demo-0.2.0.tar.gz",
            "tarball_sha256": "deadbeef" * 8,
            "homepage": "https://github.com/foo/bar/tree/main/demo",
            "min_transcoderr_version": "0.19.0",
            "kind": "subprocess",
            "provides_steps": ["demo.before", "demo.after"],
        })

    def test_provides_steps_is_a_new_list(self):
        steps = ["a"]
        manifest = {
            "name": "x", "version": "1", "kind": "subprocess",
            "entrypoint": "e", "provides_steps": steps,
            "summary": "s", "min_transcoderr_version": "0",
        }
        entry = pub.build_entry(manifest, "o", "r", "sha")
        entry["provides_steps"].append("mutated")
        self.assertEqual(steps, ["a"])  # original is untouched
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 2 new tests fail.

- [ ] **Step 3: Implement `build_entry`**

Add to `scripts/publish.py` after `build_tarball_bytes`:

```python
def build_entry(manifest: dict, owner: str, repo: str, sha256: str) -> dict:
    """Build the index.json entry dict from a manifest + repo identity + sha."""
    name = manifest["name"]
    version = manifest["version"]
    return {
        "name": name,
        "version": version,
        "summary": manifest["summary"],
        "tarball_url": (
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/"
            f"tarballs/{name}-{version}.tar.gz"
        ),
        "tarball_sha256": sha256,
        "homepage": f"https://github.com/{owner}/{repo}/tree/main/{name}",
        "min_transcoderr_version": manifest["min_transcoderr_version"],
        "kind": manifest["kind"],
        "provides_steps": list(manifest["provides_steps"]),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (22 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: add build_entry for index.json field mapping"
```

---

### Task 7: `write_index` (TDD)

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_publish.py`:

```python
class WriteIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "index.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_appends_new_entry(self):
        index = {"schema_version": 1, "catalog_name": "t",
                 "catalog_url": "u", "plugins": []}
        new = {"name": "demo", "version": "0.1.0"}
        pub.write_index(index, new, self.path)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["plugins"], [new])

    def test_replaces_existing_entry_by_name(self):
        old = {"name": "demo", "version": "0.1.0"}
        new = {"name": "demo", "version": "0.2.0"}
        index = {"schema_version": 1, "catalog_name": "t",
                 "catalog_url": "u", "plugins": [old]}
        pub.write_index(index, new, self.path)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["plugins"], [new])

    def test_plugins_sorted_by_name(self):
        index = {"schema_version": 1, "catalog_name": "t",
                 "catalog_url": "u",
                 "plugins": [{"name": "z"}, {"name": "a"}]}
        pub.write_index(index, {"name": "m"}, self.path)
        loaded = json.loads(self.path.read_text())
        self.assertEqual([p["name"] for p in loaded["plugins"]], ["a", "m", "z"])

    def test_top_level_keys_preserved(self):
        index = {"schema_version": 1, "catalog_name": "official",
                 "catalog_url": "https://example", "plugins": []}
        pub.write_index(index, {"name": "x"}, self.path)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["catalog_name"], "official")
        self.assertEqual(loaded["catalog_url"], "https://example")

    def test_indented_with_trailing_newline(self):
        index = {"schema_version": 1, "catalog_name": "t",
                 "catalog_url": "u", "plugins": []}
        pub.write_index(index, {"name": "x"}, self.path)
        text = self.path.read_text()
        self.assertTrue(text.endswith("\n"))
        self.assertIn('\n  "schema_version"', text)  # 2-space indent
```

Add `import json` to the test file's imports if not already present.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 5 new tests fail.

- [ ] **Step 3: Implement `write_index`**

Add to `scripts/publish.py` after `build_entry`:

```python
def write_index(index: dict, new_entry: dict, index_path: Path) -> None:
    """Replace or append new_entry in index, sort plugins by name, write to disk."""
    plugins = [p for p in index.get("plugins", []) if p.get("name") != new_entry["name"]]
    plugins.append(new_entry)
    plugins.sort(key=lambda p: p["name"])
    index["plugins"] = plugins
    index_path.write_text(json.dumps(index, indent=2) + "\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (27 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: add write_index with stable ordering"
```

---

### Task 8: `publish()` orchestrator + integration test (TDD)

This wires the helpers together against a real (tmpdir) git repo. The integration test is the proof that the pieces compose correctly.

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing integration tests**

Append to `tests/test_publish.py`:

```python
import subprocess as _subprocess


def _make_repo(tmp: Path, plugin_name: str = "demo", manifest_overrides: dict | None = None) -> Path:
    """Init a git repo with origin set, an index.json, and a plugin dir.

    Returns the repo root.
    """
    _subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp, check=True)
    _subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/example/cat.git"],
        cwd=tmp, check=True,
    )
    (tmp / "index.json").write_text(json.dumps({
        "schema_version": 1,
        "catalog_name": "test",
        "catalog_url": "https://github.com/example/cat",
        "plugins": [],
    }, indent=2) + "\n")
    plugin = tmp / plugin_name
    _write_manifest(plugin, name=plugin_name, **(manifest_overrides or {}))
    bin_dir = plugin / "bin"
    bin_dir.mkdir()
    (bin_dir / "run").write_text("#!/bin/sh\necho hi\n")
    (bin_dir / "run").chmod(0o755)
    return tmp


class PublishIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_publish_creates_tarball_and_appends_entry(self):
        summary = pub.publish("demo", self.repo)

        tarball = self.repo / "tarballs" / "demo-0.1.0.tar.gz"
        self.assertTrue(tarball.exists())

        index = json.loads((self.repo / "index.json").read_text())
        self.assertEqual(len(index["plugins"]), 1)
        entry = index["plugins"][0]
        self.assertEqual(entry["name"], "demo")
        self.assertEqual(entry["version"], "0.1.0")
        self.assertEqual(entry["homepage"], "https://github.com/example/cat/tree/main/demo")
        self.assertEqual(
            entry["tarball_url"],
            "https://raw.githubusercontent.com/example/cat/main/tarballs/demo-0.1.0.tar.gz",
        )

        # sha in index matches sha of the file on disk
        actual_sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        self.assertEqual(entry["tarball_sha256"], actual_sha)

        self.assertIn("demo: (new) -> 0.1.0", summary)

    def test_update_replaces_entry_and_deletes_old_tarball(self):
        # First publish at 0.1.0
        pub.publish("demo", self.repo)
        old_tarball = self.repo / "tarballs" / "demo-0.1.0.tar.gz"
        self.assertTrue(old_tarball.exists())

        # Bump manifest to 0.2.0 and publish again
        _write_manifest(self.repo / "demo", name="demo", version="0.2.0")
        summary = pub.publish("demo", self.repo)

        new_tarball = self.repo / "tarballs" / "demo-0.2.0.tar.gz"
        self.assertTrue(new_tarball.exists())
        self.assertFalse(old_tarball.exists())

        index = json.loads((self.repo / "index.json").read_text())
        self.assertEqual(len(index["plugins"]), 1)
        self.assertEqual(index["plugins"][0]["version"], "0.2.0")
        self.assertIn("demo: 0.1.0 -> 0.2.0", summary)

    def test_version_conflict_raises_after_first_publish(self):
        pub.publish("demo", self.repo)
        with self.assertRaises(pub.PublishError) as ctx:
            pub.publish("demo", self.repo)
        self.assertIn("already published", str(ctx.exception))

    def test_missing_plugin_dir_raises(self):
        with self.assertRaises(pub.PublishError) as ctx:
            pub.publish("nonexistent", self.repo)
        self.assertIn("not found", str(ctx.exception))
```

Make sure `import hashlib` is in the test file's imports (add it at the top).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 4 new tests fail with `AttributeError: module 'publish' has no attribute 'publish'`.

- [ ] **Step 3: Implement helper functions and `publish()`**

Add to `scripts/publish.py` after `write_index`:

```python
def get_origin_url(repo_root: Path) -> str:
    """Return the URL of the 'origin' remote of repo_root."""
    return subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def get_repo_root() -> Path:
    """Return the repo root via `git rev-parse --show-toplevel` for $PWD."""
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def publish(plugin_name: str, repo_root: Path) -> str:
    """Publish plugin_name from repo_root. Returns a one-line summary."""
    plugin_dir = repo_root / plugin_name
    if not plugin_dir.is_dir():
        raise PublishError(f"plugin directory '{plugin_name}/' not found")

    manifest = load_manifest(plugin_dir, plugin_name)
    new_version = manifest["version"]

    index_path = repo_root / "index.json"
    index = json.loads(index_path.read_text())
    existing = find_entry(index, plugin_name)
    check_version_conflict(existing, new_version, plugin_name)

    owner, repo = parse_owner_repo(get_origin_url(repo_root))

    tarball_bytes = build_tarball_bytes(plugin_dir)
    sha256 = hashlib.sha256(tarball_bytes).hexdigest()

    tarballs_dir = repo_root / "tarballs"
    tarballs_dir.mkdir(exist_ok=True)
    new_tarball = tarballs_dir / f"{plugin_name}-{new_version}.tar.gz"
    new_tarball.write_bytes(tarball_bytes)

    if existing is not None:
        old_version = existing["version"]
        old_tarball = tarballs_dir / f"{plugin_name}-{old_version}.tar.gz"
        if old_tarball.exists():
            old_tarball.unlink()
    else:
        old_version = "(new)"

    entry = build_entry(manifest, owner, repo, sha256)
    write_index(index, entry, index_path)

    return f"{plugin_name}: {old_version} -> {new_version} ({sha256})"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (31 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: add publish() orchestrator with integration tests"
```

---

### Task 9: Wire `main()` and add CLI tests (TDD)

**Files:**
- Modify: `scripts/publish.py`
- Modify: `tests/test_publish.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_publish.py`:

```python
import contextlib
import os


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_main_returns_zero_on_success(self):
        with _chdir(self.repo):
            self.assertEqual(pub.main(["demo"]), 0)

    def test_main_returns_one_and_prints_error_on_failure(self):
        with _chdir(self.repo):
            pub.main(["demo"])  # first publish
            # Second run with same version should fail.
            self.assertEqual(pub.main(["demo"]), 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_publish -v`
Expected: both new tests fail (current `main` is a stub that does nothing).

- [ ] **Step 3: Replace `main` in `scripts/publish.py`**

Replace the existing `main` definition with:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", help="plugin directory name")
    args = parser.parse_args(argv)

    try:
        summary = publish(args.plugin, get_repo_root())
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(summary)
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_publish -v`
Expected: all tests pass (33 total).

- [ ] **Step 5: Make the script executable and commit**

```bash
chmod +x scripts/publish.py
git add scripts/publish.py tests/test_publish.py
git commit -m "publish: wire main() with stderr error reporting"
```

---

### Task 10: Local end-to-end sanity check against `size-report` (no commit)

This validates the script works against the live repo state before exposing it via the workflow. Discard all changes at the end — the workflow PR is what publishes for real.

**Files:** none modified durably.

- [ ] **Step 1: Confirm baseline is clean**

Run: `git status`
Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: Verify the version-conflict gate fires for the current version**

Run: `python3 scripts/publish.py size-report`
Expected:
```
error: size-report 0.1.0 is already published. Bump version in manifest.toml first.
```
And exit code 1 (`echo $?` → `1`).

- [ ] **Step 3: Bump the manifest to 0.1.1 and publish locally**

Edit `size-report/manifest.toml` and change `version = "0.1.0"` to `version = "0.1.1"`.

Run: `python3 scripts/publish.py size-report`
Expected: prints `size-report: 0.1.0 -> 0.1.1 (<sha>)` and exits 0.

- [ ] **Step 4: Inspect the result**

Run: `ls tarballs/`
Expected: `size-report-0.1.1.tar.gz` (the 0.1.0 tarball is gone).

Run: `python3 -c "import json; e = json.load(open('index.json'))['plugins'][0]; print(e['version'], e['tarball_sha256'])"`
Expected: `0.1.1` followed by a sha matching `shasum -a 256 tarballs/size-report-0.1.1.tar.gz`.

- [ ] **Step 5: Verify reproducibility**

Run:
```bash
cp tarballs/size-report-0.1.1.tar.gz /tmp/first.tar.gz
# Force a rebuild: bump to 0.1.2 then back to 0.1.1
sed -i.bak 's/version = "0.1.1"/version = "0.1.2"/' size-report/manifest.toml
python3 scripts/publish.py size-report >/dev/null
sed -i.bak 's/version = "0.1.2"/version = "0.1.1"/' size-report/manifest.toml
rm -f tarballs/size-report-0.1.2.tar.gz size-report/manifest.toml.bak
python3 scripts/publish.py size-report >/dev/null
diff /tmp/first.tar.gz tarballs/size-report-0.1.1.tar.gz && echo OK
```

Wait — the second run will hit the version-conflict gate because index.json now lists 0.1.2. Adjust: instead, run `build_tarball_bytes` directly twice via Python to verify determinism without going through the publish pipeline:

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import publish, hashlib, pathlib
p = pathlib.Path('size-report')
a = publish.build_tarball_bytes(p)
b = publish.build_tarball_bytes(p)
print('identical:', a == b)
print('sha:', hashlib.sha256(a).hexdigest())
"
```

Expected: `identical: True` and a sha that matches the one in index.json.

- [ ] **Step 6: Discard all local changes**

Run:
```bash
git checkout -- size-report/manifest.toml index.json tarballs/
git status
```
Expected: working tree clean. The 0.1.0 tarball is restored, index.json is restored, manifest.toml is restored.

If `tarballs/size-report-0.1.1.tar.gz` lingers as untracked, remove it:
```bash
rm -f tarballs/size-report-0.1.1.tar.gz
```

---

### Task 11: Add the GitHub workflow

**Files:**
- Create: `.github/workflows/publish-plugin.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/publish-plugin.yml`:

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
        env:
          PLUGIN: ${{ inputs.plugin }}
        run: python3 scripts/publish.py "$PLUGIN"

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

- [ ] **Step 2: Validate the YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish-plugin.yml'))"`

If `pyyaml` isn't installed (`ModuleNotFoundError: No module named 'yaml'`), use this fallback that at least catches gross errors:

```bash
python3 -c "
import re
text = open('.github/workflows/publish-plugin.yml').read()
assert 'workflow_dispatch' in text
assert 'permissions' in text
assert 'peter-evans/create-pull-request' in text
print('OK')
"
```

Expected: `OK` (or no error from the yaml import).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish-plugin.yml
git commit -m "ci: add manual publish-plugin workflow"
```

---

### Task 12: Update README to point at the workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README section**

Run: `cat README.md`

The "Adding a plugin" section currently lists 5 manual steps (drop dir, build tarball, compute sha, edit index, open PR).

- [ ] **Step 2: Replace the "Adding a plugin" section**

Open `README.md` and replace the entire `## Adding a plugin` block (from that heading up to `## Hosting your own catalog`) with:

```markdown
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

```

- [ ] **Step 3: Verify the README still renders cleanly**

Run: `head -60 README.md`
Expected: the "What's here", "Plugins", and new "Adding a plugin" sections in order, with no leftover `tar -czf` / `shasum -a 256` instructions.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: point README at the publish-plugin workflow"
```

---

### Task 13: Final test sweep before shipping

**Files:** none modified.

- [ ] **Step 1: Run the full test suite one more time**

Run: `python3 -m unittest tests.test_publish -v`
Expected: 33 tests, all pass.

- [ ] **Step 2: Confirm git status is clean and all commits are present**

Run: `git status && git log --oneline -15`
Expected: `nothing to commit, working tree clean`, and the recent log shows (in order, most recent first):

```
docs: point README at the publish-plugin workflow
ci: add manual publish-plugin workflow
publish: wire main() with stderr error reporting
publish: add publish() orchestrator with integration tests
publish: add write_index with stable ordering
publish: add build_entry for index.json field mapping
publish: deterministic tarball builder
publish: add find_entry and version-conflict gate
publish: add load_manifest with required-field validation
publish: scaffold script and parse_owner_repo helper
size-report: add summary and min_transcoderr_version to manifest
docs: spec for publish-plugin workflow
initial catalog with size-report 0.1.0
```

- [ ] **Step 3: Push and dispatch the workflow once for real**

After the branch is merged to `main`, smoke-test by triggering the workflow against `size-report` with its version bumped (e.g. to `0.1.1`):

```
gh workflow run publish-plugin.yml -f plugin=size-report
```

Expected: a new PR titled `publish: size-report` appears, containing the new tarball and an updated `index.json` entry. Merge that PR to actually republish.
