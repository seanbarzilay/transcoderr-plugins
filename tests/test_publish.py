"""Unit + integration tests for scripts/publish.py."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
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

    def test_invalid_toml_raises_publish_error(self):
        self.plugin.mkdir()
        (self.plugin / "manifest.toml").write_text("not [valid toml = ")
        with self.assertRaises(pub.PublishError) as ctx:
            pub.load_manifest(self.plugin, "demo")
        self.assertIn("invalid TOML", str(ctx.exception))


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

    def test_tarball_includes_plugin_root_dir_entry(self):
        data = pub.build_tarball_bytes(self.plugin)
        with _tarfile_mod.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            root = tf.getmember("demo")
        self.assertTrue(root.isdir())
        self.assertEqual(root.mode, 0o755)
        self.assertEqual(root.mtime, 0)
        self.assertEqual(root.uid, 0)
        self.assertEqual(root.gid, 0)

    def test_symlink_in_plugin_raises(self):
        link = self.plugin / "bin" / "alias"
        link.symlink_to("run")
        with self.assertRaises(pub.PublishError) as ctx:
            pub.build_tarball_bytes(self.plugin)
        self.assertIn("symlink", str(ctx.exception).lower())

    def test_non_executable_file_has_644_mode(self):
        data = pub.build_tarball_bytes(self.plugin)
        with _tarfile_mod.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            manifest_member = tf.getmember("demo/manifest.toml")
        self.assertEqual(manifest_member.mode, 0o644)


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

    def test_corrupt_index_raises_publish_error(self):
        (self.repo / "index.json").write_text("not json {")
        with self.assertRaises(pub.PublishError) as ctx:
            pub.publish("demo", self.repo)
        self.assertIn("index.json", str(ctx.exception))


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
        with _chdir(self.repo), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pub.main(["demo"]), 0)

    def test_main_returns_one_and_prints_error_on_failure(self):
        with _chdir(self.repo), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            pub.main(["demo"])  # first publish
            # Second run with same version should fail.
            self.assertEqual(pub.main(["demo"]), 1)


if __name__ == "__main__":
    unittest.main()
