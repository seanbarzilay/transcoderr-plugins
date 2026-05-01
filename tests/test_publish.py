"""Unit + integration tests for scripts/publish.py."""
from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main()
