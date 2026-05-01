"""Unit + integration tests for scripts/publish.py."""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
