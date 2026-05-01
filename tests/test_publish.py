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
