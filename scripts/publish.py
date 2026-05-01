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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", help="plugin directory name")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
