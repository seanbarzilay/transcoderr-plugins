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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", help="plugin directory name")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
