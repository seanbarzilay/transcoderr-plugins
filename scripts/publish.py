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
        try:
            manifest = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise PublishError(
                f"manifest.toml is invalid TOML: {exc}"
            ) from exc
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
            # Normalize to standard Unix modes; intentionally grants group+other read.
            info.mode = 0o755 if (info.mode & 0o100) else 0o644
        return info

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            entries = sorted(plugin_dir.rglob("*"))
            for path in entries:
                if path.is_symlink():
                    raise PublishError(
                        f"symlink not allowed in plugin tree: "
                        f"{path.relative_to(plugin_dir.parent).as_posix()}"
                    )
            for path in [plugin_dir, *entries]:
                arcname = path.relative_to(plugin_dir.parent).as_posix()
                tf.add(path, arcname=arcname, recursive=False, filter=_filter)
    return buf.getvalue()


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
        "runtimes": list(manifest.get("runtimes", [])),
    }


def write_index(index: dict, new_entry: dict, index_path: Path) -> None:
    """Replace or append new_entry in index, sort plugins by name, write to disk."""
    plugins = [p for p in index.get("plugins", []) if p.get("name") != new_entry["name"]]
    plugins.append(new_entry)
    plugins.sort(key=lambda p: p["name"])
    index["plugins"] = plugins
    index_path.write_text(json.dumps(index, indent=2) + "\n")


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
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"could not read index.json: {exc}") from exc
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


if __name__ == "__main__":
    raise SystemExit(main())
