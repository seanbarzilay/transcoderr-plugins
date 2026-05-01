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
