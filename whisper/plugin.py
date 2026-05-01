#!/usr/bin/env python3
"""whisper.transcribe — sidecar .srt via faster-whisper.

Pure helpers and protocol orchestration for the whisper plugin. Tests
import this module directly; production runs via bin/run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG = {
    "model": "large-v3-turbo",
    "language": "auto",
    "skip_if_exists": True,
    "compute_type": "auto",
}


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
