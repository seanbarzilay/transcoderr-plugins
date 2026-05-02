#!/usr/bin/env python3
"""upscale.video — AI-upscale via Real-ESRGAN (ncnn-vulkan).

Pure-stdlib orchestrator that drives ffprobe + realesrgan-ncnn-vulkan +
ffmpeg subprocesses. Tests import this module directly; production runs
via bin/run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

DEFAULT_CONFIG = {
    "model": "realesr-animevideov3",
    "scale": 4,
    "target_height": 1080,
    "min_source_height": 720,
    "output_path": None,
    "denoise_strength": 0.5,
    "tile_size": 0,
}


class ProtocolError(Exception):
    """Raised when the JSON-RPC execute message or a subprocess invocation fails."""


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
