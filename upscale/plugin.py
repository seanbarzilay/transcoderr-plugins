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


_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def compute_target_height(
    source_w: int, source_h: int, target_h: int
) -> tuple[int, int]:
    """Compute the final (width, height) after a lanczos-downscale.

    `target_h == 0` disables the resize: returns the input unchanged.
    Otherwise computes the new width preserving aspect ratio, rounded
    to even (codec requirement).
    """
    if target_h <= 0:
        return source_w, source_h
    new_w = round(source_w * (target_h / source_h))
    if new_w % 2 != 0:
        new_w -= 1
    return new_w, target_h


def parse_progress_line(line: str) -> tuple[int, int] | None:
    """Pull (done, total) frame counts from a ncnn-vulkan stderr line.

    Returns None for lines that don't match or have a zero total.
    """
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    if total == 0:
        return None
    return done, total


def parse_execute(line: str) -> dict:
    """Parse a JSON-RPC execute line into step_id, file_path, and config.

    Config defaults are filled in for any missing keys.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"execute message is not valid JSON: {exc}") from exc

    step_id = msg.get("step_id")
    if not step_id:
        raise ProtocolError("execute message missing step_id")

    ctx = msg.get("ctx") or {}
    file_path = (ctx.get("file") or {}).get("path")
    if not file_path:
        raise ProtocolError("execute message missing ctx.file.path")

    user_config = msg.get("config") or {}
    config = {**DEFAULT_CONFIG, **user_config}

    return {
        "step_id": step_id,
        "file_path": file_path,
        "config": config,
    }


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
