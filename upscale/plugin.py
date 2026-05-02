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


def emit_log(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "log", "msg": msg}, separators=(",", ":")) + "\n")


def emit_progress(done: int, total: int, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "progress", "done": done, "total": total},
        separators=(",", ":"),
    ) + "\n")


def emit_context_set(key: str, value: dict, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "context_set", "key": key, "value": value},
        separators=(",", ":"),
    ) + "\n")


def emit_result_ok(out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "result", "status": "ok", "outputs": {}},
        separators=(",", ":"),
    ) + "\n")


def emit_result_err(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps(
        {"event": "result", "status": "error", "error": {"msg": msg}},
        separators=(",", ":"),
    ) + "\n")


def probe_dimensions(file_path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream in file_path."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(file_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProtocolError("ffprobe not on PATH") from exc

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise ProtocolError("no video stream found")
    s = streams[0]
    return int(s["width"]), int(s["height"])


def run_upscale_subprocess(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    scale: int,
    tile_size: int,
    stdout,
) -> None:
    """Invoke realesrgan-ncnn-vulkan, stream progress events to stdout.

    Raises ProtocolError on non-zero exit or missing binary.
    """
    argv = [
        "realesrgan-ncnn-vulkan",
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", model,
        "-s", str(scale),
        "-f", "mkv",
    ]
    if tile_size and tile_size > 0:
        argv.extend(["-t", str(tile_size)])

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProtocolError("realesrgan-ncnn-vulkan not on PATH") from exc

    stderr_lines: list[str] = []
    if proc.stderr is not None:
        for line in proc.stderr:
            line = line.rstrip("\n")
            stderr_lines.append(line)
            parsed = parse_progress_line(line)
            if parsed is not None:
                done, total = parsed
                emit_progress(done, total, out=stdout)

    rc = proc.wait()
    if rc != 0:
        stderr_blob = "\n".join(stderr_lines).strip()
        if "vkAllocateMemory" in stderr_blob or "out of device memory" in stderr_blob.lower():
            raise ProtocolError(
                "OOM during upscale; try a smaller tile_size"
            )
        raise ProtocolError(f"realesrgan failed: {stderr_blob}")


def run_downscale_subprocess(
    input_path: Path, output_path: Path, *, target_height: int
) -> None:
    """Lanczos-downscale input_path to (-2:target_height) and re-encode.

    Intermediate codec is libx264 ultrafast crf=18 — file is transient
    (consumed by the next mux step via -c copy). Raises ProtocolError on
    non-zero exit.
    """
    argv = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale=-2:{target_height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        str(output_path),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ProtocolError("ffmpeg not on PATH") from exc

    if result.returncode != 0:
        raise ProtocolError(
            f"downscale ffmpeg failed: {(result.stderr or '').strip()}"
        )


def run_mux_subprocess(
    video_only_path: Path,
    original_input: Path,
    output_path: Path,
) -> None:
    """Mux upscaled video stream from video_only_path with audio + subs from original_input.

    Streams are stream-copied (no re-encoding). Audio and subtitle maps
    use `?` suffix so missing streams don't cause ffmpeg to abort.
    """
    argv = [
        "ffmpeg", "-y",
        "-i", str(video_only_path),
        "-i", str(original_input),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-c", "copy",
        str(output_path),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ProtocolError("ffmpeg not on PATH") from exc

    if result.returncode != 0:
        raise ProtocolError(
            f"mux ffmpeg failed: {(result.stderr or '').strip()}"
        )


def main(stdin=None, stdout=None) -> int:
    """Entry point. Reads init+execute from stdin, emits events to stdout."""
    raise NotImplementedError("filled in by Task 9")


if __name__ == "__main__":
    raise SystemExit(main())
