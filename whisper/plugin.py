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


class ProtocolError(Exception):
    """Raised when the JSON-RPC execute message is missing required fields."""


def fmt_ts(secs: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    ms_total = int(round(secs * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(segments: Iterable) -> str:
    """Format an iterable of (start, end, text) segments as SRT text.

    `segments` may be any iterable of objects with `.start`, `.end`, and
    `.text` attributes — typically faster-whisper Segments. Text is
    stripped. Cues are 1-indexed and separated by a blank line. Empty
    iterable returns the empty string (no trailing newline).
    """
    parts = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        parts.append(
            f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n"
        )
    return "".join(parts)


def write_srt_atomically(path: Path, srt_text: str) -> None:
    """Write srt_text to path atomically (write to .tmp + os.replace).

    The caller is responsible for ensuring the parent directory exists.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(srt_text)
    os.replace(tmp, path)


def find_existing_sidecar(video_path: Path, language: str | None) -> Path | None:
    """Return an existing sidecar .srt for video_path, or None.

    With `language` set to an ISO code, checks for `<basename>.<lang>.srt`
    exactly. With `language=None` (autodetect), returns the first
    `<basename>.*.srt` that exists. `<basename>` here means the filename
    with only its rightmost extension stripped — `Movie.2024.1080p.mkv`
    yields a base of `Movie.2024.1080p` so dot-separated quality tags
    are preserved.
    """
    if language is not None:
        candidate = video_path.with_name(f"{video_path.stem}.{language}.srt")
        return candidate if candidate.exists() else None
    pattern = str(video_path.with_suffix("")) + ".*.srt"
    matches = sorted(glob(pattern))
    return Path(matches[0]) if matches else None


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


def cuda_available() -> bool:
    """Return True if a CUDA device looks usable.

    Indirected through this function so tests can monkeypatch it.
    Production: probe ctranslate2's device list lazily — a missing import
    or a no-CUDA-build returns False rather than raising.
    """
    try:
        import ctranslate2  # type: ignore
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_compute_type(user_value: str) -> str:
    """Resolve 'auto' to float16 (GPU) or int8 (CPU); pass others through."""
    if user_value != "auto":
        return user_value
    return "float16" if cuda_available() else "int8"


def emit_log(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "log", "msg": msg}, separators=(",", ":")) + "\n")


def emit_context_set(key: str, value: dict, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "context_set", "key": key, "value": value}, separators=(",", ":")) + "\n")


def emit_result_ok(out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "result", "status": "ok", "outputs": {}}, separators=(",", ":")) + "\n")


def emit_result_err(msg: str, out=None) -> None:
    out = out if out is not None else sys.stdout
    out.write(json.dumps({"event": "result", "status": "error", "error": {"msg": msg}}, separators=(",", ":")) + "\n")


def has_audio_stream(file_path: Path) -> bool:
    """Return True if ffprobe sees at least one audio stream in file_path."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
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
    except json.JSONDecodeError:
        return False
    return bool(data.get("streams"))


def load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper model. Indirected so tests can monkeypatch."""
    from faster_whisper import WhisperModel  # noqa: WPS433 (lazy import)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(file_path: Path, config: dict, *, stdout) -> dict | None:
    """Transcribe file_path. Returns ctx dict on success, or None on benign skip.

    Benign skips (no-audio, sidecar-already-exists, no-speech) emit a
    log event but no context_set. Errors raise; the caller maps them to
    result events.
    """
    if not file_path.exists():
        raise ProtocolError("file does not exist")

    if not has_audio_stream(file_path):
        emit_log("no audio stream, skipping", out=stdout)
        return None

    skip_lang = None if config["language"] == "auto" else config["language"]
    if config["skip_if_exists"]:
        existing = find_existing_sidecar(file_path, skip_lang)
        if existing is not None:
            emit_log(f"sidecar already exists at {existing}, skipping", out=stdout)
            return None

    compute_type = resolve_compute_type(config["compute_type"])
    model = load_model(config["model"], device="auto", compute_type=compute_type)

    started = time.monotonic()
    segments_iter, info = model.transcribe(
        str(file_path),
        language=skip_lang,
        vad_filter=True,
    )
    segments = list(segments_iter)

    if not segments:
        emit_log("no speech detected, skipping", out=stdout)
        return None

    srt_text = format_srt(segments)
    sidecar = file_path.with_suffix(f".{info.language}.srt")
    write_srt_atomically(sidecar, srt_text)

    elapsed = time.monotonic() - started
    return {
        "subtitle_path": str(sidecar),
        "language": info.language,
        "model": config["model"],
        "duration_sec": round(elapsed, 3),
    }


def main(stdin=None, stdout=None) -> int:
    """Read init+execute from stdin, drive transcribe, emit events to stdout."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    try:
        _init_line = stdin.readline()
        if not _init_line:
            emit_result_err("no init message on stdin", out=stdout)
            return 0
        exec_line = stdin.readline()
        if not exec_line:
            emit_result_err("no execute message on stdin", out=stdout)
            return 0

        try:
            parsed = parse_execute(exec_line)
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0

        if parsed["step_id"] != "whisper.transcribe":
            emit_result_err(f"unknown step_id: {parsed['step_id']}", out=stdout)
            return 0

        try:
            ctx_value = transcribe(
                Path(parsed["file_path"]),
                parsed["config"],
                stdout=stdout,
            )
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0

        if ctx_value is not None:
            emit_context_set("whisper", ctx_value, out=stdout)
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
