#!/usr/bin/env python3
"""subsync.align — re-time an out-of-sync .srt against the audio of a video.

Pure helpers and protocol orchestration for the subsync plugin. Tests
import this module directly; production runs via bin/run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from glob import glob
from pathlib import Path

# Heartbeat cadence for the dispatcher's inter-frame timer. The
# transcoderr coordinator drops a remote step that goes 30s without
# emitting any frame; ffsubsync runs ffmpeg internally for audio
# extraction and stays silent for several seconds at a time.
HEARTBEAT_INTERVAL_SECS = 10.0

DEFAULT_CONFIG = {
    "subtitle_path": "",
    "max_offset_seconds": 60.0,
    "framerate_correction": True,
    "fail_on_no_match": False,
}


class ProtocolError(Exception):
    """Raised when the JSON-RPC execute message is missing required fields."""


# ---- Event emitters (verbatim copy from whisper/plugin.py) -------------

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


# ---- Protocol parsing (skeleton; expanded in Task 2) ------------------

def parse_execute(line: str) -> dict:
    """Parse a JSON-RPC execute line. Returns {step_id, file_path, ctx, config}.

    Mirrors whisper/plugin.py:parse_execute. Tolerates both the production
    nested form (`params.{step_id, with, context}`) and the legacy flat
    form (`{step_id, ctx, config}`) so hand-crafted test fixtures still
    parse.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"execute message is not valid JSON: {exc}") from exc

    body = msg.get("params") if isinstance(msg.get("params"), dict) else msg

    step_id = body.get("step_id")
    if not step_id:
        raise ProtocolError("execute message missing step_id")

    context = body.get("context") or body.get("ctx") or {}
    file_path = (context.get("file") or {}).get("path")
    if not file_path:
        raise ProtocolError("execute message missing context.file.path")

    user_config = body.get("with") or body.get("config") or {}
    config = {**DEFAULT_CONFIG, **user_config}

    return {
        "step_id": step_id,
        "file_path": file_path,
        "ctx": context,
        "config": config,
    }


# ---- Subtitle path resolution ------------------------------------------

def find_subtitle_path(
    config: dict,
    ctx: dict,
    video_path: Path,
) -> Path | None:
    """Resolve which .srt to sync.

    Priority (per spec):
      1. config["subtitle_path"] — operator-supplied override (already
         template-resolved by the engine before we see it).
      2. First step output in ctx["steps"][*] that has a `subtitle_path`
         field. Auto-discovers the whisper plugin's output without the
         operator wiring an explicit reference.
      3. Glob `<basename>.*.srt` next to `video_path` and pick the most
         recently modified.
      4. None — caller treats this as a benign no-op (warn + result:ok).
    """
    override = (config.get("subtitle_path") or "").strip()
    if override:
        return Path(override)

    steps = ctx.get("steps") or {}
    if isinstance(steps, dict):
        for value in steps.values():
            if isinstance(value, dict):
                candidate = value.get("subtitle_path")
                if candidate:
                    return Path(candidate)

    pattern = str(video_path.with_suffix("")) + ".*.srt"
    matches = glob(pattern)
    if not matches:
        return None
    # Most recently modified among matches.
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


# ---- ffsubsync stderr parsing ------------------------------------------

# ffsubsync logs the computed offset to stderr in a line like
# `INFO:root:offset seconds: 1.234`. The framerate scale is logged as
# `INFO:root:framerate scale factor: 1.0`. Both formats are stable
# enough to parse with a simple substring match — full version-pinning
# happens via the requirements.txt but we don't depend on the exact
# logger name in case ffsubsync internals change.

def parse_offset_from_stderr(stderr: str) -> float | None:
    """Return the computed offset in seconds, or None if not found."""
    import re
    m = re.search(r"offset\s+seconds:\s*([-+]?[0-9]*\.?[0-9]+)", stderr)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_framerate_corrected_from_stderr(stderr: str) -> bool:
    """Return True if ffsubsync applied a non-1.0 framerate scale factor."""
    import re
    m = re.search(r"framerate\s+scale\s+factor:\s*([0-9]*\.?[0-9]+)", stderr)
    if not m:
        return False
    try:
        return abs(float(m.group(1)) - 1.0) > 1e-6
    except ValueError:
        return False


# ---- Atomic in-place replacement --------------------------------------

def atomic_replace(tmp_path: Path, target_path: Path) -> None:
    """Move tmp_path over target_path atomically. Raises FileNotFoundError
    if tmp_path doesn't exist; the caller is responsible for ensuring
    ffsubsync wrote it.
    """
    if not tmp_path.exists():
        raise FileNotFoundError(f"tmp output {tmp_path} not found")
    os.replace(tmp_path, target_path)


# ---- main (skeleton; expanded in Task 4) -------------------------------

def main(stdin=None, stdout=None) -> int:
    """Read init+execute from stdin, drive the sync, emit events to stdout."""
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

        if parsed["step_id"] != "subsync.align":
            emit_result_err(f"unknown step_id: {parsed['step_id']}", out=stdout)
            return 0

        # TODO(Task 4): real behavior. For Task 1 the plugin is a no-op
        # that just acknowledges the protocol.
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
