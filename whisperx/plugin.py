#!/usr/bin/env python3
"""whisperx — forced phoneme alignment for whisper output.

Two step names from one plugin:
  - whisperx.align: re-times an existing .srt against the audio
  - whisperx.transcribe_aligned: full pipeline (transcribe + align)

Pure helpers and protocol orchestration. Tests import this module
directly; production runs via bin/run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from glob import glob
from pathlib import Path

# Heartbeat cadence for the dispatcher's inter-frame timer. The
# transcoderr coordinator drops a remote step that goes 30s without
# emitting any frame; wav2vec2 inference goes silent during forward
# passes (and during model loading) easily longer than that.
HEARTBEAT_INTERVAL_SECS = 10.0

# Default config for whisperx.align.
DEFAULT_ALIGN_CONFIG = {
    "subtitle_path": "",
    "language": "",
    "alignment_model": "",
    "compute_type": "auto",
    "fail_on_no_match": False,
}

# Default config for whisperx.transcribe_aligned. Includes the whisper
# transcription knobs in addition to alignment knobs.
DEFAULT_TRANSCRIBE_ALIGNED_CONFIG = {
    "model": "large-v3-turbo",
    "language": "auto",
    "alignment_model": "",
    "compute_type": "auto",
    "batch_size": 16,
    "skip_if_exists": True,
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


# ---- Protocol parsing --------------------------------------------------

def parse_execute(line: str) -> dict:
    """Parse a JSON-RPC execute line. Returns {step_id, file_path, ctx, config}.

    Mirrors whisper/plugin.py:parse_execute. Tolerates both the production
    nested form (`params.{step_id, with, context}`) and the legacy flat
    form (`{step_id, ctx, config}`) so hand-crafted test fixtures still
    parse. Picks default config based on step_id.
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

    if step_id == "whisperx.align":
        config = {**DEFAULT_ALIGN_CONFIG, **user_config}
    elif step_id == "whisperx.transcribe_aligned":
        config = {**DEFAULT_TRANSCRIBE_ALIGNED_CONFIG, **user_config}
    else:
        # Unknown step_id; main() will reject. Pass user_config through
        # without merging defaults so the rejection path stays clean.
        config = dict(user_config)

    return {
        "step_id": step_id,
        "file_path": file_path,
        "ctx": context,
        "config": config,
    }


# ---- Main entrypoint ---------------------------------------------------

def main(stdin=None, stdout=None) -> int:
    """Read init+execute from stdin, dispatch on step_id, emit events to stdout."""
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

        if parsed["step_id"] not in ("whisperx.align", "whisperx.transcribe_aligned"):
            emit_result_err(f"unknown step_id: {parsed['step_id']}", out=stdout)
            return 0

        # TODO(Task 5): real behavior. For Task 1 the plugin is a no-op
        # that just acknowledges the protocol.
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
