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


# ---- Subtitle path resolution (verbatim from subsync/plugin.py) --------

def find_subtitle_path(
    config: dict,
    ctx: dict,
    video_path: Path,
) -> Path | None:
    """Resolve which .srt to align. Same logic as subsync — operators
    expect identical behavior across both plugins."""
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
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


# ---- Language resolution -----------------------------------------------

def resolve_language(
    config: dict,
    ctx: dict,
    srt_path: Path | None,
    *,
    stdout=None,
) -> str:
    """Resolve the language tag for the wav2vec2 alignment model.

    Priority:
      1. config["language"] override (non-empty, non-"auto")
      2. First step in ctx["steps"] with a `language` field set
      3. Parse from the .srt filename (Movie.en.srt → "en")
      4. Default "en" with a warning log
    """
    override = (config.get("language") or "").strip()
    if override and override != "auto":
        return override

    steps = ctx.get("steps") or {}
    if isinstance(steps, dict):
        for value in steps.values():
            if isinstance(value, dict):
                lang = value.get("language")
                if lang and isinstance(lang, str) and lang != "auto":
                    return lang

    if srt_path is not None:
        # `Movie.en.srt` → split off the .srt → split off the language tag.
        # `with_suffix("")` strips only the rightmost extension (`.srt`).
        without_srt = srt_path.with_suffix("")
        # Now `Movie.en` — the suffix after the last `.` is the language.
        parts = without_srt.name.rsplit(".", 1)
        if len(parts) == 2 and 2 <= len(parts[1]) <= 3 and parts[1].isalpha():
            return parts[1].lower()

    if stdout is not None:
        emit_log(
            "could not detect subtitle language; defaulting to 'en'",
            out=stdout,
        )
    return "en"


# ---- SRT parsing -------------------------------------------------------

def fmt_ts(secs: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm. Verbatim from whisper."""
    ms_total = int(round(secs * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_ts(s: str) -> float:
    """Parse `HH:MM:SS,mmm` (or `HH:MM:SS.mmm`) into seconds."""
    s = s.strip().replace(",", ".")
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def srt_to_segments(srt_text: str) -> list:
    """Parse SRT text into a list of {start, end, text} dicts (the shape
    whisperx.align expects as input). Cue numbers are ignored. Multi-line
    cue text is joined with single spaces."""
    segments = []
    blocks = srt_text.replace("\r\n", "\n").strip().split("\n\n")
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # Line 0 is the cue number (ignored); line 1 is the timestamp;
        # lines 2..N are the text. Some authoring tools omit the cue
        # number — handle that too.
        if "-->" in lines[0]:
            ts_line = lines[0]
            text_lines = lines[1:]
        else:
            ts_line = lines[1]
            text_lines = lines[2:]
        if "-->" not in ts_line:
            continue
        try:
            start_str, end_str = ts_line.split("-->")
            start = parse_srt_ts(start_str)
            end = parse_srt_ts(end_str)
        except (ValueError, IndexError):
            continue
        text = " ".join(t.strip() for t in text_lines)
        if not text:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments


def format_srt_from_aligned(segments: list) -> str:
    """Format whisperx.align's output (segments with per-word timestamps)
    as SRT text. Cue start/end are derived from the first/last aligned
    word; falls back to the segment's own start/end if no words were
    aligned successfully (rare — happens when a segment's text didn't
    phonetically match the audio)."""
    parts = []
    cue_index = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = seg.get("words") or []
        timed = [
            w for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
        if timed:
            start = float(timed[0]["start"])
            end = float(timed[-1]["end"])
        else:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        parts.append(f"{cue_index}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n\n")
        cue_index += 1
    return "".join(parts)


# ---- Atomic in-place replacement (verbatim from subsync) ---------------

def atomic_replace(tmp_path: Path, target_path: Path) -> None:
    """Move tmp_path over target_path atomically. Raises FileNotFoundError
    if tmp_path doesn't exist; the caller is responsible for ensuring
    the tmp file was written before invocation."""
    if not tmp_path.exists():
        raise FileNotFoundError(f"tmp output {tmp_path} not found")
    os.replace(tmp_path, target_path)


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
