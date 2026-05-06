#!/usr/bin/env python3
"""subsync.align — re-time an out-of-sync .srt against the audio of a video.

Pure helpers and protocol orchestration for the subsync plugin. Tests
import this module directly; production runs via bin/run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from glob import glob
from pathlib import Path
from typing import Callable, Optional

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


# ---- Protocol parsing ------------------------------------------------

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


# ---- ffsubsync invocation ----------------------------------------------

# Path to the per-plugin venv's ffsubsync binary, computed relative to
# this file at import time. Indirected through a function so tests can
# monkeypatch it without touching the filesystem.

def _ffsubsync_binary() -> str:
    return str(Path(__file__).resolve().parent / "venv" / "bin" / "ffsubsync")


# tqdm-style progress regex — matches "<digits>%|" anywhere in a line.
# Fits ffsubsync's default tqdm output (`extracting speech segments:
# 22%|##  | 12/55 [00:00<00:01, 30.49it/s]`) and survives any prefix
# tqdm slaps in front of the bar.
_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")


def run_ffsubsync(
    video_path: Path,
    srt_path: Path,
    tmp_out_path: Path,
    *,
    max_offset_seconds: float,
    framerate_correction: bool,
    on_progress: Optional[Callable[[int], None]] = None,
) -> tuple[int, str]:
    """Spawn ffsubsync; stream stderr; return (returncode, captured_stderr).

    The subprocess writes its synced output to `tmp_out_path`; the caller
    is responsible for atomically renaming it over `srt_path` on success.

    Stderr is consumed line-by-line so tqdm progress bars stream in real
    time. Each unique integer percent parsed from a tqdm line is passed
    to `on_progress(pct)` (when supplied). Repeated identical percents
    are de-duplicated so the run timeline doesn't get a hundred copies
    of the same `syncing: 50%` event. The full stderr is still captured
    and returned so the caller can parse the offset/framerate footer
    after exit.

    `bufsize=1` + `text=True` makes the iterator yield on \\r as well as
    \\n, which is exactly how tqdm flushes per-tick updates to a
    non-tty pipe.
    """
    cmd = [
        _ffsubsync_binary(),
        str(video_path),
        "-i", str(srt_path),
        "-o", str(tmp_out_path),
        "--max-offset-seconds", str(int(max_offset_seconds)),
    ]
    if not framerate_correction:
        cmd.append("--no-fix-framerate")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise ProtocolError(
            f"ffsubsync not on path at {_ffsubsync_binary()} "
            f"(plugin install incomplete?)"
        ) from exc

    stderr_lines: list[str] = []
    last_pct = -1
    if proc.stderr is not None:
        for line in proc.stderr:
            stderr_lines.append(line)
            if on_progress is None:
                continue
            m = _PROGRESS_RE.search(line)
            if not m:
                continue
            pct = int(m.group(1))
            if not 0 <= pct <= 100 or pct == last_pct:
                continue
            last_pct = pct
            try:
                on_progress(pct)
            except Exception:  # noqa: BLE001
                # A misbehaving callback must never take the run down.
                pass

    rc = proc.wait()
    return rc, "".join(stderr_lines)


# ---- High-level align orchestration ------------------------------------

def align_subtitle(
    video_path: Path,
    srt_path: Path,
    config: dict,
    *,
    stdout,
    on_progress: Optional[Callable[[int], None]] = None,
) -> dict | None:
    """Run ffsubsync against (video, srt). Apply the result if it's sane.

    Returns the metadata dict to pass into context_set on success, or
    None on a benign skip (warning emitted to stdout already).
    Raises ProtocolError on conditions the caller should turn into
    result:error.

    `on_progress(pct)` is forwarded into `run_ffsubsync`; main() uses it
    to surface live tqdm percentages on the run timeline.
    """
    if not video_path.exists():
        raise ProtocolError(f"video file does not exist: {video_path}")
    if not srt_path.exists():
        emit_log(f"subtitle file does not exist: {srt_path}, skipping", out=stdout)
        return None

    max_offset = float(config.get("max_offset_seconds", 60.0))
    framerate_correction = bool(config.get("framerate_correction", True))
    fail_on_no_match = bool(config.get("fail_on_no_match", False))

    tmp_out = srt_path.with_suffix(srt_path.suffix + ".subsync.tmp.srt")

    rc, stderr = run_ffsubsync(
        video_path,
        srt_path,
        tmp_out,
        max_offset_seconds=max_offset,
        framerate_correction=framerate_correction,
        on_progress=on_progress,
    )

    if rc != 0:
        # Log the last few stderr lines so the operator has *something*
        # to debug from. Then either fail or pass-through per config.
        tail = "\n".join(stderr.strip().splitlines()[-5:])
        emit_log(f"ffsubsync exited rc={rc}: {tail}", out=stdout)
        # Best-effort cleanup; tmp may or may not exist depending on
        # how far ffsubsync got.
        try:
            tmp_out.unlink()
        except FileNotFoundError:
            pass
        if fail_on_no_match:
            raise ProtocolError(f"ffsubsync failed (rc={rc})")
        return None

    offset = parse_offset_from_stderr(stderr)
    if offset is None:
        emit_log(
            "ffsubsync did not emit a parseable offset; leaving original srt",
            out=stdout,
        )
        try:
            tmp_out.unlink()
        except FileNotFoundError:
            pass
        return None

    if abs(offset) > max_offset:
        emit_log(
            f"computed offset {offset:.3f}s exceeds max_offset_seconds "
            f"({max_offset}s); leaving original srt",
            out=stdout,
        )
        try:
            tmp_out.unlink()
        except FileNotFoundError:
            pass
        return None

    framerate_corrected = parse_framerate_corrected_from_stderr(stderr)

    atomic_replace(tmp_out, srt_path)

    return {
        "subtitle_path": str(srt_path),
        "offset_seconds": round(offset, 3),
        "framerate_corrected": framerate_corrected,
    }


# ---- Main entrypoint --------------------------------------------------

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

        video_path = Path(parsed["file_path"])
        srt_path = find_subtitle_path(parsed["config"], parsed["ctx"], video_path)
        if srt_path is None:
            emit_log("no subtitle file found to sync, skipping", out=stdout)
            emit_result_ok(out=stdout)
            return 0

        # The lock serialises every stdout write so progress events
        # and the heartbeat can't interleave bytes. `last_emit` is a
        # 1-element list used as a thread-shared `monotonic()` snapshot
        # — the heartbeat reads it to skip ticks when progress events
        # are already keeping the coordinator's 30s inter-frame timer
        # alive. (Avoids a flood of redundant `syncing...` lines on the
        # run timeline when tqdm percent bumps are already streaming.)
        emit_lock = threading.Lock()
        heartbeat_stop = threading.Event()
        last_emit = [time.monotonic()]

        def _emit_log_safe(msg: str) -> None:
            with emit_lock:
                emit_log(msg, out=stdout)
                stdout.flush()
                last_emit[0] = time.monotonic()

        def _on_progress(pct: int) -> None:
            _emit_log_safe(f"syncing: {pct}%")

        def _heartbeat() -> None:
            # Tick every second; only emit when emit-quiet for the
            # full HEARTBEAT_INTERVAL_SECS window so progress events
            # crowd out the keep-alive when both are active.
            while not heartbeat_stop.wait(1.0):
                if time.monotonic() - last_emit[0] >= HEARTBEAT_INTERVAL_SECS:
                    _emit_log_safe("syncing...")

        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            meta = align_subtitle(
                video_path, srt_path, parsed["config"],
                stdout=stdout, on_progress=_on_progress,
            )
        except ProtocolError as exc:
            emit_result_err(str(exc), out=stdout)
            return 0
        finally:
            heartbeat_stop.set()

        if meta is not None:
            emit_context_set("subsync", meta, out=stdout)
        emit_result_ok(out=stdout)
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        emit_result_err(f"unexpected error: {exc}", out=stdout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
