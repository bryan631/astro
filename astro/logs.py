"""Structured JSON logs for remote support: one JSON object per line, one file per server run.

    log.info("said", extra={"data": {"text": "push left"}})
writes {"t": "...", "level": "INFO", "logger": "astro.server", "event": "said", "text": ...}.
Files live in data/logs/ (git-ignored); only the newest KEEP runs are kept.
"""

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

KEEP = 20  # server runs (files) to keep: R1's "sessions" are server runs
LOG_DIR = Path("data/logs")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"t": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name, "event": record.getMessage()}
        entry.update(getattr(record, "data", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup(root: Path, keep: int = KEEP) -> Path:
    """Start this run's log file (pruning old ones) and also log to stderr. Returns the path."""
    log_dir = root / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / datetime.now(UTC).strftime("astro-%Y%m%d-%H%M%S-%f.jsonl")
    older = sorted(log_dir.glob("astro-*.jsonl"))  # names sort by time
    for old in older[:max(len(older) - (keep - 1), 0)]:  # leave room for this run's file
        old.unlink()
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(JsonFormatter())
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    root_logger = logging.getLogger("astro")
    root_logger.handlers[:] = [file_handler, console]
    root_logger.setLevel(logging.INFO)
    threading.excepthook = log_thread_crash
    return path


def log_thread_crash(args: threading.ExceptHookArgs) -> None:
    """A background thread died (a solve, a capture, the encoder reader): into this run's log
    with its traceback, not only stderr."""
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread else "?"
    logging.getLogger("astro").error("thread crashed", extra={"data": {"thread": name}},
                                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
