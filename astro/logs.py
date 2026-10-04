"""Structured JSON logs for remote support: one JSON object per line, one file per server run.

    log.info("said", extra={"data": {"text": "push left"}})
writes {"t": "...", "level": "INFO", "logger": "astro.server", "event": "said", "text": ...}.
Files live in data/logs/ (git-ignored); only the newest KEEP runs are kept.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

KEEP = 20  # server runs (files) to keep
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
    return path
