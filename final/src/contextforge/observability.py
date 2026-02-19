"""
Structured logging utilities.
Prefers structlog when available and falls back to JSON-line stdlib logging.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import structlog
except ImportError:  # pragma: no cover - fallback path exercised when dependency missing
    structlog = None


_CONFIGURED = False


class _FallbackLogger:
    """Minimal structlog-like logger for environments without structlog."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any):
        payload = {"event": str(event), **fields}
        message = json.dumps(payload, ensure_ascii=True, default=str)
        self._logger.log(level, message)

    def info(self, event: str, **fields: Any):
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any):
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any):
        self._emit(logging.ERROR, event, **fields)


def configure_logging(log_path: str | Path):
    """Configures process-wide structured logging to a file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    _CONFIGURED = True


def get_logger(name: str):
    if structlog is not None:
        return structlog.get_logger(name)
    return _FallbackLogger(name)
