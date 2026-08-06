"""Logging infrastructure using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

_initialized = False


def setup_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure application-wide logging with Loguru."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    _logger.remove()

    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(path),
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            rotation="10 MB",
            retention="30 days",
            compression="gz",
        )

    _initialized = True


def get_logger(module_name: str = "portrait_builder") -> _logger.__class__:
    """Return a logger bound to a specific module."""
    return _logger.bind(name=module_name)
