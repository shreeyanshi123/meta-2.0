"""Structured logging utilities for the Tribunal environment."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging with Rich for the Tribunal."""
    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
