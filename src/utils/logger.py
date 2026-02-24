"""
Logging utilities using loguru.

This module provides centralized logging configuration for the entire application.
It supports both console and file logging with rotation, compression, and
customizable formatting.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None,
    colorize: bool = True,
) -> None:
    """
    Configure the global logger with console and file outputs.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is enabled
        rotation: When to rotate log files (e.g., "10 MB", "1 day", "12:00")
        retention: How long to keep old log files (e.g., "1 week", "30 days")
        format_string: Custom format string. If None, uses default format
        colorize: Whether to colorize console output

    Example:
        >>> setup_logger(
        ...     level="DEBUG",
        ...     log_file="logs/app.log",
        ...     rotation="10 MB",
        ...     retention="1 week"
        ... )
    """
    # Remove default handler
    logger.remove()

    # Default format if none provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",  # Compress rotated files
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe logging
        )

    logger.info(f"Logger initialized with level={level}, log_file={log_file}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to the logger for context

    Returns:
        Logger instance

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Processing started")
        >>> log.error("An error occurred", extra={"user_id": 123})
    """
    if name:
        return logger.bind(name=name)
    return logger


def configure_from_dict(config: dict) -> None:
    """
    Configure logger from a configuration dictionary.

    This is useful when loading configuration from YAML or JSON files.

    Args:
        config: Dictionary containing logging configuration with keys:
            - level: str (default: "INFO")
            - format: str (optional)
            - file: str (optional)
            - rotation: str (default: "10 MB")
            - retention: str (default: "1 week")

    Example:
        >>> config = {
        ...     "level": "DEBUG",
        ...     "file": "logs/app.log",
        ...     "rotation": "10 MB",
        ...     "retention": "1 week"
        ... }
        >>> configure_from_dict(config)
    """
    level = config.get("level", "INFO")
    log_file = config.get("file")
    rotation = config.get("rotation", "10 MB")
    retention = config.get("retention", "1 week")
    format_string = config.get("format")

    setup_logger(
        level=level,
        log_file=log_file,
        rotation=rotation,
        retention=retention,
        format_string=format_string,
    )


# Export commonly used logger instance
__all__ = ["logger", "setup_logger", "get_logger", "configure_from_dict"]
