"""
Logging utilities for Athena MoE system.
"""

import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


console = Console()


def setup_logger(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = None,
    rotation: str = "10 MB"
) -> None:
    """
    Configure the application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        format_string: Custom format string
        rotation: Log rotation size
    """
    # Remove default logger
    logger.remove()

    # Default format if not provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add console logger with colors
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file logger if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=format_string,
            rotation=rotation,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logger configured: level={level}, file={log_file}")


def log_query_start(query_id: str, content: str) -> None:
    """Log the start of query processing."""
    console.print(f"\n[bold blue]Query {query_id}[/bold blue]")
    console.print(f"[dim]{content}[/dim]\n")


def log_expert_consultation(expert_name: str, status: str = "consulting") -> None:
    """Log expert consultation."""
    if status == "consulting":
        console.print(f"  [yellow]→[/yellow] Consulting {expert_name} expert...")
    elif status == "complete":
        console.print(f"  [green]✓[/green] {expert_name} expert response received")
    elif status == "error":
        console.print(f"  [red]✗[/red] {expert_name} expert error")


def log_synthesis(confidence: float) -> None:
    """Log response synthesis."""
    console.print(f"\n[bold green]Response synthesized[/bold green] (confidence: {confidence:.2%})\n")


def log_workspace_update(workspace_summary: str) -> None:
    """Log workspace state update."""
    logger.debug(f"Workspace: {workspace_summary}")
