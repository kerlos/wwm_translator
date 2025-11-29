from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table


if TYPE_CHECKING:
    from typing import Any


console = Console(force_terminal=True, legacy_windows=False, highlight=False)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configure logging with Rich handler.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_to_console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger("wwm_translator")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    if log_to_console:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def create_progress() -> Progress:
    """Create Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


@lru_cache(maxsize=1)
def get_banner() -> str:
    """Get application banner (ASCII-safe for Windows)."""
    return """
============================================================
       WWM Translator - Where Winds Meet
       Neural Translation Tool
============================================================
"""


def print_banner() -> None:
    """Print application banner."""
    console.print(get_banner(), style="bold cyan")


def print_table(title: str, data: dict[str, Any]) -> None:
    """Print data as Rich table."""
    table = Table(title=title, show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    console.print(table)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"
    hours, remainder = divmod(seconds, 3600)
    mins = remainder // 60
    return f"{int(hours)}h {int(mins)}m"


def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    """Create safe filename by removing invalid characters."""
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, "_")
    return name


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    from rich.prompt import Confirm

    return Confirm.ask(message, default=default)


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]OK[/green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]Error:[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")
