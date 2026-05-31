"""Project logging configuration. Call configure_logging() once from main."""
import logging
import logging.handlers

from .settings import LOGS_DIR, RUN_LOG


_FMT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with stdout (level) and rotating file (DEBUG)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop any prior handlers (re-running in REPL etc).
    root.handlers.clear()

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(level.upper())
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        RUN_LOG, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers.
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
