"""
Logging Configuration Module.

Provides utility functions for setting up logging throughout the project.
Supports both console and file logging, with special handling for tqdm
progress bars to avoid display issues.

"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config.settings import config


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: The name of the logger.
        log_file: The file path to write logs to. If None, only console
                  logging is used.
        level: The logging level (e.g. logging.INFO, logging.DEBUG).
        format_string: A custom format string for log messages.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    # Use a default format string if none is provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add a console handler to output logs to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add a file handler if a log file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger instance with predefined settings from the config.

    This is the main function to use when you need a logger in any module.
    It uses the settings defined in config/settings.py.

    Args:
        name: The name for the logger, typically the module's __name__.

    Returns:
        A configured logger instance.
    """
    return setup_logger(
        name=name,
        log_file=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format_string=config.LOG_FORMAT
    )


class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that works with tqdm progress bars.
    
    When using tqdm for progress indication, standard logging can
    disrupt the progress bar display. This handler uses tqdm.write()
    to output log messages without interfering with progress bars.
    """
    
    def emit(self, record):
        """
        Format and write the log record.
        
        Uses tqdm.write() to ensure the message does not interfere
        with any active progress bars.
        """
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def setup_tqdm_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that is compatible with tqdm progress bars.

    Use this instead of get_logger() when your code uses tqdm
    progress bars extensively.

    Args:
        name: The name of the logger.
        level: The logging level.

    Returns:
        A configured logger instance compatible with tqdm.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    # Demonstration of the logging module
    
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message for detailed diagnostics.")
    logger.info("This is an informational message about normal operation.")
    logger.warning("This is a warning message about a potential issue.")
    logger.error("This is an error message about a failure.")
    logger.critical("This is a critical message about a severe failure.")
    
    print("")
    print("Log file created at: {}".format(config.LOG_FILE))