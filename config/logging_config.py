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
        name (str): The name of the logger.
        log_file (Optional[Path]): The file path to write logs to. If None, only console logging is used.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        format_string (Optional[str]): A custom format string for log messages.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    # Use a default format string if none is provided
    if format_string is None:
        format_string = "% (asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add a console handler to output logs to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add a file handler if a log file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with predefined settings from the global config.

    Args:
        name (str): The name for the logger, typically the module's __name__.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return setup_logger(
        name=name,
        log_file=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format_string=config.LOG_FORMAT
    )


class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that integrates with the tqdm progress bar library.
    This handler prevents log messages from disrupting the visual display of tqdm bars.
    """
    def emit(self, record):
        """
        Formats and writes the log record.
        This method is overridden to ensure that log messages are written to the console
        without interfering with the tqdm progress bar.
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
    Sets up a logger that is compatible with tqdm progress bars.

    Args:
        name (str): The name of the logger.
        level (int): The logging level.

    Returns:
        logging.Logger: A configured logger instance compatible with tqdm.
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

    # Use the custom TqdmLoggingHandler
    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# --- Example Usage ---
if __name__ == "__main__":
    # This block demonstrates how to use the get_logger function.
    # It will only run when the script is executed directly.
    
    # Get a logger for this module
    logger = get_logger(__name__)
    
    # Log messages at different severity levels
    logger.debug("This is a debug message for detailed diagnostics.")
    logger.info("This is an informational message about normal operation.")
    logger.warning("This is a warning message about a potential issue.")
    logger.error("This is an error message about a failure.")
    logger.critical("This is a critical message about a severe failure.")
    
    print(f"\nLog file created at: {config.LOG_FILE}")