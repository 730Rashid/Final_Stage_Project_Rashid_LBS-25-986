
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
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If it doesn't exist, create it with default settings.
    
    Args:
        name: Name of the logger (usually __name__ of the module)
    
    Returns:
        Logger instance
    """
    
    return setup_logger(
        name=name,
        log_file=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format_string=config.LOG_FORMAT
    )


class TqdmLoggingHandler(logging.Handler):
    """ Custom logging handler that works nicely with tqdm progress bars. Prevents log messages from breaking the progress bar display. """
    
    # we need to override emit method "


def setup_tqdm_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
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


# Example usage
if __name__ == "__main__":
    # Test the logging setup
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("\nLog file created at: disaster_viz.log")