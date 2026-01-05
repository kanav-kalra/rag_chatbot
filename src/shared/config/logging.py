"""
Logger Configuration
"""
import logging
import sys
from pathlib import Path

from src.shared.config.settings import settings


def setup_logger(name: str = "rag_chatbot", log_level: str = None) -> logging.Logger:
    """
    Set up and configure application logger.
    
    Args:
        name: Logger name (default: "rag_chatbot")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   If None, uses DEBUG if settings.DEBUG is True, else INFO
        
    Returns:
        Configured logger instance
    """
    # Determine log level
    if log_level is None:
        log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    else:
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional - create logs directory if it doesn't exist)
    try:
        # Path from src/shared/config/logging.py -> data/logs/
        # Go up: config -> shared -> src -> project_root -> data/logs
        log_dir = Path(__file__).parent.parent.parent.parent / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue without it
        pass
    
    return logger


# Create default logger instance
logger = setup_logger()

