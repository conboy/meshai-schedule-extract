# Centralized logging configuration
import logging
import logging.config
import os
from config import LOGGING_CONFIG, LOG_DIR, LOG_FILE

def setup_logging():
    """Setup logging configuration for the application."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: The name for the logger (typically __name__)
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)

setup_logging()