"""
Logging configuration for OpenMisty.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as loguru_logger


# Get the project root directory
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
LOG_DIR = PROJECT_ROOT / "logs"

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure loguru logger
loguru_logger.remove()  # Remove the default handler

# Add a handler for stdout with colors
loguru_logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Add a handler for file logging
log_file = LOG_DIR / f"openmisty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
loguru_logger.add(
    log_file,
    rotation="10 MB",
    retention="1 week",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Export the configured logger
logger = loguru_logger 