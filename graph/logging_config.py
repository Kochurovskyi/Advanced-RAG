"""
Logging configuration for the ARAG application.
"""
import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """
    Set up logging configuration for the application.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for the graph module
    logger = logging.getLogger("arag.graph")
    return logger

# Create the main logger
logger = setup_logging()
