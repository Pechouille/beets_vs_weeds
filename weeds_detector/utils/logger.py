import logging
import socket
import time
from pathlib import Path

# Configure logging
def setup_logging():
    """Setup logging configuration with both file and console handlers"""
    hostname = socket.gethostname()
    log_filename = f"cropping_log_{hostname}_{int(time.time())}.log"

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(hostname)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_filename),
            logging.StreamHandler()  # Console output
        ]
    )

    # Add hostname to log records
    class HostnameFilter(logging.Filter):
        def filter(self, record):
            record.hostname = hostname
            return True

    for handler in logging.getLogger().handlers:
        handler.addFilter(HostnameFilter())

    return logging.getLogger(__name__)

#define the logger for all clients
# import the logger using
# from .logger import logger
logger = setup_logging()

