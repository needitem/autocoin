"""
Error Handler

This module handles error logging and management.
"""

import logging
from typing import Optional
import traceback

class ErrorHandler:
    """Class for handling application errors."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/trading.log')
            ]
        )
    
    def handle_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Handle and log error."""
        if error:
            self.logger.error(f"{message}: {str(error)}")
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message) 