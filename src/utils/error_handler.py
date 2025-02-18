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
    
    def handle_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Handle and log error."""
        if error:
            self.logger.error(f"{message}: {str(error)}")
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message (str): Warning message to log
        """
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message (str): Info message to log
        """
        self.logger.info(message) 