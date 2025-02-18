"""
Error Handler Module

This module provides error handling and logging functionality
for the cryptocurrency trading application.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
import json

class ErrorHandler:
    """Class for handling and logging errors."""
    
    def __init__(self) -> None:
        """Initialize the ErrorHandler with default settings."""
        self.logger = self._setup_logger()
        self.error_count = 0
        self.last_error = None

    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger with file and console handlers.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('crypto_trading')
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(
            f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def log_error(self, error_message: str, error: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Log an error with detailed information.
        
        Args:
            error_message (str): Description of the error
            error (Optional[Exception]): Exception object
            
        Returns:
            Dict[str, Any]: Error details
        """
        try:
            self.error_count += 1
            
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_message': error_message,
                'error_type': type(error).__name__ if error else 'Unknown',
                'error_details': str(error) if error else None,
                'traceback': traceback.format_exc() if error else None,
                'error_count': self.error_count
            }
            
            # Log the error
            self.logger.error(
                f"Error #{self.error_count}: {error_message}",
                extra={'error_details': error_details}
            )
            
            if error:
                self.logger.debug(
                    f"Error details: {str(error)}\nTraceback: {traceback.format_exc()}"
                )
            
            self.last_error = error_details
            return error_details
            
        except Exception as e:
            # Fallback logging if error handling fails
            self.logger.critical(f"Error in error handler: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error_message': 'Error handler failure',
                'error_type': 'ErrorHandlerError',
                'error_details': str(e),
                'error_count': self.error_count
            }

    def log_warning(self, warning_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message with optional details.
        
        Args:
            warning_message (str): Warning message
            details (Optional[Dict[str, Any]]): Additional warning details
        """
        try:
            warning_data = {
                'timestamp': datetime.now().isoformat(),
                'message': warning_message,
                'details': details
            }
            
            self.logger.warning(
                warning_message,
                extra={'warning_data': warning_data}
            )
            
            if details:
                self.logger.debug(f"Warning details: {json.dumps(details, indent=2)}")
                
        except Exception as e:
            self.logger.error(f"Error logging warning: {str(e)}")

    def handle_error(self, error_type: str, error_message: str, error: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle different types of errors with appropriate responses.
        
        Args:
            error_type (str): Type of error (e.g., 'API', 'DATABASE', 'TRADING')
            error_message (str): Error message
            error (Optional[Exception]): Exception object
            
        Returns:
            Dict[str, Any]: Error handling result
        """
        try:
            # Log the error
            error_details = self.log_error(error_message, error)
            
            # Define error handling strategies
            strategies = {
                'API': self._handle_api_error,
                'DATABASE': self._handle_database_error,
                'TRADING': self._handle_trading_error,
                'SYSTEM': self._handle_system_error
            }
            
            # Get appropriate handler or use default
            handler = strategies.get(error_type.upper(), self._handle_default_error)
            
            # Handle the error
            handling_result = handler(error_details)
            
            # Update error details with handling result
            error_details.update({
                'handling_result': handling_result,
                'handled_at': datetime.now().isoformat()
            })
            
            return error_details
            
        except Exception as e:
            self.logger.critical(f"Error in error handler: {str(e)}")
            return {
                'error_type': 'ERROR_HANDLER_FAILURE',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _handle_api_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle API-related errors.
        
        Args:
            error_details (Dict[str, Any]): Error details
            
        Returns:
            Dict[str, Any]: Handling result
        """
        return {
            'action': 'API_ERROR_HANDLED',
            'recommendation': 'Retry with exponential backoff',
            'severity': 'MEDIUM'
        }

    def _handle_database_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle database-related errors.
        
        Args:
            error_details (Dict[str, Any]): Error details
            
        Returns:
            Dict[str, Any]: Handling result
        """
        return {
            'action': 'DATABASE_ERROR_HANDLED',
            'recommendation': 'Check database connection and retry',
            'severity': 'HIGH'
        }

    def _handle_trading_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle trading-related errors.
        
        Args:
            error_details (Dict[str, Any]): Error details
            
        Returns:
            Dict[str, Any]: Handling result
        """
        return {
            'action': 'TRADING_ERROR_HANDLED',
            'recommendation': 'Check trading parameters and market conditions',
            'severity': 'HIGH'
        }

    def _handle_system_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle system-related errors.
        
        Args:
            error_details (Dict[str, Any]): Error details
            
        Returns:
            Dict[str, Any]: Handling result
        """
        return {
            'action': 'SYSTEM_ERROR_HANDLED',
            'recommendation': 'Check system resources and configuration',
            'severity': 'CRITICAL'
        }

    def _handle_default_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle unknown error types.
        
        Args:
            error_details (Dict[str, Any]): Error details
            
        Returns:
            Dict[str, Any]: Handling result
        """
        return {
            'action': 'DEFAULT_ERROR_HANDLED',
            'recommendation': 'Review error details and implement specific handling',
            'severity': 'MEDIUM'
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of error statistics.
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        return {
            'total_errors': self.error_count,
            'last_error': self.last_error,
            'last_updated': datetime.now().isoformat()
        }

    def clear_error_history(self) -> None:
        """Reset error count and last error."""
        self.error_count = 0
        self.last_error = None
        self.logger.info("Error history cleared") 