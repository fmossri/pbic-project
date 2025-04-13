"""
Logger module for the RAG system.

This module provides a centralized logging configuration for the entire system,
supporting both console and file output with structured logging capabilities.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

class Logger:
    """A logger that supports JSON formatting and context information."""
    
    def __init__(self, name: str, domain: str = "default"):
        """
        Initialize the logger.
        
        Args:
            name: The name of the logger (typically the module name)
            domain: The domain context for the logger (e.g., 'public', 'test_domain')
        """
        self.logger = logging.getLogger(name)
        self.domain = domain
        self.context: Dict[str, Any] = {}
        
    def _format_message(self, message: str, level: str, **kwargs) -> str:
        """Format the log message with context and additional fields."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "domain": self.domain,
            "message": message,
            **self.context,
            **kwargs
        }
        return json.dumps(log_data)
    
    def set_context(self, **kwargs) -> None:
        """Set additional context for all subsequent log messages."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context information."""
        self.context.clear()
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message with context."""
        self.logger.info(self._format_message(message, "INFO", **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message with context."""
        self.logger.error(self._format_message(message, "ERROR", **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message with context."""
        self.logger.warning(self._format_message(message, "WARNING", **kwargs))
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message with context."""
        self.logger.debug(self._format_message(message, "DEBUG", **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message with context."""
        self.logger.critical(self._format_message(message, "CRITICAL", **kwargs))

def setup_logging(
    log_dir: str = "logs",
    show_logs: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5  # Keep 5 backup files by default
) -> None:
    """
    Configure the logging system for the application.
    Creates a new log file for each run of the application.
    If the log file grows too large, creates a new one with the same run identifier.
    
    Args:
        log_dir: Directory where log files will be stored
        show_logs: Whether to show logs in console
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep when rotating
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the base log file name
    base_log_file = os.path.join(log_dir, f"rag_system_{run_id}.log")
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if show_logs else logging.ERROR)
    
    # Create formatters
    json_formatter = logging.Formatter('%(message)s')  # For file output
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    file_handler = RotatingFileHandler(
        base_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.DEBUG if show_logs else logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO if show_logs else logging.ERROR) 
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log the configuration
    root_logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": "Logging system configured",
        "run_id": run_id,
        "log_file": base_log_file,
        "show_logs": show_logs,
        "max_file_size": max_file_size,
        "backup_count": backup_count
    }))

def get_logger(name: str, log_domain: str = "default") -> Logger:
    """
    Get a logger instance.
    
    Args:
        name: The name of the logger (typically the module name)
        log_domain: The domain context for the logger
        
    Returns:
        A Logger instance
    """
    return Logger(name, log_domain) 