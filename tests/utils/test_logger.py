"""
Test module for the logger functionality.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pytest
from src.utils.logger import setup_logging, get_logger, Logger
import time

class TestLogger:
    """Test suite for the logger functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test."""
        # Create a test log directory
        self.test_log_dir = "test_logs"
        os.makedirs(self.test_log_dir, exist_ok=True)
        
        # Cleanup after test
        yield
        # Remove all log files after each test
        for log_file in Path(self.test_log_dir).glob("rag_system_*.log*"):
            try:
                log_file.unlink()
            except Exception:
                pass

    def get_latest_log_file(self):
        """Get the most recent log file in the test directory."""
        log_files = list(Path(self.test_log_dir).glob("rag_system_*.log"))
        return max(log_files, key=lambda x: x.stat().st_mtime) if log_files else None

    def test_logger_initialization(self):
        """Test the basic initialization of the logger."""
        print("\nRunning initialization test...")
        # Setup logging in the test directory with DEBUG level
        setup_logging(log_dir=self.test_log_dir, show_logs=True)
        
        # Get a logger instance and set level to DEBUG
        logger = get_logger(__name__, domain="test")
        logger.logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG
        
        # Test logging different levels
        test_message = "Test log message"
        logger.info(test_message)
        logger.error(test_message)
        logger.warning(test_message)
        logger.debug(test_message)
        logger.critical(test_message)
        
        # Verify log file was created
        log_file = self.get_latest_log_file()
        assert log_file is not None, "No log file was created"
        print(f"Created log file: {log_file}")
        
        # Verify log entries
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message
            log_entries = [json.loads(line) for line in lines[1:]]
            assert len(log_entries) == 5, "Should have 5 log entries"
            
            # Verify all log levels are present
            levels = {entry["level"] for entry in log_entries}
            assert levels == {"INFO", "ERROR", "WARNING", "DEBUG", "CRITICAL"}, "Missing some log levels"

    def test_logger_context(self):
        """Test the context functionality of the logger."""
        print("\nRunning context test...")
        setup_logging(log_dir=self.test_log_dir, show_logs=True)
        logger = get_logger(__name__, domain="test")
        
        # Set context and log
        logger.set_context(user_id="123", action="test")
        logger.info("Context test")
        
        # Read the log file
        log_file = self.get_latest_log_file()
        assert log_file is not None, "No log file was created"
        print(f"Created log file: {log_file}")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message
            log_entry = json.loads(lines[1])
            
            # Verify context was included
            assert log_entry["user_id"] == "123"
            assert log_entry["action"] == "test"

    def test_logger_file_rotation(self):
        """Test that log files are rotated when they reach the size limit."""
        # Configure logger with 100 bytes max file size
        max_file_size = 400
        setup_logging(
            log_dir=self.test_log_dir,
            show_logs=True,
            max_file_size=max_file_size,
            backup_count=5
        )
        
        message = "test"

        # Log first message and check file size
        logger = get_logger("test_rotation", domain="test")
        logger.info(message)
        
        # Force flush and check size
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Get files after first message
        log_files = sorted([f for f in os.listdir(self.test_log_dir)])

        assert len(log_files) == 1, f"Expected exactly 1 log file after first message, found: {log_files}"
        
        # Log second message which should trigger truncation
        logger.info(message)
        
        # Force flush
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Get files after second message
        log_files = sorted([f for f in os.listdir(self.test_log_dir)])
        
        assert len(log_files) == 2, f"Expected exactly 2 log files after second message, found: {log_files}"

        latest_file = os.path.join(self.test_log_dir, log_files[0])

        # Verify that the file contains valid JSON
        with open(latest_file, 'r') as f:
            content = f.read()
            # Each line should be valid JSON
            for line in content.splitlines():
                json_obj = json.loads(line)
                assert isinstance(json_obj, dict), "Log line is not valid JSON"

    def test_logger_domain(self):
        """Test that the domain is correctly included in log messages."""
        print("\nRunning domain test...")
        setup_logging(log_dir=self.test_log_dir, show_logs=True)
        
        # Test different domains
        test_domains = ["public", "test_domain", "another_domain"]
        for domain in test_domains:
            logger = get_logger(__name__, domain=domain)
            logger.info("Domain test")
        
        # Verify domain in log entries
        log_file = self.get_latest_log_file()
        assert log_file is not None, "No log file was created"
        print(f"Created log file: {log_file}")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message
            log_entries = [json.loads(line) for line in lines[1:4]]
            
            # Verify each domain was logged correctly
            assert {entry["domain"] for entry in log_entries} == set(test_domains) 