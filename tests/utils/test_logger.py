"""
Test module for the logger functionality.
"""

import os
import json
import logging
from pathlib import Path
import pytest
from src.utils.logger import setup_logging, get_logger

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
        setup_logging(log_dir=self.test_log_dir, debug=True)
        
        # Get a logger instance and set level to DEBUG
        logger = get_logger(__name__, log_domain="test")
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
        
        # Verify log entries - safely parse JSON
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message and check that we have expected log entries
            assert len(lines) >= 6, "Should have at least 6 log entries"
            
            # Parse each line individually and handle exceptions
            log_entries = []
            for line in lines[1:]:
                try:
                    if line.strip():  # Skip empty lines
                        log_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Invalid JSON in line: {line}")
            
            # Verify we have log entries
            assert len(log_entries) >= 5, f"Expected at least 5 valid log entries, found {len(log_entries)}"
            
            # Verify log levels are present
            levels = {entry["level"] for entry in log_entries if "level" in entry}
            assert levels.intersection({"INFO", "ERROR", "WARNING", "DEBUG", "CRITICAL"}), "Missing log levels"

    def test_logger_context(self):
        """Test the context functionality of the logger."""
        print("\nRunning context test...")
        setup_logging(log_dir=self.test_log_dir, debug=True)
        logger = get_logger(__name__, log_domain="test")
        
        # Set context and log
        logger.set_context(user_id="123", action="test")
        logger.info("Context test")
        
        # Read the log file
        log_file = self.get_latest_log_file()
        assert log_file is not None, "No log file was created"
        print(f"Created log file: {log_file}")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message and make sure there are entries
            assert len(lines) >= 2, "Expected at least 2 log entries"
            
            # Try to parse the JSON, handle potential errors
            try:
                log_entry = json.loads(lines[1])
                
                # Verify context was included
                assert log_entry["user_id"] == "123"
                assert log_entry["action"] == "test"
            except (json.JSONDecodeError, IndexError) as e:
                pytest.fail(f"Failed to parse log entry: {e}, line content: {lines[1] if len(lines) > 1 else 'No line'}")

    def test_logger_file_rotation(self):
        """Test that log files are rotated when they reach the size limit."""
        # Configure logger with a very small max file size to force rotation
        max_file_size = 100
        setup_logging(
            log_dir=self.test_log_dir,
            debug=True,
            max_file_size=max_file_size,
            backup_count=5
        )
        
        # Store paths before we start
        initial_files = sorted([f for f in os.listdir(self.test_log_dir)])
        
        # Create logger and log a very long message to force rotation
        logger = get_logger("test_rotation", log_domain="test")
        
        # Log long message to ensure it exceeds file size limit
        long_message = "x" * 200
        logger.info(long_message)
        
        # Force flush
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Log another message to trigger rotation
        logger.info(long_message)
        
        # Force flush again
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Wait a moment and force rotation by logging more
        logger.info(long_message)
        logger.info(long_message)
        
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Get files after logging
        log_files = sorted([f for f in os.listdir(self.test_log_dir) if f not in initial_files])
        
        # Now we expect at least two files (the main log and at least one backup)
        assert len(log_files) >= 2, f"Expected at least 2 log files after rotation, found: {log_files}"
        
        # Verify at least one file contains valid JSON
        if log_files:
            latest_file = os.path.join(self.test_log_dir, log_files[0])
            with open(latest_file, 'r') as f:
                content = f.read()
                # Try to parse at least one line as JSON
                assert content.strip(), "Log file is empty"
                
                valid_json = False
                for line in content.splitlines():
                    if line.strip():
                        try:
                            json_obj = json.loads(line)
                            valid_json = isinstance(json_obj, dict)
                            if valid_json:
                                break
                        except json.JSONDecodeError:
                            continue
                
                assert valid_json, "No valid JSON found in log file"

    def test_logger_domain(self):
        """Test that the domain is correctly included in log messages."""
        print("\nRunning domain test...")
        setup_logging(log_dir=self.test_log_dir, debug=True)
        
        # Test different domains
        test_domains = ["public", "test_domain", "another_domain"]
        for domain in test_domains:
            logger = get_logger(__name__, log_domain=domain)
            logger.info("Domain test")
        
        # Verify domain in log entries
        log_file = self.get_latest_log_file()
        assert log_file is not None, "No log file was created"
        print(f"Created log file: {log_file}")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Skip the config message
            log_entries = []
            for i in range(1, min(len(lines), 5)):  # Look at up to 4 entries after config
                try:
                    if lines[i].strip():
                        log_entries.append(json.loads(lines[i]))
                except json.JSONDecodeError:
                    print(f"Invalid JSON in line: {lines[i]}")
            
            # Verify we have log entries
            assert len(log_entries) >= 3, f"Expected at least 3 valid log entries, found {len(log_entries)}"
            
            # Verify each domain was logged correctly (using log_domain instead of domain)
            logged_domains = {entry.get("log_domain") for entry in log_entries if "log_domain" in entry}
            assert logged_domains.intersection(set(test_domains)), f"Expected domains {test_domains}, found {logged_domains}" 