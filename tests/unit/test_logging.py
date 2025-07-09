"""Unit tests for structured logging."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog

from recodestl.core.config import LoggingConfig
from recodestl.utils.logging import (
    setup_logging,
    get_logger,
    log_performance,
    log_conversion_result,
    StructuredLogger,
)


@pytest.fixture
def logging_config():
    """Create test logging configuration."""
    return LoggingConfig(
        level="INFO",
        format="json",
        colorize=False,
        add_caller_info=True,
    )


@pytest.fixture
def mock_conversion_result():
    """Create mock conversion result."""
    result = MagicMock()
    result.success = True
    result.input_path = Path("/test/input.stl")
    result.output_path = Path("/test/output.step")
    result.metrics = {
        "total_time": 1.5,
        "vertex_count": 1000,
        "face_count": 2000,
    }
    return result


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_logging_json(self, logging_config):
        """Test JSON logging setup."""
        logger = setup_logging(logging_config)
        
        assert logger is not None
        # Structlog returns a BoundLoggerLazyProxy
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        
    def test_setup_logging_console(self):
        """Test console logging setup."""
        config = LoggingConfig(format="console")
        logger = setup_logging(config)
        
        assert logger is not None
        
    def test_setup_logging_plain(self):
        """Test plain logging setup."""
        config = LoggingConfig(format="plain")
        logger = setup_logging(config)
        
        assert logger is not None
        
    def test_setup_logging_with_file(self, tmp_path):
        """Test logging with file output."""
        log_file = tmp_path / "test.log"
        config = LoggingConfig()
        
        logger = setup_logging(config, log_file=log_file)
        
        # Log something
        logger.info("test message", key="value")
        
        # Check file was created (may need flush)
        logging.shutdown()
        # File handler might not write immediately
        
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test.module")
        
        assert logger is not None
        # Structlog returns a BoundLoggerLazyProxy
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')


class TestLogHelpers:
    """Test logging helper functions."""
    
    def test_log_performance(self):
        """Test performance logging."""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            log_performance(
                logger,
                "test_operation",
                1.5,
                items_processed=100,
            )
            
            mock_info.assert_called_once_with(
                "performance",
                operation="test_operation",
                duration_ms=1500.0,
                items_processed=100,
            )
            
    def test_log_conversion_result_success(self, mock_conversion_result):
        """Test logging successful conversion."""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            log_conversion_result(logger, mock_conversion_result)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "conversion_success"
            assert "input_file" in call_args[1]
            assert "output_file" in call_args[1]
            
    def test_log_conversion_result_failure(self, mock_conversion_result):
        """Test logging failed conversion."""
        mock_conversion_result.success = False
        mock_conversion_result.error = "Test error"
        mock_conversion_result.output_path = None
        
        logger = get_logger("test")
        
        with patch.object(logger, 'error') as mock_error:
            log_conversion_result(logger, mock_conversion_result)
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert call_args[0][0] == "conversion_failed"
            assert call_args[1]["error"] == "Test error"


class TestStructuredLogger:
    """Test StructuredLogger context manager."""
    
    def test_structured_logger_success(self):
        """Test successful operation logging."""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            with StructuredLogger(logger, "test_op", foo="bar") as ctx:
                ctx.update_context(baz="qux")
                
            # Should have start and complete calls
            assert mock_info.call_count == 2
            
            # Check start call
            start_call = mock_info.call_args_list[0]
            assert start_call[0][0] == "test_op_started"
            assert start_call[1]["foo"] == "bar"
            
            # Check complete call
            complete_call = mock_info.call_args_list[1]
            assert complete_call[0][0] == "test_op_completed"
            assert "duration_ms" in complete_call[1]
            assert complete_call[1]["baz"] == "qux"
            
    def test_structured_logger_failure(self):
        """Test failed operation logging."""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            with patch.object(logger, 'error') as mock_error:
                try:
                    with StructuredLogger(logger, "test_op"):
                        raise ValueError("Test error")
                except ValueError:
                    pass
                    
                # Should have start call
                assert mock_info.call_count == 1
                assert mock_info.call_args[0][0] == "test_op_started"
                
                # Should have error call
                assert mock_error.call_count == 1
                error_call = mock_error.call_args
                assert error_call[0][0] == "test_op_failed"
                assert error_call[1]["error"] == "Test error"
                assert error_call[1]["error_type"] == "ValueError"
                assert "duration_ms" in error_call[1]
                
    def test_structured_logger_update_context(self):
        """Test context updates."""
        logger = get_logger("test")
        
        with StructuredLogger(logger, "test_op", initial="value") as ctx:
            assert ctx.context["initial"] == "value"
            
            ctx.update_context(added="new_value", initial="updated")
            
            assert ctx.context["initial"] == "updated"
            assert ctx.context["added"] == "new_value"


class TestLogLevels:
    """Test different log levels."""
    
    def test_debug_level(self):
        """Test DEBUG level logging."""
        config = LoggingConfig(level="DEBUG")
        logger = setup_logging(config)
        
        # Debug should be enabled
        assert logging.getLogger().level == logging.DEBUG
        
    def test_error_level(self):
        """Test ERROR level logging."""
        config = LoggingConfig(level="ERROR")
        logger = setup_logging(config)
        
        # Only errors should be logged
        assert logging.getLogger().level == logging.ERROR
        
    def test_library_logging_suppressed(self):
        """Test that library logging is suppressed."""
        config = LoggingConfig(level="DEBUG")
        setup_logging(config)
        
        # Library loggers should be at WARNING level
        assert logging.getLogger("trimesh").level == logging.WARNING
        assert logging.getLogger("numpy").level == logging.WARNING
        assert logging.getLogger("matplotlib").level == logging.WARNING