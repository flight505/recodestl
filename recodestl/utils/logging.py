"""Structured logging configuration using structlog."""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.processors import CallsiteParameter

from recodestl.core.config import LoggingConfig


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(
    config: Optional[LoggingConfig] = None,
    log_file: Optional[Path] = None,
) -> structlog.stdlib.BoundLogger:
    """Set up structured logging with structlog.
    
    Args:
        config: Logging configuration
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = LoggingConfig()
        
    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt=config.timestamp_format)
    
    # Configure processors
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if config.add_caller_info:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.FUNC_NAME,
                ],
            ),
        )
    
    # Set up structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure formatters based on output format
    if config.format == "json":
        formatter = structlog.processors.JSONRenderer()
    elif config.format == "console":
        formatter = structlog.dev.ConsoleRenderer(
            colors=config.colorize and sys.stdout.isatty(),
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:  # plain
        formatter = structlog.processors.KeyValueRenderer(
            key_order=["timestamp", "level", "logger", "event"],
            drop_missing=True,
        )
    
    # Configure standard logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                formatter,
            ],
        )
    )
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, config.level))
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),  # Always use JSON for files
                ],
            )
        )
        root_logger.addHandler(file_handler)
    
    # Configure library loggers
    for lib in ["trimesh", "numpy", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return structlog.get_logger("recodestl")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return structlog.get_logger(name)


def log_performance(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration: float,
    **kwargs: Any,
) -> None:
    """Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    logger.info(
        "performance",
        operation=operation,
        duration_ms=round(duration * 1000, 2),
        **kwargs,
    )


def log_conversion_result(
    logger: structlog.stdlib.BoundLogger,
    result: Any,  # ConversionResult
) -> None:
    """Log conversion result.
    
    Args:
        logger: Logger instance
        result: Conversion result object
    """
    if result.success:
        logger.info(
            "conversion_success",
            input_file=str(result.input_path),
            output_file=str(result.output_path),
            **result.metrics,
        )
    else:
        logger.error(
            "conversion_failed",
            input_file=str(result.input_path),
            error=result.error,
            **result.metrics,
        )


class StructuredLogger:
    """Context manager for structured logging of operations."""
    
    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation: str,
        **context: Any,
    ):
        """Initialize structured logger context.
        
        Args:
            logger: Logger instance
            operation: Operation name
            **context: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self._start_time: Optional[float] = None
        
    def __enter__(self) -> "StructuredLogger":
        """Enter context and log start."""
        import time
        self._start_time = time.time()
        self.logger.info(f"{self.operation}_started", **self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and log completion."""
        import time
        duration = time.time() - self._start_time
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation}_completed",
                duration_ms=round(duration * 1000, 2),
                **self.context,
            )
        else:
            self.logger.error(
                f"{self.operation}_failed",
                duration_ms=round(duration * 1000, 2),
                error=str(exc_val),
                error_type=exc_type.__name__,
                **self.context,
            )
            
    def update_context(self, **kwargs: Any) -> None:
        """Update logging context."""
        self.context.update(kwargs)