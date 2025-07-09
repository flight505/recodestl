"""Custom exceptions for RecodeSTL."""

from pathlib import Path
from typing import Any, Optional


class RecodeSTLError(Exception):
    """Base exception for RecodeSTL."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(RecodeSTLError):
    """Raised when configuration is invalid."""

    pass


class ModelError(RecodeSTLError):
    """Raised when model operations fail."""

    pass


class STLLoadError(RecodeSTLError):
    """Raised when STL file cannot be loaded."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"Failed to load STL file '{path}': {reason}")
        self.path = path
        self.reason = reason


class STLValidationError(RecodeSTLError):
    """Raised when STL file validation fails."""

    def __init__(self, path: Path, errors: list[str]):
        message = f"STL validation failed for '{path}': {', '.join(errors)}"
        super().__init__(message)
        self.path = path
        self.errors = errors


class PointCloudError(RecodeSTLError):
    """Raised when point cloud generation fails."""

    pass


class InsufficientPointsError(PointCloudError):
    """Raised when not enough points can be sampled."""

    def __init__(self, requested: int, available: int):
        super().__init__(
            f"Requested {requested} points but only {available} available"
        )
        self.requested = requested
        self.available = available


class CodeGenerationError(RecodeSTLError):
    """Raised when CAD code generation fails."""

    pass


class CodeExecutionError(RecodeSTLError):
    """Raised when generated code execution fails."""

    def __init__(self, code: str, error: str):
        super().__init__(f"Code execution failed: {error}")
        self.code = code
        self.error = error


class CodeValidationError(RecodeSTLError):
    """Raised when generated code validation fails."""

    def __init__(self, code: str, reason: str):
        super().__init__(f"Code validation failed: {reason}")
        self.code = code
        self.reason = reason


class ExportError(RecodeSTLError):
    """Raised when STEP export fails."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"Failed to export STEP file '{path}': {reason}")
        self.path = path
        self.reason = reason


class CacheError(RecodeSTLError):
    """Raised when cache operations fail."""

    pass


class TimeoutError(RecodeSTLError):
    """Raised when operation times out."""

    def __init__(self, operation: str, timeout: int):
        super().__init__(f"{operation} timed out after {timeout} seconds")
        self.operation = operation
        self.timeout = timeout


class DeviceError(RecodeSTLError):
    """Raised when device is not available."""

    def __init__(self, device: str, reason: str):
        super().__init__(f"Device '{device}' not available: {reason}")
        self.device = device
        self.reason = reason


class MemoryError(RecodeSTLError):
    """Raised when memory limit is exceeded."""

    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient memory: {required:.1f}GB required, {available:.1f}GB available"
        )
        self.required = required
        self.available = available