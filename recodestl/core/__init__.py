"""Core functionality for RecodeSTL."""

from recodestl.core.config import (
    CacheConfig,
    Config,
    ExportConfig,
    FeatureDetectionConfig,
    LoggingConfig,
    ModelConfig,
    ProcessingConfig,
    SamplingConfig,
    get_default_config,
    load_config,
)
from recodestl.core.converter import ConversionResult, Converter
from recodestl.core.exceptions import (
    CacheError,
    CodeExecutionError,
    CodeGenerationError,
    CodeValidationError,
    ConfigurationError,
    ConverterError,
    DeviceError,
    ExportError,
    InsufficientPointsError,
    MemoryError,
    ModelError,
    PointCloudError,
    RecodeSTLError,
    STLLoadError,
    STLValidationError,
    TimeoutError,
)

__all__ = [
    # Config classes
    "Config",
    "SamplingConfig",
    "ModelConfig",
    "ExportConfig",
    "FeatureDetectionConfig",
    "ProcessingConfig",
    "CacheConfig",
    "LoggingConfig",
    # Config functions
    "get_default_config",
    "load_config",
    # Converter
    "Converter",
    "ConversionResult",
    # Exceptions
    "RecodeSTLError",
    "ConfigurationError",
    "ModelError",
    "STLLoadError",
    "STLValidationError",
    "PointCloudError",
    "InsufficientPointsError",
    "CodeGenerationError",
    "CodeExecutionError",
    "CodeValidationError",
    "ExportError",
    "CacheError",
    "TimeoutError",
    "DeviceError",
    "MemoryError",
    "ConverterError",
]