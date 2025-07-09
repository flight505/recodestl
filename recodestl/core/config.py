"""Configuration management for RecodeSTL using Pydantic."""

from pathlib import Path
from typing import Literal, Optional

import tomli
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SamplingConfig(BaseModel):
    """Configuration for point cloud sampling."""

    model_config = ConfigDict(frozen=True)

    num_points: int = Field(256, ge=10, le=10000, description="Number of points to sample")
    method: Literal["uniform", "poisson", "adaptive"] = Field(
        "adaptive", description="Sampling method to use"
    )
    pre_samples: int = Field(
        8192, ge=1000, description="Number of initial samples before filtering"
    )
    curvature_radius: float = Field(
        0.1, gt=0, description="Radius for curvature computation"
    )
    feature_multiplier: int = Field(
        5, ge=1, description="Multiplier for feature-rich areas"
    )
    curvature_weight: float = Field(
        0.7, ge=0, le=1, description="Weight for high-curvature areas in adaptive sampling"
    )

    @field_validator("pre_samples")
    @classmethod
    def validate_pre_samples(cls, v: int, info) -> int:
        """Ensure pre_samples is greater than num_points."""
        if "num_points" in info.data and v < info.data["num_points"]:
            raise ValueError("pre_samples must be greater than num_points")
        return v


class ModelConfig(BaseModel):
    """Configuration for CAD-Recode model."""

    model_config = ConfigDict(frozen=True)

    device: Literal["cpu", "cuda", "mps"] = Field(
        "mps", description="Device to use for inference"
    )
    max_tokens: int = Field(768, ge=100, description="Maximum tokens to generate")
    temperature: float = Field(
        0.0, ge=0, le=1, description="Temperature for generation (0 = deterministic)"
    )
    use_flash_attention: bool = Field(
        False, description="Use flash attention (not available on MPS)"
    )
    model_name: str = Field(
        "filapro/cad-recode-v1.5", description="HuggingFace model name"
    )
    tokenizer_name: str = Field("Qwen/Qwen2-1.5B", description="Tokenizer name")
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = Field(
        "auto", description="Model dtype (auto = device-specific)"
    )


class ExportConfig(BaseModel):
    """Configuration for STEP file export."""

    model_config = ConfigDict(frozen=True)

    step_precision: float = Field(
        0.001, gt=0, description="Precision for STEP file export"
    )
    angular_tolerance: float = Field(
        0.1, gt=0, description="Angular tolerance in degrees"
    )
    validate_output: bool = Field(True, description="Validate generated STEP files")


class FeatureDetectionConfig(BaseModel):
    """Configuration for mechanical feature detection."""

    model_config = ConfigDict(frozen=True)

    edge_angle_threshold: float = Field(
        4.0, gt=0, description="Angle threshold for edge detection (degrees)"
    )
    angle_defect_threshold: float = Field(
        3.0, gt=0, description="Angle defect threshold (degrees)"
    )
    min_hole_diameter: float = Field(
        2.0, gt=0, description="Minimum hole diameter to detect (mm)"
    )
    fillet_detection: bool = Field(True, description="Enable fillet detection")
    chamfer_detection: bool = Field(True, description="Enable chamfer detection")
    thread_detection: bool = Field(True, description="Enable thread detection")


class ProcessingConfig(BaseModel):
    """Configuration for processing pipeline."""

    model_config = ConfigDict(frozen=True)

    parallel_enabled: bool = Field(True, description="Enable parallel processing")
    batch_size: int = Field(
        4, ge=1, description="Batch size for parallel processing"
    )
    timeout: int = Field(60, ge=1, description="Timeout for code execution (seconds)")
    max_workers: Optional[int] = Field(
        None, description="Max workers for parallel processing (None = auto)"
    )
    memory_limit_gb: Optional[float] = Field(
        None, gt=0, description="Memory limit per process (GB)"
    )


class CacheConfig(BaseModel):
    """Configuration for caching system."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(True, description="Enable caching")
    cache_dir: Path = Field(
        Path.home() / ".cache" / "recodestl", description="Cache directory"
    )
    max_size_gb: float = Field(20.0, gt=0, description="Maximum cache size (GB)")
    ttl_days: int = Field(30, ge=1, description="Cache time-to-live (days)")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )
    format: Literal["json", "console"] = Field(
        "console", description="Log format"
    )
    log_dir: Optional[Path] = Field(None, description="Directory for log files")
    log_to_file: bool = Field(False, description="Enable file logging")


class Config(BaseModel):
    """Main configuration for RecodeSTL."""

    model_config = ConfigDict(frozen=True)

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="Sampling configuration"
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    export: ExportConfig = Field(
        default_factory=ExportConfig, description="Export configuration"
    )
    feature_detection: FeatureDetectionConfig = Field(
        default_factory=FeatureDetectionConfig,
        description="Feature detection configuration",
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Processing configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    @classmethod
    def from_toml(cls, path: Path | str) -> "Config":
        """Load configuration from TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            tomli.TOMLDecodeError: If TOML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def save_toml(self, path: Path | str) -> None:
        """Save configuration to TOML file.

        Args:
            path: Path to save TOML file
        """
        import tomli_w

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Default Config instance
    """
    return Config()


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load configuration from file or return defaults.

    Args:
        path: Optional path to configuration file

    Returns:
        Config instance
    """
    if path:
        return Config.from_toml(path)
    return get_default_config()