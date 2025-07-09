"""Point cloud sampling strategies for RecodeSTL."""

from recodestl.sampling.base import (
    SamplingStrategy,
    farthest_point_sampling,
    normalize_points,
    denormalize_points,
    sample_surface_points,
)
from recodestl.sampling.factory import SamplingFactory
from recodestl.sampling.uniform import UniformSampler
from recodestl.sampling.poisson import PoissonDiskSampler, BlueNoiseSampler
from recodestl.sampling.adaptive import AdaptiveSampler

__all__ = [
    "SamplingStrategy",
    "SamplingFactory",
    "UniformSampler",
    "PoissonDiskSampler",
    "BlueNoiseSampler",
    "AdaptiveSampler",
    "farthest_point_sampling",
    "normalize_points",
    "denormalize_points",
    "sample_surface_points",
]