"""Factory for creating sampling strategies."""

from typing import Any, Dict, Optional, Type

from recodestl.sampling.base import SamplingStrategy
from recodestl.sampling.uniform import UniformSampler
from recodestl.sampling.poisson import PoissonDiskSampler, BlueNoiseSampler
from recodestl.sampling.adaptive import AdaptiveSampler


class SamplingFactory:
    """Factory for creating sampling strategies."""

    _strategies: Dict[str, Type[SamplingStrategy]] = {
        "uniform": UniformSampler,
        "poisson": PoissonDiskSampler,
        "blue_noise": BlueNoiseSampler,
        "adaptive": AdaptiveSampler,
    }

    @classmethod
    def create(
        cls,
        method: str,
        num_points: int = 256,
        **kwargs: Any,
    ) -> SamplingStrategy:
        """Create a sampling strategy.

        Args:
            method: Sampling method name
            num_points: Number of points to sample
            **kwargs: Additional arguments for the strategy

        Returns:
            Sampling strategy instance

        Raises:
            ValueError: If method is unknown
        """
        if method not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown sampling method: {method}. Available: {available}"
            )
            
        strategy_class = cls._strategies[method]
        return strategy_class(num_points=num_points, **kwargs)

    @classmethod
    def register(cls, name: str, strategy_class: Type[SamplingStrategy]) -> None:
        """Register a new sampling strategy.

        Args:
            name: Name for the strategy
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def available_methods(cls) -> list[str]:
        """Get list of available sampling methods.

        Returns:
            List of method names
        """
        return list(cls._strategies.keys())