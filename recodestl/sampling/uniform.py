"""Uniform point cloud sampling strategy."""

from typing import Optional

import numpy as np
import trimesh

from recodestl.sampling.base import (
    SamplingStrategy,
    farthest_point_sampling,
    sample_surface_points,
)


class UniformSampler(SamplingStrategy):
    """Uniform sampling strategy for point clouds."""

    def __init__(
        self,
        num_points: int = 256,
        seed: Optional[int] = None,
        use_farthest_point: bool = True,
        oversample_ratio: float = 10.0,
    ):
        """Initialize uniform sampler.

        Args:
            num_points: Number of points to sample
            seed: Random seed for reproducibility
            use_farthest_point: Whether to use farthest point sampling for final selection
            oversample_ratio: Ratio for initial oversampling before FPS
        """
        super().__init__(num_points, seed)
        self.use_farthest_point = use_farthest_point
        self.oversample_ratio = oversample_ratio

    def sample(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample points uniformly from mesh surface.

        Args:
            mesh: Input mesh

        Returns:
            Array of sampled points (num_points, 3)
        """
        self._validate_mesh(mesh)
        
        if self.use_farthest_point:
            # Oversample then use FPS for better distribution
            initial_samples = int(self.num_points * self.oversample_ratio)
            initial_samples = min(initial_samples, len(mesh.vertices) * 10)
            
            # Sample initial points
            points, _ = sample_surface_points(mesh, initial_samples, use_face_areas=True)
            
            # Use farthest point sampling for final selection
            if len(points) > self.num_points:
                points, _ = farthest_point_sampling(points, self.num_points)
        else:
            # Direct sampling
            points, _ = sample_surface_points(mesh, self.num_points, use_face_areas=True)
            
        return points.astype(np.float32)