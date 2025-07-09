"""Base classes and utilities for point cloud sampling."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh


class SamplingStrategy(ABC):
    """Abstract base class for point cloud sampling strategies."""

    def __init__(self, num_points: int = 256, seed: Optional[int] = None):
        """Initialize sampling strategy.

        Args:
            num_points: Number of points to sample
            seed: Random seed for reproducibility
        """
        self.num_points = num_points
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def sample(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample points from mesh.

        Args:
            mesh: Input mesh

        Returns:
            Array of sampled points (num_points, 3)
        """
        pass

    def _validate_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Validate mesh before sampling.

        Args:
            mesh: Mesh to validate

        Raises:
            ValueError: If mesh is invalid
        """
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
        if mesh.area == 0:
            raise ValueError("Mesh has zero surface area")


def farthest_point_sampling(
    points: np.ndarray,
    num_samples: int,
    start_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform farthest point sampling on a point cloud.

    Args:
        points: Input points (N, 3)
        num_samples: Number of points to sample
        start_idx: Optional starting point index

    Returns:
        Tuple of (sampled_points, indices)
    """
    if num_samples > len(points):
        raise ValueError(
            f"Cannot sample {num_samples} points from {len(points)} points"
        )
        
    if num_samples == len(points):
        return points, np.arange(len(points))
        
    # Initialize
    n_points = len(points)
    selected_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n_points, np.inf)
    
    # Select first point
    if start_idx is None:
        start_idx = np.random.randint(n_points)
    selected_indices[0] = start_idx
    
    # Update distances
    current_point = points[start_idx]
    new_distances = np.linalg.norm(points - current_point, axis=1)
    distances = np.minimum(distances, new_distances)
    
    # Select remaining points
    for i in range(1, num_samples):
        # Select point with maximum distance to selected set
        next_idx = np.argmax(distances)
        selected_indices[i] = next_idx
        
        # Update distances
        current_point = points[next_idx]
        new_distances = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
    return points[selected_indices], selected_indices


def farthest_point_sampling_torch(
    points: torch.Tensor,
    num_samples: int,
    start_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform farthest point sampling using PyTorch (GPU-accelerated).

    Args:
        points: Input points tensor (B, N, 3) or (N, 3)
        num_samples: Number of points to sample
        start_idx: Optional starting point index

    Returns:
        Tuple of (sampled_points, indices)
    """
    # Handle both batched and unbatched inputs
    if points.dim() == 2:
        points = points.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
        
    B, N, C = points.shape
    
    if num_samples > N:
        raise ValueError(f"Cannot sample {num_samples} points from {N} points")
        
    device = points.device
    
    # Initialize
    indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)
    
    # Select first point
    if start_idx is None:
        indices[:, 0] = torch.randint(0, N, (B,), device=device)
    else:
        indices[:, 0] = start_idx
        
    # Gather first points
    batch_indices = torch.arange(B, device=device)
    current_points = points[batch_indices, indices[:, 0], :]
    
    # Update distances
    new_distances = torch.norm(points - current_points.unsqueeze(1), dim=2)
    distances = torch.minimum(distances, new_distances)
    
    # Select remaining points
    for i in range(1, num_samples):
        # Select points with maximum distance
        indices[:, i] = torch.argmax(distances, dim=1)
        
        # Gather selected points
        current_points = points[batch_indices, indices[:, i], :]
        
        # Update distances
        new_distances = torch.norm(points - current_points.unsqueeze(1), dim=2)
        distances = torch.minimum(distances, new_distances)
        
    # Gather sampled points
    sampled_points = torch.gather(
        points,
        1,
        indices.unsqueeze(-1).expand(-1, -1, C)
    )
    
    if squeeze_output:
        return sampled_points.squeeze(0), indices.squeeze(0)
    else:
        return sampled_points, indices


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Normalize points to unit cube centered at origin.

    Args:
        points: Input points (N, 3)

    Returns:
        Tuple of (normalized_points, normalization_info)
    """
    # Calculate center and scale
    center = np.mean(points, axis=0)
    centered = points - center
    
    # Calculate scale (max extent)
    scale = np.max(np.abs(centered))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
        scale = 1.0
        
    info = {
        "center": center,
        "scale": scale,
    }
    
    return normalized, info


def denormalize_points(
    points: np.ndarray,
    normalization_info: dict,
) -> np.ndarray:
    """Denormalize points using normalization info.

    Args:
        points: Normalized points (N, 3)
        normalization_info: Info from normalize_points

    Returns:
        Denormalized points
    """
    return points * normalization_info["scale"] + normalization_info["center"]


def sample_surface_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    use_face_areas: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on mesh surface.

    Args:
        mesh: Input mesh
        num_points: Number of points to sample
        use_face_areas: Whether to weight by face areas

    Returns:
        Tuple of (points, face_indices)
    """
    if use_face_areas:
        # Sample weighted by face areas
        points, face_indices = trimesh.sample.sample_surface(
            mesh,
            num_points,
            face_weight=mesh.area_faces,
        )
    else:
        # Sample uniformly across faces
        points, face_indices = trimesh.sample.sample_surface(
            mesh,
            num_points,
            face_weight=None,
        )
        
    return points, face_indices