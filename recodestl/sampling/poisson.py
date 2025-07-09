"""Poisson disk sampling strategy for point clouds."""

from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from recodestl.sampling.base import (
    SamplingStrategy,
    farthest_point_sampling,
    sample_surface_points,
)


class PoissonDiskSampler(SamplingStrategy):
    """Poisson disk sampling for even point distribution."""

    def __init__(
        self,
        num_points: int = 256,
        seed: Optional[int] = None,
        k_candidates: int = 30,
        use_face_areas: bool = True,
    ):
        """Initialize Poisson disk sampler.

        Args:
            num_points: Number of points to sample
            seed: Random seed for reproducibility
            k_candidates: Number of candidates to try for each point
            use_face_areas: Whether to weight by face areas
        """
        super().__init__(num_points, seed)
        self.k_candidates = k_candidates
        self.use_face_areas = use_face_areas

    def sample(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample points using Poisson disk sampling.

        Args:
            mesh: Input mesh

        Returns:
            Array of sampled points (num_points, 3)
        """
        self._validate_mesh(mesh)
        
        # Calculate minimum distance between points
        # Based on surface area and desired number of points
        surface_area = mesh.area
        avg_area_per_point = surface_area / self.num_points
        # Approximate minimum distance (assuming roughly circular regions)
        min_distance = np.sqrt(avg_area_per_point / np.pi) * 0.8
        
        # Initialize with one random point
        initial_points, _ = sample_surface_points(
            mesh, 1, use_face_areas=self.use_face_areas
        )
        
        points = [initial_points[0]]
        active_list = [0]
        
        # Build spatial index as we go
        while len(points) < self.num_points and active_list:
            # Pick a random point from active list
            active_idx = np.random.randint(len(active_list))
            point_idx = active_list[active_idx]
            base_point = points[point_idx]
            
            # Try k candidates
            found_valid = False
            for _ in range(self.k_candidates):
                # Generate candidate on mesh surface near base point
                candidates, _ = sample_surface_points(
                    mesh, 
                    self.k_candidates * 2,  # Oversample
                    use_face_areas=self.use_face_areas
                )
                
                # Find candidates within annulus [r, 2r] from base point
                distances = np.linalg.norm(candidates - base_point, axis=1)
                mask = (distances >= min_distance) & (distances <= 2 * min_distance)
                valid_candidates = candidates[mask]
                
                if len(valid_candidates) == 0:
                    continue
                    
                # Check distance to all existing points
                if len(points) > 1:
                    tree = cKDTree(np.array(points))
                    distances, _ = tree.query(valid_candidates)
                    far_enough = distances >= min_distance
                    valid_candidates = valid_candidates[far_enough]
                
                if len(valid_candidates) > 0:
                    # Add the first valid candidate
                    new_point = valid_candidates[0]
                    points.append(new_point)
                    active_list.append(len(points) - 1)
                    found_valid = True
                    break
                    
                if len(points) >= self.num_points:
                    break
                    
            # Remove from active list if no valid candidate found
            if not found_valid:
                active_list.pop(active_idx)
                
        # Convert to array
        points = np.array(points[:self.num_points])
        
        # If we couldn't get enough points with Poisson disk,
        # fill remaining with FPS from larger sample
        if len(points) < self.num_points:
            # Sample more points and use FPS to fill
            extra_points, _ = sample_surface_points(
                mesh,
                self.num_points * 10,
                use_face_areas=self.use_face_areas
            )
            
            # Remove points too close to existing ones
            if len(points) > 0:
                tree = cKDTree(points)
                distances, _ = tree.query(extra_points)
                valid_extra = extra_points[distances >= min_distance * 0.5]
                
                if len(valid_extra) > 0:
                    # Combine and use FPS
                    all_points = np.vstack([points, valid_extra])
                    points, _ = farthest_point_sampling(all_points, self.num_points)
                    
        return points.astype(np.float32)


class BlueNoiseSampler(SamplingStrategy):
    """Blue noise sampling using void-and-cluster method."""

    def __init__(
        self,
        num_points: int = 256,
        seed: Optional[int] = None,
        iterations: int = 10,
        initial_samples: int = 10000,
    ):
        """Initialize blue noise sampler.

        Args:
            num_points: Number of points to sample
            seed: Random seed for reproducibility
            iterations: Number of Lloyd relaxation iterations
            initial_samples: Initial number of samples before filtering
        """
        super().__init__(num_points, seed)
        self.iterations = iterations
        self.initial_samples = max(initial_samples, num_points * 20)

    def sample(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample points using blue noise distribution.

        Args:
            mesh: Input mesh

        Returns:
            Array of sampled points (num_points, 3)
        """
        self._validate_mesh(mesh)
        
        # Start with many random samples
        points, face_indices = sample_surface_points(
            mesh, self.initial_samples, use_face_areas=True
        )
        
        # Use void-and-cluster method
        # Start with farthest point sampling for initial distribution
        selected_points, selected_indices = farthest_point_sampling(
            points, self.num_points
        )
        
        # Lloyd relaxation on mesh surface
        for _ in range(self.iterations):
            # Build Voronoi regions (approximate with nearest neighbor)
            tree = cKDTree(selected_points)
            _, assignments = tree.query(points)
            
            # Compute centroids of each region
            new_points = []
            for i in range(self.num_points):
                mask = assignments == i
                if np.any(mask):
                    # Get points in this Voronoi cell
                    cell_points = points[mask]
                    # Compute centroid
                    centroid = np.mean(cell_points, axis=0)
                    
                    # Project centroid back to mesh surface
                    # Find closest point on mesh
                    _, _, closest = trimesh.proximity.closest_point(
                        mesh, centroid.reshape(1, 3)
                    )
                    new_points.append(closest[0])
                else:
                    # Keep original point if no points in cell
                    new_points.append(selected_points[i])
                    
            selected_points = np.array(new_points)
            
        return selected_points.astype(np.float32)