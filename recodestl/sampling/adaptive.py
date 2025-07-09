"""Adaptive point cloud sampling based on geometric features."""

from typing import Optional, Tuple

import numpy as np
import trimesh

from recodestl.core.exceptions import PointCloudError
from recodestl.sampling.base import (
    SamplingStrategy,
    farthest_point_sampling,
    sample_surface_points,
)


class AdaptiveSampler(SamplingStrategy):
    """Adaptive sampling based on curvature and geometric features."""

    def __init__(
        self,
        num_points: int = 256,
        seed: Optional[int] = None,
        curvature_weight: float = 0.7,
        edge_weight: float = 0.2,
        normal_weight: float = 0.1,
        feature_radius: float = 0.1,
        edge_angle_threshold: float = 30.0,
    ):
        """Initialize adaptive sampler.

        Args:
            num_points: Number of points to sample
            seed: Random seed for reproducibility
            curvature_weight: Weight for high-curvature regions (0-1)
            edge_weight: Weight for edge regions (0-1)
            normal_weight: Weight for normal variation regions (0-1)
            feature_radius: Radius for feature detection
            edge_angle_threshold: Angle threshold for edge detection (degrees)
        """
        super().__init__(num_points, seed)
        self.curvature_weight = curvature_weight
        self.edge_weight = edge_weight
        self.normal_weight = normal_weight
        self.feature_radius = feature_radius
        self.edge_angle_threshold = edge_angle_threshold
        
        # Normalize weights
        total_weight = curvature_weight + edge_weight + normal_weight
        if total_weight > 0:
            self.curvature_weight /= total_weight
            self.edge_weight /= total_weight
            self.normal_weight /= total_weight

    def sample(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample points adaptively based on geometric features.

        Args:
            mesh: Input mesh

        Returns:
            Array of sampled points (num_points, 3)
        """
        self._validate_mesh(mesh)
        
        # Calculate feature importance for each face
        importance_scores = self._calculate_importance(mesh)
        
        # Allocate points based on importance
        n_feature = int(self.num_points * 0.7)  # 70% for features
        n_uniform = self.num_points - n_feature  # 30% uniform
        
        # Sample from high-importance regions
        feature_points = self._sample_by_importance(
            mesh, importance_scores, n_feature
        )
        
        # Sample uniformly for coverage
        uniform_points, _ = sample_surface_points(
            mesh, n_uniform * 10, use_face_areas=True
        )
        
        # Use FPS on uniform points for better distribution
        if len(uniform_points) > n_uniform:
            uniform_points, _ = farthest_point_sampling(uniform_points, n_uniform)
        
        # Combine points
        all_points = np.vstack([feature_points, uniform_points[:n_uniform]])
        
        # Final FPS to ensure good distribution
        if len(all_points) > self.num_points:
            all_points, _ = farthest_point_sampling(all_points, self.num_points)
            
        return all_points.astype(np.float32)

    def _calculate_importance(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate importance score for each face.

        Args:
            mesh: Input mesh

        Returns:
            Array of importance scores per face
        """
        n_faces = len(mesh.faces)
        importance = np.zeros(n_faces)
        
        # 1. Curvature-based importance
        if self.curvature_weight > 0:
            curvature_importance = self._calculate_curvature_importance(mesh)
            importance += self.curvature_weight * curvature_importance
            
        # 2. Edge-based importance
        if self.edge_weight > 0:
            edge_importance = self._calculate_edge_importance(mesh)
            importance += self.edge_weight * edge_importance
            
        # 3. Normal variation importance
        if self.normal_weight > 0:
            normal_importance = self._calculate_normal_importance(mesh)
            importance += self.normal_weight * normal_importance
            
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
            
        return importance

    def _calculate_curvature_importance(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate importance based on curvature.

        Args:
            mesh: Input mesh

        Returns:
            Curvature importance per face
        """
        try:
            # Get vertex defects (related to Gaussian curvature)
            vertex_defects = mesh.vertex_defects
            
            # Map vertex curvature to faces
            face_curvature = np.abs(vertex_defects[mesh.faces]).mean(axis=1)
            
            # Normalize using percentiles for robustness
            p90 = np.percentile(face_curvature, 90)
            if p90 > 0:
                face_curvature = np.clip(face_curvature / p90, 0, 1)
                
            return face_curvature
            
        except Exception:
            # Fallback: use face normal variation
            return self._calculate_normal_importance(mesh)

    def _calculate_edge_importance(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate importance based on edges.

        Args:
            mesh: Input mesh

        Returns:
            Edge importance per face
        """
        edge_importance = np.zeros(len(mesh.faces))
        
        # Get face adjacency
        adjacency = mesh.face_adjacency
        
        if len(adjacency) > 0:
            # Calculate angles between adjacent faces
            angles = mesh.face_adjacency_angles
            
            # Mark faces with sharp edges
            sharp_edges = angles > np.radians(self.edge_angle_threshold)
            
            # Create face-to-edge mapping
            for i, (f1, f2) in enumerate(adjacency):
                if sharp_edges[i]:
                    edge_importance[f1] = 1.0
                    edge_importance[f2] = 1.0
                    
        return edge_importance

    def _calculate_normal_importance(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate importance based on normal variation.

        Args:
            mesh: Input mesh

        Returns:
            Normal variation importance per face
        """
        face_normals = mesh.face_normals
        normal_importance = np.zeros(len(mesh.faces))
        
        # For each face, check normal variation with neighbors
        adjacency = mesh.face_adjacency
        
        if len(adjacency) > 0:
            # Build face neighbor list
            neighbors = [[] for _ in range(len(mesh.faces))]
            for f1, f2 in adjacency:
                neighbors[f1].append(f2)
                neighbors[f2].append(f1)
                
            # Calculate normal variation
            for i, face_neighbors in enumerate(neighbors):
                if face_neighbors:
                    # Get normal differences
                    current_normal = face_normals[i]
                    neighbor_normals = face_normals[face_neighbors]
                    
                    # Calculate average angle difference
                    dots = np.clip(np.dot(neighbor_normals, current_normal), -1, 1)
                    angles = np.arccos(dots)
                    normal_importance[i] = np.mean(angles) / np.pi
                    
        return normal_importance

    def _sample_by_importance(
        self,
        mesh: trimesh.Trimesh,
        importance: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """Sample points weighted by importance.

        Args:
            mesh: Input mesh
            importance: Importance score per face
            num_samples: Number of points to sample

        Returns:
            Sampled points
        """
        # Adjust face areas by importance
        weighted_areas = mesh.area_faces * (importance + 0.1)  # Add small constant
        
        # Normalize to probabilities
        probabilities = weighted_areas / weighted_areas.sum()
        
        # Sample faces based on weighted probabilities
        sampled_faces = np.random.choice(
            len(mesh.faces),
            size=num_samples,
            p=probabilities,
            replace=True
        )
        
        # Sample points on selected faces
        points = []
        for face_idx in sampled_faces:
            # Get face vertices
            vertices = mesh.vertices[mesh.faces[face_idx]]
            
            # Random barycentric coordinates
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            
            # Compute point
            point = r1 * vertices[0] + r2 * vertices[1] + r3 * vertices[2]
            points.append(point)
            
        return np.array(points)