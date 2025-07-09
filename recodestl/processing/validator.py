"""Mesh validation and analysis functionality."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import trimesh

from recodestl.core.exceptions import STLValidationError


@dataclass
class ValidationReport:
    """Report containing mesh validation results."""

    is_valid: bool
    is_watertight: bool
    vertex_count: int
    face_count: int
    edge_count: int
    volume: Optional[float]
    surface_area: float
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    extents: np.ndarray
    errors: List[str]
    warnings: List[str]
    features: List[str]
    
    # Quality metrics
    face_area_min: float
    face_area_max: float
    face_area_mean: float
    face_area_std: float
    edge_length_min: float
    edge_length_max: float
    edge_length_mean: float
    edge_length_std: float
    
    # Feature detection
    has_holes: bool
    hole_count: int
    connected_components: int
    genus: int
    euler_characteristic: int
    
    # Recommendations
    recommended_method: str
    recommended_points: int
    curvature_percentage: float


class MeshValidator:
    """Validates and analyzes mesh properties."""

    # Thresholds for validation
    MIN_FACE_AREA = 1e-10
    MIN_EDGE_LENGTH = 1e-10
    MAX_ASPECT_RATIO = 100.0
    
    # Feature detection thresholds
    HIGH_CURVATURE_PERCENTILE = 75
    FEATURE_ANGLE_THRESHOLD = 30.0  # degrees

    def validate(self, mesh: trimesh.Trimesh) -> ValidationReport:
        """Perform comprehensive mesh validation.

        Args:
            mesh: Trimesh object to validate

        Returns:
            ValidationReport with results
        """
        errors = []
        warnings = []
        features = []
        
        # Basic validation
        if len(mesh.vertices) == 0:
            errors.append("Mesh has no vertices")
        if len(mesh.faces) == 0:
            errors.append("Mesh has no faces")
            
        # Calculate metrics
        face_areas = mesh.area_faces if len(mesh.faces) > 0 else np.array([0])
        edge_lengths = mesh.edges_unique_length if len(mesh.edges) > 0 else np.array([0])
        
        # Check for degenerate faces
        degenerate_faces = np.sum(face_areas < self.MIN_FACE_AREA)
        if degenerate_faces > 0:
            errors.append(f"{degenerate_faces} degenerate faces found")
            
        # Check for very small edges
        tiny_edges = np.sum(edge_lengths < self.MIN_EDGE_LENGTH)
        if tiny_edges > 0:
            warnings.append(f"{tiny_edges} very small edges found")
            
        # Check aspect ratios
        if len(mesh.faces) > 0:
            aspect_ratios = self._calculate_aspect_ratios(mesh)
            high_aspect = np.sum(aspect_ratios > self.MAX_ASPECT_RATIO)
            if high_aspect > 0:
                warnings.append(f"{high_aspect} faces with high aspect ratio")
                
        # Feature detection
        if mesh.is_watertight:
            features.append("watertight")
        else:
            features.append("open mesh")
            
        # Detect mechanical features
        detected_features = self._detect_features(mesh)
        features.extend(detected_features)
        
        # Calculate curvature percentage
        curvature_pct = self._calculate_curvature_percentage(mesh)
        
        # Determine recommendations
        recommended_method, recommended_points = self._get_recommendations(
            mesh, features, curvature_pct
        )
        
        # Check topology
        components = mesh.split(only_watertight=False)
        hole_count = 0
        if mesh.is_watertight:
            # For watertight meshes, we can calculate genus
            euler = mesh.euler_number
            genus = (2 - euler) // 2
        else:
            genus = 0
            # Count boundary loops as holes
            if hasattr(mesh, "facets_boundary"):
                hole_count = len(mesh.facets_boundary)
                
        return ValidationReport(
            is_valid=len(errors) == 0,
            is_watertight=mesh.is_watertight,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            edge_count=len(mesh.edges),
            volume=float(mesh.volume) if mesh.is_watertight else None,
            surface_area=float(mesh.area),
            bounds_min=mesh.bounds[0] if mesh.bounds is not None else np.zeros(3),
            bounds_max=mesh.bounds[1] if mesh.bounds is not None else np.zeros(3),
            extents=mesh.extents if mesh.extents is not None else np.zeros(3),
            errors=errors,
            warnings=warnings,
            features=features,
            face_area_min=float(np.min(face_areas)),
            face_area_max=float(np.max(face_areas)),
            face_area_mean=float(np.mean(face_areas)),
            face_area_std=float(np.std(face_areas)),
            edge_length_min=float(np.min(edge_lengths)),
            edge_length_max=float(np.max(edge_lengths)),
            edge_length_mean=float(np.mean(edge_lengths)),
            edge_length_std=float(np.std(edge_lengths)),
            has_holes=hole_count > 0 or not mesh.is_watertight,
            hole_count=hole_count,
            connected_components=len(components),
            genus=genus,
            euler_characteristic=mesh.euler_number,
            recommended_method=recommended_method,
            recommended_points=recommended_points,
            curvature_percentage=curvature_pct,
        )

    def analyze(self, mesh: trimesh.Trimesh) -> ValidationReport:
        """Alias for validate method.

        Args:
            mesh: Trimesh object to analyze

        Returns:
            ValidationReport with results
        """
        return self.validate(mesh)

    def _calculate_aspect_ratios(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate aspect ratios for all faces.

        Args:
            mesh: Trimesh object

        Returns:
            Array of aspect ratios
        """
        aspect_ratios = []
        
        for face in mesh.faces:
            # Get vertices of the triangle
            v0, v1, v2 = mesh.vertices[face]
            
            # Calculate edge lengths
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            
            # Aspect ratio is max edge / min edge
            if min(e0, e1, e2) > 0:
                ratio = max(e0, e1, e2) / min(e0, e1, e2)
            else:
                ratio = np.inf
                
            aspect_ratios.append(ratio)
            
        return np.array(aspect_ratios)

    def _detect_features(self, mesh: trimesh.Trimesh) -> List[str]:
        """Detect mechanical features in the mesh.

        Args:
            mesh: Trimesh object

        Returns:
            List of detected features
        """
        features = []
        
        # Detect sharp edges
        if len(mesh.faces) > 0:
            # Calculate face normals
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) > 0:
                # Get angles between adjacent faces
                angles = mesh.face_adjacency_angles
                sharp_edges = np.sum(angles > np.radians(self.FEATURE_ANGLE_THRESHOLD))
                if sharp_edges > 0:
                    features.append(f"{sharp_edges} sharp edges")
                    
        # Detect cylindrical features (potential holes)
        # This is a simplified detection - real implementation would be more sophisticated
        if hasattr(mesh, "principal_inertia_components"):
            inertia = mesh.principal_inertia_components
            # Check if two principal components are similar (cylindrical)
            ratios = inertia / np.max(inertia)
            if np.sum(np.abs(ratios - 1.0) < 0.1) >= 2:
                features.append("cylindrical features")
                
        # Detect planar regions
        if len(mesh.faces) > 10:
            # Group faces by similar normals
            normals = mesh.face_normals
            unique_normals = np.unique(
                np.round(normals, decimals=2), axis=0
            )
            if len(unique_normals) < len(normals) * 0.1:
                features.append("planar regions")
                
        return features

    def _calculate_curvature_percentage(self, mesh: trimesh.Trimesh) -> float:
        """Calculate percentage of high curvature regions.

        Args:
            mesh: Trimesh object

        Returns:
            Percentage of vertices in high curvature regions
        """
        try:
            # Simplified curvature estimation using vertex defects
            vertex_defects = mesh.vertex_defects
            
            # High curvature vertices have large defects
            threshold = np.percentile(
                np.abs(vertex_defects),
                self.HIGH_CURVATURE_PERCENTILE
            )
            high_curvature = np.sum(np.abs(vertex_defects) > threshold)
            
            return (high_curvature / len(mesh.vertices)) * 100
            
        except Exception:
            # If curvature calculation fails, return 0
            return 0.0

    def _get_recommendations(
        self,
        mesh: trimesh.Trimesh,
        features: List[str],
        curvature_pct: float,
    ) -> tuple[str, int]:
        """Get sampling recommendations based on mesh properties.

        Args:
            mesh: Trimesh object
            features: Detected features
            curvature_pct: Percentage of high curvature regions

        Returns:
            Tuple of (recommended_method, recommended_points)
        """
        # Determine sampling method
        if curvature_pct > 20:
            method = "adaptive"
        elif "sharp edges" in " ".join(features):
            method = "adaptive"
        elif mesh.is_watertight and len(features) < 3:
            method = "poisson"
        else:
            method = "uniform"
            
        # Determine point count
        # Base it on mesh complexity
        face_count = len(mesh.faces)
        if face_count < 1000:
            points = 256
        elif face_count < 10000:
            points = 512
        else:
            points = 1024
            
        return method, points


def validate_stl(file_path: str) -> ValidationReport:
    """Convenience function to validate an STL file.

    Args:
        file_path: Path to STL file

    Returns:
        ValidationReport with results

    Raises:
        STLLoadError: If file cannot be loaded
    """
    from recodestl.processing.mesh_loader import load_stl
    
    mesh = load_stl(file_path, validate=False)
    validator = MeshValidator()
    return validator.validate(mesh)