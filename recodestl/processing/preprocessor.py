"""Mesh preprocessing and normalization functionality."""

from typing import Optional, Tuple

import numpy as np
import trimesh


class MeshPreprocessor:
    """Preprocesses meshes for point cloud generation."""

    def __init__(
        self,
        target_scale: float = 2.0,
        center: bool = True,
        fix_normals: bool = True,
        remove_degenerate: bool = True,
    ):
        """Initialize mesh preprocessor.

        Args:
            target_scale: Target scale for normalized mesh (max extent)
            center: Whether to center mesh at origin
            fix_normals: Whether to fix face normals
            remove_degenerate: Whether to remove degenerate faces
        """
        self.target_scale = target_scale
        self.center = center
        self.fix_normals = fix_normals
        self.remove_degenerate = remove_degenerate

    def preprocess(
        self,
        mesh: trimesh.Trimesh,
        inplace: bool = False,
    ) -> Tuple[trimesh.Trimesh, dict]:
        """Preprocess mesh for point cloud generation.

        Args:
            mesh: Input mesh
            inplace: Whether to modify mesh in place

        Returns:
            Tuple of (processed_mesh, transform_info)
        """
        if not inplace:
            mesh = mesh.copy()
            
        transform_info = {
            "original_bounds": mesh.bounds.copy() if mesh.bounds is not None else None,
            "original_center": mesh.center_mass.copy(),
            "transformations": [],
        }
        
        # Remove degenerate faces
        if self.remove_degenerate:
            mesh = self._remove_degenerate_faces(mesh)
            transform_info["transformations"].append("remove_degenerate")
            
        # Fix normals
        if self.fix_normals:
            mesh.fix_normals()
            transform_info["transformations"].append("fix_normals")
            
        # Center mesh
        if self.center:
            translation = self._center_mesh(mesh)
            transform_info["translation"] = translation
            transform_info["transformations"].append("center")
            
        # Scale mesh
        scale_factor = self._normalize_scale(mesh)
        transform_info["scale_factor"] = scale_factor
        transform_info["transformations"].append("scale")
        
        # Store final bounds
        transform_info["final_bounds"] = mesh.bounds.copy() if mesh.bounds is not None else None
        
        return mesh, transform_info

    def _remove_degenerate_faces(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Remove degenerate (zero-area) faces from mesh.

        Args:
            mesh: Input mesh

        Returns:
            Mesh with degenerate faces removed
        """
        if len(mesh.faces) == 0:
            return mesh
            
        # Calculate face areas
        areas = mesh.area_faces
        
        # Find non-degenerate faces (area > threshold)
        valid_faces = areas > 1e-10
        
        if np.all(valid_faces):
            return mesh
            
        # Remove degenerate faces
        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()
        
        return mesh

    def _center_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Center mesh at origin.

        Args:
            mesh: Input mesh

        Returns:
            Translation vector applied
        """
        # Calculate center (use centroid for better stability)
        if mesh.is_watertight:
            center = mesh.center_mass
        else:
            center = mesh.centroid
            
        # Translate to origin
        mesh.apply_translation(-center)
        
        return -center

    def _normalize_scale(self, mesh: trimesh.Trimesh) -> float:
        """Normalize mesh scale to target size.

        Args:
            mesh: Input mesh

        Returns:
            Scale factor applied
        """
        if mesh.extents is None or np.all(mesh.extents == 0):
            return 1.0
            
        # Calculate scale factor
        max_extent = np.max(mesh.extents)
        scale_factor = self.target_scale / max_extent
        
        # Apply scaling
        mesh.apply_scale(scale_factor)
        
        return scale_factor

    def apply_transform(
        self,
        points: np.ndarray,
        transform_info: dict,
        inverse: bool = False,
    ) -> np.ndarray:
        """Apply or invert transformation to points.

        Args:
            points: Points to transform (N, 3)
            transform_info: Transform info from preprocess
            inverse: Whether to apply inverse transform

        Returns:
            Transformed points
        """
        points = points.copy()
        
        if inverse:
            # Apply transformations in reverse order
            if "scale_factor" in transform_info:
                points /= transform_info["scale_factor"]
                
            if "translation" in transform_info:
                points -= transform_info["translation"]
        else:
            # Apply transformations in forward order
            if "translation" in transform_info:
                points += transform_info["translation"]
                
            if "scale_factor" in transform_info:
                points *= transform_info["scale_factor"]
                
        return points

    def repair_mesh(
        self,
        mesh: trimesh.Trimesh,
        fill_holes: bool = True,
        fix_winding: bool = True,
        remove_duplicate_faces: bool = True,
    ) -> trimesh.Trimesh:
        """Attempt to repair common mesh issues.

        Args:
            mesh: Input mesh
            fill_holes: Whether to fill holes
            fix_winding: Whether to fix face winding
            remove_duplicate_faces: Whether to remove duplicate faces

        Returns:
            Repaired mesh
        """
        mesh = mesh.copy()
        
        # Remove duplicate faces
        if remove_duplicate_faces:
            mesh.merge_vertices()
            mesh.remove_duplicate_faces()
            
        # Fix face winding
        if fix_winding:
            mesh.fix_normals()
            
        # Fill holes (simple method)
        if fill_holes and not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except Exception:
                # Hole filling can fail for complex meshes
                pass
                
        return mesh

    def decimate_mesh(
        self,
        mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
        target_fraction: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """Decimate mesh to reduce complexity.

        Args:
            mesh: Input mesh
            target_faces: Target number of faces
            target_fraction: Target fraction of original faces

        Returns:
            Decimated mesh
        """
        if target_faces is None and target_fraction is None:
            return mesh
            
        current_faces = len(mesh.faces)
        
        if target_fraction is not None:
            target_faces = int(current_faces * target_fraction)
            
        if target_faces >= current_faces:
            return mesh
            
        # Use trimesh simplification
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        return simplified


def preprocess_mesh(
    mesh: trimesh.Trimesh,
    target_scale: float = 2.0,
    center: bool = True,
    fix_normals: bool = True,
    remove_degenerate: bool = True,
) -> Tuple[trimesh.Trimesh, dict]:
    """Convenience function to preprocess a mesh.

    Args:
        mesh: Input mesh
        target_scale: Target scale for normalized mesh
        center: Whether to center mesh
        fix_normals: Whether to fix normals
        remove_degenerate: Whether to remove degenerate faces

    Returns:
        Tuple of (processed_mesh, transform_info)
    """
    preprocessor = MeshPreprocessor(
        target_scale=target_scale,
        center=center,
        fix_normals=fix_normals,
        remove_degenerate=remove_degenerate,
    )
    return preprocessor.preprocess(mesh)