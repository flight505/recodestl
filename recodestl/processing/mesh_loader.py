"""STL mesh loading and preprocessing functionality."""

import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh
from tqdm import tqdm

from recodestl.core.exceptions import STLLoadError, STLValidationError
from recodestl.utils import CacheManager


class MeshLoader:
    """Loads and preprocesses STL files for point cloud generation."""

    # Maximum file size in bytes (1GB)
    MAX_FILE_SIZE = 1_000_000_000
    
    # Minimum file size for valid STL (header + at least one triangle)
    MIN_FILE_SIZE = 84 + 50

    def __init__(self, show_progress: bool = True, cache_manager: Optional[CacheManager] = None):
        """Initialize mesh loader.

        Args:
            show_progress: Whether to show progress bars
            cache_manager: Optional cache manager for storing loaded meshes
        """
        self.show_progress = show_progress
        self.cache_manager = cache_manager

    def load(
        self,
        file_path: Union[str, Path],
        process: bool = True,
        validate: bool = True,
    ) -> trimesh.Trimesh:
        """Load STL file and return trimesh object.

        Args:
            file_path: Path to STL file
            process: Whether to process mesh (merge vertices, etc.)
            validate: Whether to validate mesh after loading

        Returns:
            Loaded trimesh object

        Raises:
            STLLoadError: If file cannot be loaded
            STLValidationError: If validation fails
        """
        file_path = Path(file_path)
        
        # Basic file validation
        self._validate_file(file_path)
        
        # Check cache if available
        if self.cache_manager:
            cache_key = self.cache_manager.generate_key(
                file_path,
                params={"process": process, "validate": validate},
                prefix="mesh"
            )
            cached_mesh = self.cache_manager.get_mesh(cache_key)
            if cached_mesh is not None:
                return cached_mesh
        
        try:
            # Determine file format
            is_binary = self._is_binary_stl(file_path)
            
            if is_binary and self.show_progress:
                # Load with progress for binary files
                mesh = self._load_binary_with_progress(file_path)
            else:
                # Load using trimesh (handles both ASCII and binary)
                mesh = trimesh.load(
                    file_path,
                    file_type="stl",
                    process=process,
                    force="mesh",
                )
                
            # Ensure we have a Trimesh object
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                if len(mesh.geometry) == 0:
                    raise STLLoadError(file_path, "No geometry found in file")
                mesh = list(mesh.geometry.values())[0]
                
            if not isinstance(mesh, trimesh.Trimesh):
                raise STLLoadError(
                    file_path,
                    f"Expected Trimesh object, got {type(mesh).__name__}",
                )
                
            # Validate if requested
            if validate:
                self.validate_mesh(mesh, file_path)
            
            # Cache the loaded mesh
            if self.cache_manager and cache_key:
                self.cache_manager.cache_mesh(cache_key, mesh)
                
            return mesh
            
        except STLLoadError:
            raise
        except STLValidationError:
            raise
        except Exception as e:
            raise STLLoadError(file_path, str(e))

    def _validate_file(self, file_path: Path) -> None:
        """Validate file before loading.

        Args:
            file_path: Path to validate

        Raises:
            STLLoadError: If file validation fails
        """
        if not file_path.exists():
            raise STLLoadError(file_path, "File does not exist")
            
        if not file_path.is_file():
            raise STLLoadError(file_path, "Path is not a file")
            
        file_size = file_path.stat().st_size
        
        if file_size == 0:
            raise STLLoadError(file_path, "File is empty")
            
        if file_size < self.MIN_FILE_SIZE:
            raise STLLoadError(
                file_path,
                f"File too small ({file_size} bytes) to be valid STL",
            )
            
        if file_size > self.MAX_FILE_SIZE:
            raise STLLoadError(
                file_path,
                f"File too large ({file_size / 1e9:.1f}GB > 1GB limit)",
            )
            
        # Check extension
        if file_path.suffix.lower() not in [".stl"]:
            raise STLLoadError(
                file_path,
                f"Unsupported file extension: {file_path.suffix}",
            )

    def _is_binary_stl(self, file_path: Path) -> bool:
        """Check if STL file is binary format.

        Args:
            file_path: Path to STL file

        Returns:
            True if binary, False if ASCII
        """
        with open(file_path, "rb") as f:
            # Read first 5 bytes
            header = f.read(5)
            
        # ASCII STL files start with "solid"
        try:
            if header.decode("ascii").lower() == "solid":
                # Could still be binary with "solid" in header
                # Check if file size matches binary format
                file_size = file_path.stat().st_size
                with open(file_path, "rb") as f:
                    f.seek(80)  # Skip header
                    triangle_count = struct.unpack("<I", f.read(4))[0]
                    expected_size = 84 + (50 * triangle_count)
                    
                return file_size == expected_size
            else:
                return True
        except UnicodeDecodeError:
            return True

    def _load_binary_with_progress(self, file_path: Path) -> trimesh.Trimesh:
        """Load binary STL file with progress bar.

        Args:
            file_path: Path to binary STL file

        Returns:
            Loaded trimesh object
        """
        with open(file_path, "rb") as f:
            # Read header (80 bytes)
            header = f.read(80)
            
            # Read triangle count
            triangle_count = struct.unpack("<I", f.read(4))[0]
            
            # Pre-allocate arrays
            vertices = np.zeros((triangle_count * 3, 3), dtype=np.float32)
            faces = np.arange(triangle_count * 3, dtype=np.int64).reshape((-1, 3))
            
            # Read triangles with progress
            with tqdm(
                total=triangle_count,
                desc="Loading STL",
                disable=not self.show_progress,
                unit="triangles",
            ) as pbar:
                for i in range(triangle_count):
                    # Read normal (12 bytes) - we ignore this
                    f.read(12)
                    
                    # Read vertices (36 bytes = 3 vertices * 3 coords * 4 bytes)
                    for j in range(3):
                        vertices[i * 3 + j] = struct.unpack("<fff", f.read(12))
                    
                    # Read attribute byte count (2 bytes) - we ignore this
                    f.read(2)
                    
                    pbar.update(1)
                    
        # Create trimesh object
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    def validate_mesh(
        self,
        mesh: trimesh.Trimesh,
        file_path: Optional[Path] = None,
    ) -> None:
        """Validate mesh for common issues.

        Args:
            mesh: Trimesh object to validate
            file_path: Optional file path for error messages

        Raises:
            STLValidationError: If validation fails
        """
        errors = []
        path = file_path or Path("mesh")
        
        # Check if mesh has vertices and faces
        if len(mesh.vertices) == 0:
            errors.append("Mesh has no vertices")
            
        if len(mesh.faces) == 0:
            errors.append("Mesh has no faces")
            
        # Check for degenerate triangles
        areas = mesh.area_faces
        degenerate_count = np.sum(areas < 1e-10)
        if degenerate_count > 0:
            errors.append(f"Mesh has {degenerate_count} degenerate triangles")
            
        # Check bounds
        if mesh.bounds is not None:
            extents = mesh.extents
            if np.any(extents < 1e-10):
                errors.append("Mesh has zero extent in one or more dimensions")
                
        # Check for isolated vertices
        vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
        vertex_mask[mesh.faces.flatten()] = True
        isolated_count = np.sum(~vertex_mask)
        if isolated_count > 0:
            errors.append(f"Mesh has {isolated_count} isolated vertices")
            
        if errors:
            raise STLValidationError(path, errors)

    def get_mesh_info(self, mesh: trimesh.Trimesh) -> dict:
        """Get information about the mesh.

        Args:
            mesh: Trimesh object

        Returns:
            Dictionary with mesh information
        """
        info = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "edges": len(mesh.edges),
            "watertight": mesh.is_watertight,
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "bounds": {
                "min": mesh.bounds[0].tolist() if mesh.bounds is not None else None,
                "max": mesh.bounds[1].tolist() if mesh.bounds is not None else None,
            },
            "extents": mesh.extents.tolist() if mesh.extents is not None else None,
            "center_mass": mesh.center_mass.tolist(),
            "moment_inertia": mesh.moment_inertia.tolist(),
            "euler_number": mesh.euler_number,
        }
        
        # Add quality metrics
        if len(mesh.faces) > 0:
            areas = mesh.area_faces
            info["face_areas"] = {
                "min": float(np.min(areas)),
                "max": float(np.max(areas)),
                "mean": float(np.mean(areas)),
                "std": float(np.std(areas)),
            }
            
        return info


def load_stl(
    file_path: Union[str, Path],
    process: bool = True,
    validate: bool = True,
    show_progress: bool = True,
    cache_manager: Optional[CacheManager] = None,
) -> trimesh.Trimesh:
    """Convenience function to load an STL file.

    Args:
        file_path: Path to STL file
        process: Whether to process mesh
        validate: Whether to validate mesh
        show_progress: Whether to show progress bar
        cache_manager: Optional cache manager for caching loaded meshes

    Returns:
        Loaded trimesh object

    Raises:
        STLLoadError: If file cannot be loaded
        STLValidationError: If validation fails
    """
    loader = MeshLoader(show_progress=show_progress, cache_manager=cache_manager)
    return loader.load(file_path, process=process, validate=validate)