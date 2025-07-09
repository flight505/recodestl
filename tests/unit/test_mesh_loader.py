"""Unit tests for mesh loading functionality."""

import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from recodestl.core.exceptions import STLLoadError, STLValidationError
from recodestl.processing import MeshLoader, load_stl


class TestMeshLoader:
    """Test mesh loading functionality."""

    def test_load_simple_mesh(self, sample_stl_path: Path):
        """Test loading a simple STL file."""
        loader = MeshLoader(show_progress=False)
        mesh = loader.load(sample_stl_path)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_load_with_cache(self, sample_stl_path: Path, cache_manager):
        """Test loading with cache enabled."""
        loader = MeshLoader(show_progress=False, cache_manager=cache_manager)
        
        # First load - cache miss
        mesh1 = loader.load(sample_stl_path)
        
        # Second load - cache hit  
        mesh2 = loader.load(sample_stl_path)
        
        # Should return the same mesh data
        assert len(mesh1.vertices) == len(mesh2.vertices)
        assert len(mesh1.faces) == len(mesh2.faces)
        
        # Since trimesh objects may be different instances,
        # we check that cache was at least accessed
        stats = cache_manager.get_stats()
        assert stats["entries"] > 0  # Should have cached the mesh

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = MeshLoader(show_progress=False)
        
        with pytest.raises(STLLoadError) as exc_info:
            loader.load(Path("nonexistent.stl"))
        
        assert "does not exist" in str(exc_info.value)

    def test_load_empty_file(self, temp_dir: Path):
        """Test loading an empty file."""
        empty_file = temp_dir / "empty.stl"
        empty_file.touch()
        
        loader = MeshLoader(show_progress=False)
        
        with pytest.raises(STLLoadError) as exc_info:
            loader.load(empty_file)
        
        assert "empty" in str(exc_info.value).lower()

    def test_load_corrupt_file(self, temp_dir: Path):
        """Test loading a corrupt STL file."""
        corrupt_file = temp_dir / "corrupt.stl"
        corrupt_file.write_bytes(b"This is not a valid STL file content")
        
        loader = MeshLoader(show_progress=False)
        
        with pytest.raises(STLLoadError):
            loader.load(corrupt_file)

    def test_load_binary_stl(self, temp_dir: Path, simple_box_mesh: trimesh.Trimesh):
        """Test loading a binary STL file."""
        binary_stl = temp_dir / "binary.stl"
        simple_box_mesh.export(binary_stl)  # Trimesh will auto-detect binary format
        
        loader = MeshLoader(show_progress=True)
        mesh = loader.load(binary_stl)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == len(simple_box_mesh.vertices)

    def test_load_ascii_stl(self, temp_dir: Path, simple_box_mesh: trimesh.Trimesh):
        """Test loading an ASCII STL file."""
        ascii_stl = temp_dir / "ascii.stl"
        with open(ascii_stl, 'w') as f:
            simple_box_mesh.export(f, file_type='stl_ascii')
        
        loader = MeshLoader(show_progress=False)
        mesh = loader.load(ascii_stl)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == len(simple_box_mesh.vertices)

    def test_validate_mesh_valid(self, simple_box_mesh: trimesh.Trimesh):
        """Test validating a valid mesh."""
        loader = MeshLoader(show_progress=False)
        
        # Should not raise
        loader.validate_mesh(simple_box_mesh)

    def test_validate_mesh_no_vertices(self):
        """Test validating a mesh with no vertices."""
        empty_mesh = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        loader = MeshLoader(show_progress=False)
        
        with pytest.raises(STLValidationError) as exc_info:
            loader.validate_mesh(empty_mesh)
        
        assert "no vertices" in str(exc_info.value).lower()

    def test_validate_mesh_degenerate_triangles(self, temp_dir: Path):
        """Test validating a mesh with degenerate triangles."""
        # Create mesh with degenerate triangle (all vertices at same point)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],  # Duplicate of first vertex
        ])
        faces = np.array([
            [0, 1, 2],  # Valid triangle
            [0, 0, 3],  # Degenerate triangle
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        loader = MeshLoader(show_progress=False)
        
        with pytest.raises(STLValidationError) as exc_info:
            loader.validate_mesh(mesh)
        
        assert "degenerate" in str(exc_info.value).lower()

    def test_get_mesh_info(self, simple_box_mesh: trimesh.Trimesh):
        """Test getting mesh information."""
        loader = MeshLoader(show_progress=False)
        info = loader.get_mesh_info(simple_box_mesh)
        
        assert info["vertices"] == len(simple_box_mesh.vertices)
        assert info["faces"] == len(simple_box_mesh.faces)
        assert info["watertight"] == simple_box_mesh.is_watertight
        assert "bounds" in info
        assert "extents" in info
        assert "center_mass" in info

    def test_file_size_limits(self, temp_dir: Path):
        """Test file size validation."""
        loader = MeshLoader(show_progress=False)
        
        # Test file too small
        tiny_file = temp_dir / "tiny.stl"
        tiny_file.write_bytes(b"ABC")  # Too small to be valid STL
        
        with pytest.raises(STLLoadError) as exc_info:
            loader.load(tiny_file)
        
        assert "too small" in str(exc_info.value).lower()

    def test_binary_stl_detection(self, temp_dir: Path):
        """Test binary STL format detection."""
        loader = MeshLoader(show_progress=False)
        
        # Create a file that starts with "solid" but is actually binary
        tricky_file = temp_dir / "tricky.stl"
        
        # Binary STL structure: 80 byte header + 4 byte count + triangles
        header = b"solid fake" + b" " * 70  # 80 bytes
        count = struct.pack("<I", 1)  # 1 triangle
        # Each triangle: 12 bytes normal + 36 bytes vertices + 2 bytes attribute
        triangle = b"\x00" * 50
        
        tricky_file.write_bytes(header + count + triangle)
        
        assert loader._is_binary_stl(tricky_file)

    @pytest.mark.parametrize("process", [True, False])
    @pytest.mark.parametrize("validate", [True, False])
    def test_load_options(self, sample_stl_path: Path, process: bool, validate: bool):
        """Test different loading options."""
        loader = MeshLoader(show_progress=False)
        
        mesh = loader.load(sample_stl_path, process=process, validate=validate)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_convenience_function(self, sample_stl_path: Path):
        """Test the convenience load_stl function."""
        mesh = load_stl(sample_stl_path, show_progress=False)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestMeshLoaderIntegration:
    """Integration tests with real STL files."""

    @pytest.mark.integration
    def test_load_sonos_fixture(self, sonos_stl_path: Path):
        """Test loading the Sonos wall mount fixture."""
        loader = MeshLoader(show_progress=True)
        mesh = loader.load(sonos_stl_path)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert len(mesh.vertices) == 11774
        assert len(mesh.faces) == 23552
        
        # Check detected features
        info = loader.get_mesh_info(mesh)
        assert info["watertight"] is True
        assert info["volume"] > 0
        assert info["surface_area"] > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_file_handling(self, temp_dir: Path):
        """Test handling of large STL files."""
        # Create a large mesh (sphere with many faces)
        large_mesh = trimesh.creation.icosphere(subdivisions=5)
        large_stl = temp_dir / "large.stl"
        large_mesh.export(large_stl)
        
        loader = MeshLoader(show_progress=True)
        mesh = loader.load(large_stl)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.faces) > 10000  # Should have many faces