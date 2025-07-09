"""Shared test fixtures and configuration."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import trimesh

from recodestl.core import Config
from recodestl.utils import CacheManager


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        cache={"enabled": False},  # Disable cache for tests by default
        model={"device": "cpu"},   # Use CPU for tests
        sampling={"num_points": 128},  # Smaller for speed
    )


@pytest.fixture
def cache_manager(temp_dir: Path) -> CacheManager:
    """Create a test cache manager."""
    from recodestl.core.config import CacheConfig
    
    cache_config = CacheConfig(
        enabled=True,
        cache_dir=temp_dir / "cache",
        max_size_gb=0.1,  # Small size for tests
        ttl_days=1,
    )
    return CacheManager(cache_config)


@pytest.fixture
def simple_box_mesh() -> trimesh.Trimesh:
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[1, 1, 1])


@pytest.fixture
def simple_cylinder_mesh() -> trimesh.Trimesh:
    """Create a simple cylinder mesh for testing."""
    return trimesh.creation.cylinder(radius=0.5, height=2.0, sections=32)


@pytest.fixture
def complex_mesh() -> trimesh.Trimesh:
    """Create a more complex mesh with features."""
    # Combine multiple primitives
    box = trimesh.creation.box(extents=[2, 1, 0.5])
    cylinder1 = trimesh.creation.cylinder(radius=0.3, height=1.5)
    cylinder1.apply_translation([0.5, 0, 0])
    
    cylinder2 = trimesh.creation.cylinder(radius=0.3, height=1.5)
    cylinder2.apply_translation([-0.5, 0, 0])
    
    # Combine meshes
    mesh = trimesh.util.concatenate([box, cylinder1, cylinder2])
    return mesh


@pytest.fixture
def sample_stl_path(temp_dir: Path, simple_box_mesh: trimesh.Trimesh) -> Path:
    """Create a sample STL file."""
    stl_path = temp_dir / "test_box.stl"
    simple_box_mesh.export(stl_path)
    return stl_path


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """Create a sample point cloud."""
    # Create 256 random points in a unit cube
    points = np.random.rand(256, 3).astype(np.float32)
    points = points * 2 - 1  # Center around origin
    return points


@pytest.fixture
def sonos_stl_path() -> Path:
    """Path to the Sonos wall mount test fixture."""
    fixture_path = Path("tests/fixtures/Sonos_play_1_wall.stl")
    if fixture_path.exists():
        return fixture_path
    else:
        pytest.skip("Sonos fixture not found")


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Tests that require GPU (CUDA or MPS)"
    )
    config.addinivalue_line(
        "markers", "requires_model: Tests that require model weights"
    )