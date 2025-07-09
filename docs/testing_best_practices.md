# Best Practices for Unit Testing 3D Geometry Processing Code

This guide covers best practices for testing 3D geometry processing code using pytest, with a focus on Trimesh operations, NumPy array comparisons, and handling file-based test fixtures.

## Table of Contents
1. [Testing Strategies for Mesh Loading and Validation](#1-testing-strategies-for-mesh-loading-and-validation)
2. [Testing Point Cloud Sampling Algorithms](#2-testing-point-cloud-sampling-algorithms)
3. [Creating and Using Test Fixtures for 3D Data](#3-creating-and-using-test-fixtures-for-3d-data)
4. [Mocking Strategies for Large Models and File I/O](#4-mocking-strategies-for-large-models-and-file-io)
5. [Testing Numerical Accuracy in 3D Computations](#5-testing-numerical-accuracy-in-3d-computations)
6. [Property-Based Testing for Geometry Algorithms](#6-property-based-testing-for-geometry-algorithms)
7. [Testing Visualization Code Without Displaying](#7-testing-visualization-code-without-displaying)

## 1. Testing Strategies for Mesh Loading and Validation

### Structure Your Tests
Following pytest best practices, create a test file for each module:
```
tests/
├── unit/
│   ├── test_mesh_loader.py
│   ├── test_validator.py
│   └── test_preprocessor.py
└── fixtures/
    ├── valid_mesh.stl
    ├── invalid_mesh.stl
    └── edge_cases/
```

### Example Test for Mesh Loader
```python
import pytest
import numpy as np
import trimesh
from pathlib import Path
from recodestl.processing.mesh_loader import MeshLoader
from recodestl.core.exceptions import STLLoadError, STLValidationError


class TestMeshLoader:
    """Test suite for MeshLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a MeshLoader instance."""
        return MeshLoader(show_progress=False)
    
    @pytest.fixture
    def valid_mesh_path(self, tmp_path):
        """Create a valid test mesh file."""
        # Create a simple triangle mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save to temporary file
        file_path = tmp_path / "test_mesh.stl"
        mesh.export(file_path)
        return file_path
    
    def test_load_valid_mesh(self, loader, valid_mesh_path):
        """Test loading a valid mesh file."""
        mesh = loader.load(valid_mesh_path)
        
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1
    
    def test_load_nonexistent_file(self, loader):
        """Test loading a non-existent file raises STLLoadError."""
        with pytest.raises(STLLoadError, match="File does not exist"):
            loader.load("nonexistent.stl")
    
    def test_validate_empty_mesh(self, loader, tmp_path):
        """Test validation of empty mesh raises STLValidationError."""
        # Create empty mesh
        mesh = trimesh.Trimesh()
        file_path = tmp_path / "empty.stl"
        mesh.export(file_path)
        
        with pytest.raises(STLValidationError, match="no vertices"):
            loader.load(file_path, validate=True)
    
    @pytest.mark.parametrize("file_size,expected_error", [
        (0, "File is empty"),
        (50, "File too small"),
        (2_000_000_000, "File too large"),
    ])
    def test_file_size_validation(self, loader, tmp_path, file_size, expected_error):
        """Test file size validation."""
        file_path = tmp_path / "test.stl"
        
        # Create file with specific size
        with open(file_path, 'wb') as f:
            f.write(b'\0' * file_size)
        
        with pytest.raises(STLLoadError, match=expected_error):
            loader.load(file_path)
```

### Testing Edge Cases
```python
class TestMeshEdgeCases:
    """Test edge cases in mesh processing."""
    
    @pytest.fixture
    def degenerate_mesh(self):
        """Create a mesh with degenerate triangles."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0, 0],  # Collinear point
        ])
        faces = np.array([[0, 1, 2]])
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def test_degenerate_triangle_detection(self, degenerate_mesh):
        """Test detection of degenerate triangles."""
        areas = degenerate_mesh.area_faces
        assert np.any(areas < 1e-10)
```

## 2. Testing Point Cloud Sampling Algorithms

### Test Sampling Strategies
```python
import pytest
import numpy as np
import trimesh
from recodestl.sampling.base import SamplingStrategy, farthest_point_sampling
from recodestl.sampling.uniform import UniformSampling


class TestSamplingStrategies:
    """Test suite for sampling strategies."""
    
    @pytest.fixture
    def sphere_mesh(self):
        """Create a unit sphere mesh for testing."""
        return trimesh.creation.icosphere(subdivisions=3)
    
    @pytest.fixture
    def cube_mesh(self):
        """Create a unit cube mesh for testing."""
        return trimesh.creation.box()
    
    def test_uniform_sampling_point_count(self, sphere_mesh):
        """Test that uniform sampling returns correct number of points."""
        sampler = UniformSampling(num_points=256)
        points = sampler.sample(sphere_mesh)
        
        assert points.shape == (256, 3)
        assert points.dtype == np.float64
    
    def test_sampling_reproducibility(self, sphere_mesh):
        """Test that sampling with same seed produces identical results."""
        sampler1 = UniformSampling(num_points=100, seed=42)
        sampler2 = UniformSampling(num_points=100, seed=42)
        
        points1 = sampler1.sample(sphere_mesh)
        points2 = sampler2.sample(sphere_mesh)
        
        np.testing.assert_array_equal(points1, points2)
    
    def test_farthest_point_sampling(self):
        """Test farthest point sampling algorithm."""
        # Create random point cloud
        np.random.seed(42)
        points = np.random.randn(1000, 3)
        
        sampled, indices = farthest_point_sampling(points, 50)
        
        assert sampled.shape == (50, 3)
        assert len(indices) == 50
        assert np.all(indices < 1000)
        assert len(np.unique(indices)) == 50  # All indices unique
    
    @pytest.mark.parametrize("num_points,num_samples", [
        (100, 101),  # More samples than points
        (0, 10),     # Empty point set
    ])
    def test_farthest_point_sampling_errors(self, num_points, num_samples):
        """Test error handling in farthest point sampling."""
        points = np.random.randn(num_points, 3) if num_points > 0 else np.array([])
        
        with pytest.raises(ValueError):
            farthest_point_sampling(points, num_samples)
```

### Testing Sampling Distribution Properties
```python
class TestSamplingDistribution:
    """Test statistical properties of sampling algorithms."""
    
    def test_uniform_coverage(self, sphere_mesh):
        """Test that sampling provides uniform coverage."""
        sampler = UniformSampling(num_points=1000)
        points = sampler.sample(sphere_mesh)
        
        # Normalize points to unit sphere
        points_normalized = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Divide sphere into octants and check distribution
        octant_counts = np.zeros(8)
        for p in points_normalized:
            octant = (p[0] > 0) * 4 + (p[1] > 0) * 2 + (p[2] > 0)
            octant_counts[int(octant)] += 1
        
        # Check that distribution is roughly uniform (within 20%)
        expected_count = len(points) / 8
        assert np.all(np.abs(octant_counts - expected_count) < expected_count * 0.2)
```

## 3. Creating and Using Test Fixtures for 3D Data

### Fixture Organization
```python
# conftest.py - Shared fixtures across tests
import pytest
import numpy as np
import trimesh
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def standard_meshes(test_data_dir):
    """Load standard test meshes once per session."""
    return {
        "sphere": trimesh.creation.icosphere(),
        "cube": trimesh.creation.box(),
        "cylinder": trimesh.creation.cylinder(),
        "torus": trimesh.creation.torus(),
    }


@pytest.fixture
def mesh_factory():
    """Factory for creating custom test meshes."""
    def _create_mesh(vertices, faces, **kwargs):
        return trimesh.Trimesh(vertices=vertices, faces=faces, **kwargs)
    return _create_mesh


@pytest.fixture
def stl_file_factory(tmp_path):
    """Factory for creating temporary STL files."""
    created_files = []
    
    def _create_stl(mesh, filename="test.stl"):
        filepath = tmp_path / filename
        mesh.export(filepath)
        created_files.append(filepath)
        return filepath
    
    yield _create_stl
    
    # Cleanup
    for filepath in created_files:
        if filepath.exists():
            filepath.unlink()
```

### Parameterized Fixtures
```python
@pytest.fixture(params=[
    # (vertices, faces, description)
    (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 2]]),
        "simple_triangle"
    ),
    (
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        "square"
    ),
])
def parametric_mesh(request, mesh_factory):
    """Parametric fixture providing different test meshes."""
    vertices, faces, description = request.param
    mesh = mesh_factory(vertices, faces)
    mesh.metadata['description'] = description
    return mesh


def test_mesh_properties(parametric_mesh):
    """Test basic properties across different mesh types."""
    assert parametric_mesh.is_valid
    assert len(parametric_mesh.vertices) > 0
    assert len(parametric_mesh.faces) > 0
```

### Large Mesh Fixtures
```python
@pytest.fixture(scope="module")
def large_mesh():
    """Create a large mesh for performance testing."""
    # Use caching to avoid regenerating
    cache_path = Path("tests/cache/large_mesh.stl")
    
    if cache_path.exists():
        return trimesh.load(cache_path)
    
    # Generate large mesh
    mesh = trimesh.creation.icosphere(subdivisions=6)
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(cache_path)
    
    return mesh


@pytest.mark.slow
def test_large_mesh_processing(large_mesh):
    """Test processing of large meshes."""
    assert len(large_mesh.vertices) > 10000
    # ... performance tests ...
```

## 4. Mocking Strategies for Large Models and File I/O

### Mocking File I/O Operations
```python
import pytest
from unittest.mock import Mock, patch, mock_open
import trimesh


class TestMeshLoaderWithMocks:
    """Test mesh loading with mocked file operations."""
    
    @patch('trimesh.load')
    def test_load_with_mocked_trimesh(self, mock_load):
        """Test loading with mocked trimesh.load."""
        # Setup mock
        mock_mesh = Mock(spec=trimesh.Trimesh)
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.faces = np.array([[0, 1, 2]])
        mock_mesh.is_watertight = True
        mock_load.return_value = mock_mesh
        
        # Test
        from recodestl.processing.mesh_loader import load_stl
        mesh = load_stl("dummy.stl", validate=False)
        
        # Verify
        mock_load.assert_called_once_with(
            Path("dummy.stl"),
            file_type="stl",
            process=True,
            force="mesh"
        )
        assert len(mesh.vertices) == 3
    
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.exists')
    def test_file_validation_mocking(self, mock_exists, mock_stat):
        """Test file validation with mocked filesystem."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=1000000)  # 1MB file
        
        from recodestl.processing.mesh_loader import MeshLoader
        loader = MeshLoader()
        
        # Should not raise
        loader._validate_file(Path("dummy.stl"))
    
    def test_binary_stl_loading_mock(self, mocker):
        """Test binary STL loading with pytest-mock."""
        # Mock file reading
        mock_data = b'\x00' * 80  # Header
        mock_data += struct.pack('<I', 1)  # One triangle
        mock_data += b'\x00' * 50  # Triangle data
        
        mocker.patch('builtins.open', mock_open(read_data=mock_data))
        mocker.patch('pathlib.Path.stat').return_value = Mock(st_size=134)
        
        loader = MeshLoader()
        assert loader._is_binary_stl(Path("dummy.stl"))
```

### Mocking Large Model Processing
```python
@pytest.fixture
def mock_large_mesh():
    """Mock a large mesh without actually creating it."""
    mesh = Mock(spec=trimesh.Trimesh)
    mesh.vertices = Mock(shape=(1000000, 3))
    mesh.faces = Mock(shape=(2000000, 3))
    mesh.bounds = np.array([[0, 0, 0], [100, 100, 100]])
    mesh.extents = np.array([100, 100, 100])
    mesh.area = 60000.0
    mesh.volume = 1000000.0
    mesh.is_watertight = True
    
    # Mock expensive operations
    mesh.area_faces = np.ones(2000000) * 0.03
    mesh.face_normals = np.ones((2000000, 3)) / np.sqrt(3)
    
    return mesh


def test_large_mesh_processing(mock_large_mesh):
    """Test processing logic without actual large mesh."""
    # Test your processing functions
    info = get_mesh_info(mock_large_mesh)
    assert info['vertices'] == 1000000
    assert info['faces'] == 2000000
```

### Mocking External Dependencies
```python
class TestCADQueryExecution:
    """Test CADQuery execution with mocks."""
    
    @patch('recodestl.execution.cadquery_exec.cq')
    def test_cadquery_execution(self, mock_cq):
        """Test CADQuery execution with mocked module."""
        # Setup mock
        mock_workplane = Mock()
        mock_cq.Workplane.return_value = mock_workplane
        mock_workplane.box.return_value = mock_workplane
        mock_workplane.val.return_value = Mock(
            exportStl=Mock(return_value=b'mock_stl_data')
        )
        
        # Test execution
        from recodestl.execution.cadquery_exec import execute_cad_script
        result = execute_cad_script("Workplane().box(1,1,1)")
        
        # Verify
        assert result == b'mock_stl_data'
        mock_cq.Workplane.assert_called_once()
```

## 5. Testing Numerical Accuracy in 3D Computations

### Using NumPy Testing Functions
```python
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal


class TestNumericalAccuracy:
    """Test numerical accuracy in 3D computations."""
    
    def test_rotation_matrix_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        # Generate random rotation
        angle = np.pi / 4
        axis = np.array([1, 1, 1])
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Test orthogonality: R @ R.T should be identity
        assert_allclose(R @ R.T, np.eye(3), rtol=1e-10, atol=1e-12)
        
        # Test determinant is 1
        assert_allclose(np.linalg.det(R), 1.0, rtol=1e-10)
    
    def test_mesh_volume_calculation(self, sphere_mesh):
        """Test volume calculation accuracy for known shapes."""
        # Unit sphere volume should be 4/3 * pi
        expected_volume = 4/3 * np.pi
        
        # Allow 1% error due to mesh discretization
        assert_allclose(
            sphere_mesh.volume, 
            expected_volume, 
            rtol=0.01,
            err_msg="Sphere volume calculation inaccurate"
        )
    
    def test_normal_calculation_accuracy(self):
        """Test accuracy of normal vector calculations."""
        # Create a triangle in XY plane
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        # Calculate normal using cross product
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Expected normal should be [0, 0, 1]
        assert_allclose(normal, [0, 0, 1], atol=1e-10)
    
    @pytest.mark.parametrize("tolerance,should_pass", [
        (1e-10, True),   # High precision
        (1e-15, False),  # Too high precision for float64
    ])
    def test_precision_limits(self, tolerance, should_pass):
        """Test understanding of floating point precision limits."""
        a = 1.0
        b = 1.0 + 1e-12
        
        if should_pass:
            assert_allclose(a, b, atol=tolerance)
        else:
            with pytest.raises(AssertionError):
                assert_allclose(a, b, atol=tolerance)
```

### Testing Transformations
```python
class TestTransformations:
    """Test 3D transformations with numerical accuracy."""
    
    def test_transformation_composition(self):
        """Test that transformation composition is accurate."""
        # Create test points
        points = np.random.randn(100, 3)
        
        # Define transformations
        T1 = np.eye(4)
        T1[:3, 3] = [1, 2, 3]  # Translation
        
        R = trimesh.transformations.rotation_matrix(np.pi/3, [1, 1, 1])
        T2 = np.eye(4)
        T2[:3, :3] = R[:3, :3]  # Rotation
        
        # Apply transformations separately
        points_h = np.c_[points, np.ones(len(points))]
        result1 = (T2 @ (T1 @ points_h.T)).T[:, :3]
        
        # Apply composed transformation
        T_composed = T2 @ T1
        result2 = (T_composed @ points_h.T).T[:, :3]
        
        # Results should be identical
        assert_allclose(result1, result2, rtol=1e-14, atol=1e-14)
    
    def test_inverse_transformation(self):
        """Test transformation inverse accuracy."""
        # Random transformation matrix
        T = trimesh.transformations.random_rotation_matrix()
        T[:3, 3] = np.random.randn(3)
        
        # Apply transformation and its inverse
        points = np.random.randn(50, 3)
        points_h = np.c_[points, np.ones(len(points))]
        
        transformed = (T @ points_h.T).T[:, :3]
        recovered = (np.linalg.inv(T) @ np.c_[transformed, np.ones(len(transformed))].T).T[:, :3]
        
        # Should recover original points
        assert_allclose(recovered, points, rtol=1e-10, atol=1e-12)
```

## 6. Property-Based Testing for Geometry Algorithms

### Setting Up Hypothesis
```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import trimesh


# Custom strategies for 3D geometry
@st.composite
def valid_3d_points(draw, min_points=3, max_points=1000):
    """Generate valid 3D point clouds."""
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    points = draw(arrays(
        dtype=np.float64,
        shape=(n_points, 3),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
    ))
    return points


@st.composite
def valid_triangles(draw, vertices):
    """Generate valid triangle indices for given vertices."""
    n_vertices = len(vertices)
    n_triangles = draw(st.integers(min_value=1, max_value=n_vertices * 2))
    
    triangles = []
    for _ in range(n_triangles):
        # Generate three distinct indices
        indices = draw(st.lists(
            st.integers(0, n_vertices - 1),
            min_size=3, max_size=3, unique=True
        ))
        triangles.append(indices)
    
    return np.array(triangles)


@st.composite
def valid_mesh(draw):
    """Generate a valid trimesh object."""
    vertices = draw(valid_3d_points(min_points=4))
    faces = draw(valid_triangles(vertices))
    
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        assume(mesh.is_valid)  # Only use valid meshes
        return mesh
    except:
        assume(False)  # Skip invalid configurations
```

### Property-Based Tests
```python
class TestGeometryProperties:
    """Property-based tests for geometry algorithms."""
    
    @given(points=valid_3d_points())
    def test_normalize_denormalize_invariant(self, points):
        """Test that normalize->denormalize returns original points."""
        from recodestl.sampling.base import normalize_points, denormalize_points
        
        normalized, info = normalize_points(points)
        recovered = denormalize_points(normalized, info)
        
        assert_allclose(recovered, points, rtol=1e-10, atol=1e-12)
    
    @given(mesh=valid_mesh())
    def test_mesh_properties_consistency(self, mesh):
        """Test consistency of mesh properties."""
        if mesh.is_watertight:
            # Watertight meshes should have consistent winding
            assert mesh.is_winding_consistent
            # Volume should be positive for properly oriented meshes
            if mesh.is_winding_consistent:
                assert mesh.volume >= 0
        
        # Face areas should all be non-negative
        assert np.all(mesh.area_faces >= 0)
        
        # Total area should equal sum of face areas
        assert_allclose(mesh.area, np.sum(mesh.area_faces), rtol=1e-10)
    
    @given(
        points=valid_3d_points(min_points=10),
        n_samples=st.integers(min_value=1, max_value=10)
    )
    def test_farthest_point_sampling_properties(self, points, n_samples):
        """Test properties of farthest point sampling."""
        assume(n_samples <= len(points))
        
        from recodestl.sampling.base import farthest_point_sampling
        sampled, indices = farthest_point_sampling(points, n_samples)
        
        # Properties that should always hold
        assert len(sampled) == n_samples
        assert len(indices) == n_samples
        assert len(np.unique(indices)) == n_samples  # All unique
        assert np.all(indices < len(points))  # Valid indices
        assert_allclose(sampled, points[indices])  # Correct selection
    
    @given(mesh=valid_mesh(), num_points=st.integers(min_value=10, max_value=1000))
    @settings(max_examples=50)  # Limit examples for performance
    def test_sampling_on_surface(self, mesh, num_points):
        """Test that sampled points lie on mesh surface."""
        from recodestl.sampling.base import sample_surface_points
        
        points, face_indices = sample_surface_points(mesh, num_points)
        
        # All points should be within mesh bounds
        bounds = mesh.bounds
        assert np.all(points >= bounds[0] - 1e-6)
        assert np.all(points <= bounds[1] + 1e-6)
        
        # Check a subset of points are actually on triangles
        for i in range(min(10, len(points))):
            face = mesh.faces[face_indices[i]]
            triangle = mesh.vertices[face]
            
            # Point should be coplanar with triangle
            # (This is a simplified check - full barycentric validation would be better)
            v0 = triangle[1] - triangle[0]
            v1 = triangle[2] - triangle[0]
            normal = np.cross(v0, v1)
            normal = normal / np.linalg.norm(normal)
            
            dist = np.abs(np.dot(points[i] - triangle[0], normal))
            assert dist < 1e-6
```

### Advanced Property Tests
```python
@st.composite
def transformation_matrix(draw):
    """Generate valid 4x4 transformation matrices."""
    # Random rotation
    angle = draw(st.floats(min_value=0, max_value=2*np.pi))
    axis = draw(arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-1, max_value=1, exclude_min=True, exclude_max=True)
    ))
    axis = axis / np.linalg.norm(axis)
    
    # Build transformation matrix
    T = trimesh.transformations.rotation_matrix(angle, axis)
    
    # Add translation
    translation = draw(arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-10, max_value=10)
    ))
    T[:3, 3] = translation
    
    return T


class TestTransformationProperties:
    """Property-based tests for transformations."""
    
    @given(T1=transformation_matrix(), T2=transformation_matrix())
    def test_transformation_associativity(self, T1, T2):
        """Test that matrix multiplication is associative."""
        T3 = trimesh.transformations.random_rotation_matrix()
        
        # (T1 * T2) * T3 should equal T1 * (T2 * T3)
        result1 = (T1 @ T2) @ T3
        result2 = T1 @ (T2 @ T3)
        
        assert_allclose(result1, result2, rtol=1e-10, atol=1e-12)
    
    @given(T=transformation_matrix(), points=valid_3d_points())
    def test_transformation_preserves_distances(self, T, points):
        """Test that rigid transformations preserve distances."""
        # Extract rotation part
        R = T[:3, :3]
        
        # Check if it's a valid rotation (orthogonal with det=1)
        if np.abs(np.linalg.det(R) - 1) < 1e-6:
            RTR = R.T @ R
            if np.allclose(RTR, np.eye(3), atol=1e-6):
                # This is a rigid transformation
                # Transform points
                points_h = np.c_[points, np.ones(len(points))]
                transformed = (T @ points_h.T).T[:, :3]
                
                # Check distance preservation for a few pairs
                for i in range(min(5, len(points)-1)):
                    dist_before = np.linalg.norm(points[i] - points[i+1])
                    dist_after = np.linalg.norm(transformed[i] - transformed[i+1])
                    assert_allclose(dist_before, dist_after, rtol=1e-10)
```

## 7. Testing Visualization Code Without Displaying

### Configure Matplotlib for Headless Testing
```python
# conftest.py or at the top of test files
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
```

### Testing Visualization Functions
```python
import pytest
from unittest.mock import patch, Mock
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


class TestVisualization:
    """Test visualization functions without display."""
    
    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Ensure matplotlib uses non-interactive backend."""
        plt.ioff()  # Turn off interactive mode
        yield
        plt.close('all')  # Clean up any figures
    
    def test_mesh_visualization_creation(self, sphere_mesh):
        """Test that mesh visualization creates correct figure."""
        from recodestl.visualization import plot_mesh
        
        fig = plot_mesh(sphere_mesh)
        
        assert fig is not None
        assert len(fig.axes) > 0
        assert fig.axes[0].has_data()
        
        # Test that we can save the figure
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        assert buffer.getvalue()  # Should have content
    
    @patch('matplotlib.pyplot.show')
    def test_show_is_not_called(self, mock_show):
        """Test that plt.show() is properly mocked."""
        # Your visualization code that calls plt.show()
        plt.plot([1, 2, 3])
        plt.show()
        
        # Verify show was called but didn't actually display
        mock_show.assert_called_once()
    
    def test_3d_plot_without_display(self):
        """Test 3D plotting without display."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot some 3D data
        points = np.random.randn(100, 3)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        
        # Test that plot was created
        assert len(ax.collections) > 0
        
        # Save to buffer to verify it works
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        assert len(buffer.getvalue()) > 0
```

### Testing Trimesh Visualization
```python
class TestTrimeshVisualization:
    """Test trimesh visualization features."""
    
    @patch('trimesh.viewer.windowed.SceneViewer')
    def test_mesh_show_mocked(self, mock_viewer, sphere_mesh):
        """Test mesh.show() with mocked viewer."""
        # Call show
        sphere_mesh.show()
        
        # Verify viewer was created but not displayed
        mock_viewer.assert_called()
    
    def test_mesh_export_image(self, sphere_mesh, tmp_path):
        """Test exporting mesh visualization to image."""
        # Export to image file
        image_path = tmp_path / "mesh.png"
        
        # Create scene and export
        scene = trimesh.Scene(sphere_mesh)
        
        # Use trimesh's built-in PNG export if available
        try:
            png = scene.save_image(resolution=(400, 400))
            with open(image_path, 'wb') as f:
                f.write(png)
            assert image_path.exists()
            assert image_path.stat().st_size > 0
        except:
            # Skip if not available
            pytest.skip("Trimesh image export not available")
```

### Image Comparison Testing
```python
# Install: pip install pytest-mpl
import pytest
import matplotlib.pyplot as plt
import numpy as np


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images', tolerance=10)
def test_plot_generation():
    """Test plot generation with image comparison."""
    fig, ax = plt.subplots()
    
    # Generate test data
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    ax.plot(x, y)
    ax.set_title('Sine Wave')
    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    
    return fig


# Alternative: Manual image comparison
class TestImageComparison:
    """Manual image comparison tests."""
    
    def test_plot_consistency(self, tmp_path):
        """Test that plots are generated consistently."""
        # Generate plot twice
        def create_plot():
            fig, ax = plt.subplots(figsize=(6, 4))
            np.random.seed(42)
            ax.hist(np.random.randn(1000), bins=30)
            return fig
        
        fig1 = create_plot()
        fig2 = create_plot()
        
        # Save to temporary files
        path1 = tmp_path / "plot1.png"
        path2 = tmp_path / "plot2.png"
        
        fig1.savefig(path1, dpi=100)
        fig2.savefig(path2, dpi=100)
        
        # Compare file sizes (should be identical)
        assert path1.stat().st_size == path2.stat().st_size
        
        plt.close('all')
```

## Summary and Best Practices

1. **Test Organization**
   - Follow the module structure in your tests
   - Use descriptive test names that explain what is being tested
   - Group related tests in classes

2. **Fixtures**
   - Use session-scoped fixtures for expensive operations
   - Create factory fixtures for generating test data
   - Clean up resources in fixture teardown

3. **Numerical Testing**
   - Always use appropriate tolerances (rtol and atol)
   - Prefer `assert_allclose` over exact equality
   - Test edge cases and degenerate inputs

4. **Mocking**
   - Mock expensive operations and external dependencies
   - Use `pytest-mock` for cleaner mocking syntax
   - Verify mock calls to ensure correct usage

5. **Property-Based Testing**
   - Define properties that should always hold
   - Use Hypothesis strategies for generating test data
   - Limit examples for expensive operations

6. **Visualization Testing**
   - Set matplotlib backend to 'Agg' for headless testing
   - Mock display functions like `plt.show()`
   - Test figure generation without displaying

7. **Performance**
   - Mark slow tests with `@pytest.mark.slow`
   - Use caching for expensive test data generation
   - Consider parallel test execution with `pytest-xdist`

Remember: Good tests are maintainable, fast, and catch real bugs. Focus on testing behavior and edge cases rather than implementation details.