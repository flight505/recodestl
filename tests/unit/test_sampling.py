"""Unit tests for point cloud sampling strategies."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from recodestl.sampling import (
    AdaptiveSampler,
    BlueNoiseSampler,
    PoissonDiskSampler,
    SamplingFactory,
    UniformSampler,
    farthest_point_sampling,
    normalize_points,
)


class TestSamplingBase:
    """Test base sampling functionality."""

    def test_farthest_point_sampling(self):
        """Test FPS algorithm."""
        # Create random points
        points = np.random.rand(1000, 3)
        
        # Sample 100 points
        sampled, indices = farthest_point_sampling(points, 100)
        
        assert sampled.shape == (100, 3)
        assert indices.shape == (100,)
        assert len(np.unique(indices)) == 100
        
        # Check that sampled points are from original
        assert np.allclose(sampled, points[indices])

    def test_farthest_point_sampling_edge_cases(self):
        """Test FPS edge cases."""
        points = np.random.rand(10, 3)
        
        # Sample all points
        sampled, indices = farthest_point_sampling(points, 10)
        assert len(sampled) == 10
        
        # Try to sample more than available
        with pytest.raises(ValueError):
            farthest_point_sampling(points, 20)

    def test_normalize_points(self):
        """Test point normalization."""
        # Create points with known bounds
        points = np.array([
            [-1, -2, -3],
            [1, 2, 3],
            [0, 0, 0],
        ])
        
        normalized, info = normalize_points(points)
        
        # Check centering
        assert np.allclose(np.mean(normalized, axis=0), 0)
        
        # Check scale (should fit in unit cube)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Check info
        assert "center" in info
        assert "scale" in info


class TestUniformSampler:
    """Test uniform sampling strategy."""

    def test_uniform_sampling_basic(self, simple_box_mesh: trimesh.Trimesh):
        """Test basic uniform sampling."""
        sampler = UniformSampler(num_points=256, seed=42)
        points = sampler.sample(simple_box_mesh)
        
        assert points.shape == (256, 3)
        assert points.dtype == np.float32
        
        # Points should be on surface (within tolerance)
        distances = simple_box_mesh.nearest.on_surface(points)[1]
        assert np.max(distances) < 1e-3  # Within 0.001 units

    def test_uniform_sampling_with_fps(self, simple_box_mesh: trimesh.Trimesh):
        """Test uniform sampling with FPS refinement."""
        sampler = UniformSampler(
            num_points=128,
            seed=42,
            use_farthest_point=True,
            oversample_ratio=10.0
        )
        points = sampler.sample(simple_box_mesh)
        
        assert points.shape == (128, 3)
        
        # Check distribution is relatively uniform
        # FPS should give well-distributed points
        from scipy.spatial import distance_matrix
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Standard deviation of minimum distances should be low
        assert np.std(min_distances) < np.mean(min_distances) * 0.5

    def test_uniform_sampling_reproducibility(self, simple_box_mesh: trimesh.Trimesh):
        """Test sampling reproducibility with seed."""
        sampler1 = UniformSampler(num_points=100, seed=42, use_farthest_point=False)
        sampler2 = UniformSampler(num_points=100, seed=42, use_farthest_point=False)
        
        points1 = sampler1._sample_impl(simple_box_mesh)
        points2 = sampler2._sample_impl(simple_box_mesh)
        
        # With same seed, should get similar points (not exact due to mesh sampling)
        # Check that at least some points are very close
        min_distances = []
        for p1 in points1[:10]:  # Check first 10 points
            distances = np.linalg.norm(points2 - p1, axis=1)
            min_distances.append(np.min(distances))
        assert np.mean(min_distances) < 0.1  # Points should be close on average

    def test_uniform_sampling_with_cache(self, simple_box_mesh: trimesh.Trimesh, cache_manager):
        """Test uniform sampling with caching."""
        sampler = UniformSampler(num_points=128, cache_manager=cache_manager)
        
        # First sample - cache miss
        points1 = sampler.sample(simple_box_mesh)
        
        # Second sample - cache hit
        points2 = sampler.sample(simple_box_mesh)
        
        # Should return the same points
        assert np.allclose(points1, points2)
        
        # Check cache was used
        stats = cache_manager.get_stats()
        assert stats["hits"] > 0


class TestPoissonDiskSampler:
    """Test Poisson disk sampling strategy."""

    def test_poisson_sampling_basic(self, simple_box_mesh: trimesh.Trimesh):
        """Test basic Poisson disk sampling."""
        sampler = PoissonDiskSampler(num_points=64, seed=42)
        points = sampler._sample_impl(simple_box_mesh)
        
        assert points.shape == (64, 3)
        assert points.dtype == np.float32
        
        # Points should be on surface (within tolerance)
        distances = simple_box_mesh.nearest.on_surface(points)[1]
        assert np.max(distances) < 1e-3  # Within 0.001 units

    @pytest.mark.slow
    def test_poisson_minimum_distance(self, simple_box_mesh: trimesh.Trimesh):
        """Test that Poisson sampling maintains minimum distance."""
        sampler = PoissonDiskSampler(
            num_points=32,
            seed=42,
            k_candidates=30
        )
        points = sampler._sample_impl(simple_box_mesh)
        
        # Calculate pairwise distances
        from scipy.spatial import distance_matrix
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Minimum distance should be reasonable
        min_dist = np.min(distances)
        avg_dist = np.mean(distances[distances != np.inf])
        
        # Poisson disk should maintain some minimum distance
        assert min_dist > avg_dist * 0.1


class TestAdaptiveSampler:
    """Test adaptive sampling strategy."""

    def test_adaptive_sampling_basic(self, complex_mesh: trimesh.Trimesh):
        """Test basic adaptive sampling."""
        sampler = AdaptiveSampler(num_points=256, seed=42)
        points = sampler._sample_impl(complex_mesh)
        
        assert points.shape == (256, 3)
        assert points.dtype == np.float32
        
        # Points should be on surface
        distances = complex_mesh.nearest.on_surface(points)[1]
        assert np.allclose(distances, 0, atol=1e-6)

    def test_adaptive_sampling_weights(self, complex_mesh: trimesh.Trimesh):
        """Test adaptive sampling with different weights."""
        # High curvature weight
        sampler1 = AdaptiveSampler(
            num_points=128,
            curvature_weight=0.9,
            edge_weight=0.05,
            normal_weight=0.05
        )
        
        # High edge weight
        sampler2 = AdaptiveSampler(
            num_points=128,
            curvature_weight=0.1,
            edge_weight=0.8,
            normal_weight=0.1
        )
        
        points1 = sampler1._sample_impl(complex_mesh)
        points2 = sampler2._sample_impl(complex_mesh)
        
        # Both should produce valid points
        assert points1.shape == (128, 3)
        assert points2.shape == (128, 3)
        
        # But distributions should be different
        # (Can't easily test exact distribution without visual inspection)

    def test_adaptive_importance_calculation(self, complex_mesh: trimesh.Trimesh):
        """Test importance score calculation."""
        sampler = AdaptiveSampler(num_points=128)
        
        # Calculate importance scores
        importance = sampler._calculate_importance(complex_mesh)
        
        assert importance.shape == (len(complex_mesh.faces),)
        assert np.all(importance >= 0)
        assert np.all(importance <= 1)
        assert np.max(importance) > 0  # Some faces should have high importance

    def test_adaptive_edge_detection(self, simple_box_mesh: trimesh.Trimesh):
        """Test edge importance calculation."""
        sampler = AdaptiveSampler(edge_angle_threshold=30.0)
        
        # Box should have sharp edges
        edge_importance = sampler._calculate_edge_importance(simple_box_mesh)
        
        assert edge_importance.shape == (len(simple_box_mesh.faces),)
        assert np.any(edge_importance > 0)  # Should detect some edges


class TestBlueNoiseSampler:
    """Test blue noise sampling strategy."""

    def test_blue_noise_sampling_basic(self, simple_box_mesh: trimesh.Trimesh):
        """Test basic blue noise sampling."""
        sampler = BlueNoiseSampler(
            num_points=64,
            seed=42,
            iterations=5,  # Fewer iterations for speed
            initial_samples=1000
        )
        points = sampler._sample_impl(simple_box_mesh)
        
        assert points.shape == (64, 3)
        assert points.dtype == np.float32
        
        # Points should be on surface
        distances = simple_box_mesh.nearest.on_surface(points)[1]
        assert np.allclose(distances, 0, atol=1e-5)  # Slightly higher tolerance due to Lloyd relaxation

    @pytest.mark.slow
    def test_blue_noise_distribution(self, simple_cylinder_mesh: trimesh.Trimesh):
        """Test that blue noise produces even distribution."""
        sampler = BlueNoiseSampler(
            num_points=128,
            seed=42,
            iterations=10
        )
        points = sampler._sample_impl(simple_cylinder_mesh)
        
        # Check distribution quality
        from scipy.spatial import distance_matrix
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Blue noise should have very uniform distribution
        cv = np.std(min_distances) / np.mean(min_distances)
        assert cv < 0.3  # Coefficient of variation should be low


class TestSamplingFactory:
    """Test sampling factory."""

    def test_factory_create_all_methods(self, simple_box_mesh: trimesh.Trimesh):
        """Test creating all sampling methods."""
        methods = SamplingFactory.available_methods()
        
        for method in methods:
            sampler = SamplingFactory.create(method, num_points=64)
            points = sampler._sample_impl(simple_box_mesh)
            
            assert points.shape == (64, 3)

    def test_factory_unknown_method(self):
        """Test creating unknown method."""
        with pytest.raises(ValueError) as exc_info:
            SamplingFactory.create("unknown_method")
        
        assert "Unknown sampling method" in str(exc_info.value)

    def test_factory_with_kwargs(self, simple_box_mesh: trimesh.Trimesh):
        """Test creating sampler with custom parameters."""
        sampler = SamplingFactory.create(
            "adaptive",
            num_points=128,
            curvature_weight=0.8,
            edge_weight=0.2
        )
        
        assert sampler.num_points == 128
        # Weights are normalized, so check the ratio instead
        assert abs(sampler.curvature_weight - 0.8) < 0.1
        assert abs(sampler.edge_weight - 0.2) < 0.1

    def test_factory_available_methods(self):
        """Test listing available methods."""
        methods = SamplingFactory.available_methods()
        
        assert "uniform" in methods
        assert "poisson" in methods
        assert "adaptive" in methods
        assert "blue_noise" in methods
        assert len(methods) >= 4


class TestSamplingIntegration:
    """Integration tests for sampling strategies."""

    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["uniform", "poisson", "adaptive", "blue_noise"])
    def test_all_methods_on_sonos(self, sonos_stl_path: Path, method: str):
        """Test all sampling methods on the Sonos fixture."""
        from recodestl.processing import load_stl
        
        mesh = load_stl(sonos_stl_path, show_progress=False)
        sampler = SamplingFactory.create(method, num_points=256, seed=42)
        
        points = sampler.sample(mesh)
        
        assert points.shape == (256, 3)
        assert points.dtype == np.float32
        
        # All points should be on surface
        distances = mesh.nearest.on_surface(points)[1]
        assert np.max(distances) < 1e-3  # Within 1mm tolerance