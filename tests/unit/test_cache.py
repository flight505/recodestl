"""Unit tests for caching functionality."""

import time
from pathlib import Path

import numpy as np
import pytest

from recodestl.core.config import CacheConfig
from recodestl.utils import CacheManager, create_cache_manager


class TestCacheManager:
    """Test cache manager functionality."""

    def test_cache_creation(self, temp_dir: Path):
        """Test creating a cache manager."""
        config = CacheConfig(
            enabled=True,
            cache_dir=temp_dir / "cache",
            max_size_gb=1.0,
            ttl_days=7
        )
        
        cache_mgr = CacheManager(config)
        
        assert cache_mgr.enabled
        assert cache_mgr.cache_dir.exists()
        assert cache_mgr.config.max_size_gb == 1.0

    def test_cache_disabled(self):
        """Test disabled cache behavior."""
        config = CacheConfig(enabled=False)
        cache_mgr = CacheManager(config)
        
        # All operations should be no-ops
        cache_mgr.set("key", "value")
        assert cache_mgr.get("key") is None
        
        stats = cache_mgr.get_stats()
        assert stats == {"enabled": False}

    def test_basic_get_set(self, cache_manager):
        """Test basic get/set operations."""
        # Set a value
        cache_manager.set("test_key", "test_value", ttl=60)
        
        # Get the value
        value = cache_manager.get("test_key")
        assert value == "test_value"
        
        # Non-existent key
        assert cache_manager.get("nonexistent") is None

    def test_ttl_expiration(self, cache_manager):
        """Test TTL expiration."""
        # Set with very short TTL
        cache_manager.set("expiring_key", "value", ttl=0.1)  # 100ms
        
        # Should exist immediately
        assert cache_manager.get("expiring_key") == "value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache_manager.get("expiring_key") is None

    def test_cache_key_generation(self, cache_manager, temp_dir: Path):
        """Test cache key generation."""
        # Create a test file
        test_file = temp_dir / "test.stl"
        test_file.write_text("test content")
        
        # Generate key without params
        key1 = cache_manager.generate_key(test_file)
        assert isinstance(key1, str)
        assert len(key1) > 0
        
        # Generate key with params
        key2 = cache_manager.generate_key(
            test_file,
            params={"method": "adaptive", "points": 256}
        )
        assert key1 != key2  # Different params = different key
        
        # Same params = same key
        key3 = cache_manager.generate_key(
            test_file,
            params={"method": "adaptive", "points": 256}
        )
        assert key2 == key3
        
        # With prefix
        key4 = cache_manager.generate_key(test_file, prefix="mesh")
        assert key4.startswith("mesh_")

    def test_file_hash_consistency(self, cache_manager, temp_dir: Path):
        """Test file hash consistency."""
        # Create a test file
        test_file = temp_dir / "test.dat"
        test_file.write_bytes(b"Hello, World!" * 1000)
        
        # Get hash multiple times
        hash1 = cache_manager._get_file_hash(test_file)
        hash2 = cache_manager._get_file_hash(test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256

    def test_mesh_caching(self, cache_manager):
        """Test mesh-specific caching."""
        # Mock mesh data
        mesh_data = {"vertices": [[0, 0, 0], [1, 1, 1]], "faces": [[0, 1, 2]]}
        
        # Cache mesh
        cache_manager.cache_mesh("test_mesh_key", mesh_data)
        
        # Retrieve mesh
        cached = cache_manager.get_mesh("test_mesh_key")
        assert cached == mesh_data

    def test_point_cloud_caching(self, cache_manager):
        """Test point cloud caching."""
        # Create point cloud
        points = np.random.rand(256, 3).astype(np.float32)
        metadata = {"method": "adaptive", "num_points": 256}
        
        # Cache point cloud
        cache_manager.cache_point_cloud("test_pc_key", points, metadata)
        
        # Retrieve point cloud
        cached_points, cached_meta = cache_manager.get_point_cloud("test_pc_key")
        
        assert cached_points is not None
        assert np.allclose(cached_points, points)
        assert cached_meta == metadata

    def test_cad_code_caching(self, cache_manager):
        """Test CAD code caching."""
        code = """
import cadquery as cq
result = cq.Workplane("XY").box(1, 1, 1)
"""
        metadata = {"model": "cad-recode-v1.5", "temperature": 0.0}
        
        # Cache code
        cache_manager.cache_cad_code("test_cad_key", code, metadata)
        
        # Retrieve code
        cached_code, cached_meta = cache_manager.get_cad_code("test_cad_key")
        
        assert cached_code == code
        assert cached_meta == metadata

    def test_cache_delete(self, cache_manager):
        """Test cache deletion."""
        # Set a value
        cache_manager.set("delete_me", "value")
        assert cache_manager.get("delete_me") == "value"
        
        # Delete it
        deleted = cache_manager.delete("delete_me")
        assert deleted is True
        
        # Should be gone
        assert cache_manager.get("delete_me") is None
        
        # Delete non-existent
        assert cache_manager.delete("nonexistent") is False

    def test_cache_clear(self, cache_manager):
        """Test clearing entire cache."""
        # Add multiple items
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")
        
        # Clear cache
        cache_manager.clear()
        
        # All should be gone
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") is None

    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        # Clear first
        cache_manager.clear()
        
        # Add some items
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Get stats
        stats = cache_manager.get_stats()
        
        assert stats["enabled"] is True
        assert stats["entries"] >= 2
        assert stats["size_mb"] > 0
        assert "location" in stats
        assert "hit_rate" in stats

    def test_cache_decorator(self, cache_manager):
        """Test cache decorator functionality."""
        call_count = 0
        
        @cache_manager.cache_decorator(prefix="test", ttl=60)
        def expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return x + y
        
        # First call - should execute
        result1 = expensive_function(5, 3)
        assert result1 == 8
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(5, 3)
        assert result2 == 8
        assert call_count == 1  # Not incremented
        
        # Different args - should execute
        result3 = expensive_function(10, 20)
        assert result3 == 30
        assert call_count == 2

    def test_create_cache_manager_function(self, temp_dir: Path):
        """Test the create_cache_manager convenience function."""
        config = CacheConfig(
            enabled=True,
            cache_dir=temp_dir / "cache2"
        )
        
        cache_mgr = create_cache_manager(config)
        
        assert isinstance(cache_mgr, CacheManager)
        assert cache_mgr.enabled
        assert cache_mgr.cache_dir == temp_dir / "cache2"


class TestCacheIntegration:
    """Integration tests for caching with other components."""

    @pytest.mark.integration
    def test_mesh_loader_caching(self, sample_stl_path: Path, cache_manager):
        """Test mesh loader with caching."""
        from recodestl.processing import MeshLoader
        
        loader = MeshLoader(show_progress=False, cache_manager=cache_manager)
        
        # Time first load
        start = time.time()
        mesh1 = loader.load(sample_stl_path)
        load_time1 = time.time() - start
        
        # Time second load (cached)
        start = time.time()
        mesh2 = loader.load(sample_stl_path)
        load_time2 = time.time() - start
        
        # Cached should be much faster
        assert load_time2 < load_time1 * 0.5
        
        # Should return equivalent meshes
        assert np.allclose(mesh1.vertices, mesh2.vertices)

    @pytest.mark.integration
    def test_sampling_caching(self, simple_box_mesh, cache_manager):
        """Test sampling with caching."""
        from recodestl.sampling import AdaptiveSampler
        
        sampler = AdaptiveSampler(
            num_points=256,
            seed=42,
            cache_manager=cache_manager
        )
        
        # Time first sample
        start = time.time()
        points1 = sampler.sample(simple_box_mesh)
        sample_time1 = time.time() - start
        
        # Time second sample (cached)
        start = time.time()
        points2 = sampler.sample(simple_box_mesh)
        sample_time2 = time.time() - start
        
        # Cached should be much faster
        assert sample_time2 < sample_time1 * 0.5
        
        # Should return same points
        assert np.allclose(points1, points2)