"""Caching system for RecodeSTL using DiskCache and joblib."""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import numpy as np
from diskcache import Cache
from joblib import Memory

from recodestl.core.config import CacheConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheManager:
    """Manages caching for the RecodeSTL pipeline."""

    # TTL values in seconds
    TTL_MESH = 7 * 24 * 3600     # 7 days
    TTL_POINTCLOUD = 3 * 24 * 3600  # 3 days
    TTL_CAD_CODE = 1 * 24 * 3600    # 1 day
    TTL_STEP_FILE = 14 * 24 * 3600  # 14 days

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.

        Args:
            config: Cache configuration
        """
        if config is None:
            config = CacheConfig()
            
        self.config = config
        self.enabled = config.enabled
        
        if not self.enabled:
            logger.info("Cache disabled")
            return
            
        # Create cache directory
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DiskCache for general caching
        self.cache = Cache(
            str(self.cache_dir / "general"),
            size_limit=int(config.max_size_gb * 1024**3),
            eviction_policy="least-recently-used",
        )
        
        # Initialize joblib Memory for numpy arrays
        self.memory = Memory(
            str(self.cache_dir / "arrays"),
            verbose=0,
            compress=True,
        )
        
        logger.info(f"Cache initialized at {self.cache_dir}")
        logger.info(f"Cache size limit: {config.max_size_gb}GB")

    def generate_key(
        self,
        file_path: Union[str, Path],
        params: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> str:
        """Generate cache key from file and parameters.

        Args:
            file_path: Path to file
            params: Optional parameters dict
            prefix: Optional key prefix

        Returns:
            Cache key string
        """
        file_path = Path(file_path)
        
        # Get file hash (first 16 chars of SHA256)
        file_hash = self._get_file_hash(file_path)
        
        # Get parameters hash
        param_hash = ""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            
        # Include file modification time
        mtime = int(file_path.stat().st_mtime)
        
        # Build key
        key_parts = [prefix] if prefix else []
        key_parts.extend([file_hash, param_hash, str(mtime)])
        key = "_".join(filter(None, key_parts))
        
        return key

    def _get_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Get hash of file contents.

        Args:
            file_path: Path to file
            chunk_size: Chunk size for reading

        Returns:
            First 16 characters of SHA256 hash
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
                
        return sha256.hexdigest()[:16]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        if not self.enabled:
            return default
            
        try:
            value = self.cache.get(key, default)
            if value != default:
                logger.debug(f"Cache hit: {key}")
            else:
                logger.debug(f"Cache miss: {key}")
            return value
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tag: Optional tag for grouped operations
        """
        if not self.enabled:
            return
            
        try:
            expire = ttl if ttl else self.config.ttl_days * 24 * 3600
            self.cache.set(key, value, expire=expire, tag=tag)
            logger.debug(f"Cache set: {key} (TTL: {expire}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            result = self.cache.delete(key)
            if result:
                logger.debug(f"Cache delete: {key}")
            return result
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def clear(self) -> None:
        """Clear entire cache."""
        if not self.enabled:
            return
            
        try:
            self.cache.clear()
            self.memory.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def evict_expired(self) -> int:
        """Evict expired entries.

        Returns:
            Number of entries evicted
        """
        if not self.enabled:
            return 0
            
        try:
            # DiskCache handles expiration automatically
            # We'll check for expired entries manually
            evicted_count = 0
            current_time = time.time()
            
            # Check all cache entries
            for key in list(self.cache.iterkeys()):
                try:
                    # Get the entry with its metadata
                    value = self.cache.get(key, expire_time=True)
                    if value is None:
                        # Already expired
                        evicted_count += 1
                except Exception:
                    pass
                    
            return evicted_count
        except Exception as e:
            logger.warning(f"Cache evict error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {"enabled": False}
            
        try:
            # Get cache statistics
            stats = {
                "enabled": True,
                "location": str(self.cache_dir),
                "size_limit_gb": self.config.max_size_gb,
                "entries": len(self.cache),
                "size_bytes": self.cache.volume(),
                "size_mb": self.cache.volume() / 1024 / 1024,
                "hits": self.cache.stats(enable=True)[0],
                "misses": self.cache.stats(enable=True)[1],
            }
            
            # Calculate hit rate
            total = stats["hits"] + stats["misses"]
            if total > 0:
                stats["hit_rate"] = stats["hits"] / total
            else:
                stats["hit_rate"] = 0.0
                
            return stats
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}

    def cache_mesh(
        self,
        key: str,
        mesh_data: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache processed mesh data.

        Args:
            key: Cache key
            mesh_data: Mesh data to cache
            ttl: Optional TTL override
        """
        self.set(
            f"mesh_{key}",
            mesh_data,
            ttl=ttl or self.TTL_MESH,
            tag="mesh"
        )

    def get_mesh(self, key: str) -> Optional[Any]:
        """Get cached mesh data.

        Args:
            key: Cache key

        Returns:
            Cached mesh data or None
        """
        return self.get(f"mesh_{key}")

    def cache_point_cloud(
        self,
        key: str,
        points: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache point cloud data.

        Args:
            key: Cache key
            points: Point cloud array
            metadata: Optional metadata
            ttl: Optional TTL override
        """
        if not self.enabled:
            return
            
        # Store point cloud data with metadata
        data = {
            "points": points,
            "metadata": metadata or {},
            "shape": points.shape,
            "dtype": str(points.dtype),
        }
        
        self.set(
            f"pc_{key}",
            data,
            ttl=ttl or self.TTL_POINTCLOUD,
            tag="pointcloud"
        )

    def get_point_cloud(
        self,
        key: str
    ) -> tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Get cached point cloud data.

        Args:
            key: Cache key

        Returns:
            Tuple of (points, metadata)
        """
        if not self.enabled:
            return None, None
            
        data = self.get(f"pc_{key}")
        if data:
            return data.get("points"), data.get("metadata")
                
        return None, None

    def cache_cad_code(
        self,
        key: str,
        code: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache generated CAD code.

        Args:
            key: Cache key
            code: Generated code
            metadata: Optional metadata
            ttl: Optional TTL override
        """
        data = {"code": code, "metadata": metadata}
        self.set(
            f"cad_{key}",
            data,
            ttl=ttl or self.TTL_CAD_CODE,
            tag="cad"
        )

    def get_cad_code(self, key: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Get cached CAD code.

        Args:
            key: Cache key

        Returns:
            Tuple of (code, metadata)
        """
        data = self.get(f"cad_{key}")
        if data:
            return data.get("code"), data.get("metadata")
        return None, None

    def cache_decorator(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
    ):
        """Decorator for caching function results.

        Args:
            prefix: Cache key prefix
            ttl: Time to live
            key_func: Function to generate cache key from args

        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs) -> T:
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Simple key from args
                    key = hashlib.md5(
                        f"{args}{kwargs}".encode()
                    ).hexdigest()[:16]
                    
                full_key = f"{prefix}_{key}"
                
                # Check cache
                result = self.get(full_key)
                if result is not None:
                    return result
                    
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(full_key, result, ttl=ttl)
                
                return result
                
            return wrapper
        return decorator


def create_cache_manager(
    config: Optional[CacheConfig] = None
) -> CacheManager:
    """Create a cache manager instance.

    Args:
        config: Optional cache configuration

    Returns:
        CacheManager instance
    """
    return CacheManager(config)