"""
Caching system for FinNewsTrader with support for in-memory and Redis caching
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import pickle
from functools import wraps

from config import get_config
from logger import get_logger

logger = get_logger(__name__)

class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

class MemoryCache(CacheBackend):
    """In-memory cache implementation"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.max_size = 1000  # Maximum number of entries
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._cache:
            return True
        
        entry = self._cache[key]
        if 'expires_at' not in entry:
            return False
        
        return time.time() > entry['expires_at']
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [key for key in self._cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            self.delete(key)
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size using LRU eviction"""
        if len(self._cache) <= self.max_size:
            return
        
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        keys_to_remove = sorted_keys[:len(self._cache) - self.max_size]
        
        for key, _ in keys_to_remove:
            self.delete(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._cleanup_expired()
        
        if key not in self._cache or self._is_expired(key):
            return None
        
        self._access_times[key] = time.time()
        return self._cache[key]['value']
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            entry = {'value': value}
            
            if ttl is not None:
                entry['expires_at'] = time.time() + ttl
            
            self._cache[key] = entry
            self._access_times[key] = time.time()
            
            self._enforce_size_limit()
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self._cache.clear()
            self._access_times.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        self._cleanup_expired()
        return key in self._cache and not self._is_expired(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()
        return {
            'entries': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': 'N/A',  # Would need to track hits/misses for this
        }

class RedisCache(CacheBackend):
    """Redis cache implementation"""
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=False  # We'll handle encoding ourselves
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            data = self._serialize(value)
            if ttl is not None:
                return self.redis_client.setex(key, ttl, data)
            else:
                return self.redis_client.set(key, data)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            return self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {str(e)}")
            return False

class CacheManager:
    """Main cache manager that abstracts backend selection"""
    
    def __init__(self, backend_type: str = 'memory'):
        self.config = get_config()
        self.backend = MemoryCache()  # Start with memory cache
        self.default_ttl = self.config.data_sources.cache_ttl_seconds
        logger.info(f"Cache manager initialized with {type(self.backend).__name__} backend")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments"""
        # Create a unique key based on all arguments
        key_data = f"{prefix}:{':'.join(map(str, args))}"
        if kwargs:
            key_data += f":{json.dumps(kwargs, sort_keys=True)}"
        
        # Hash long keys to keep them manageable
        if len(key_data) > 200:
            key_data = f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
        
        return key_data
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl
        return self.backend.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return self.backend.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        return self.backend.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.backend.exists(key)
    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        return self._generate_key(prefix, *args, **kwargs)

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cached(prefix: str, ttl: int = None, key_func=None):
    """
    Decorator to cache function results
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds (None for default)
        key_func: Function to generate cache key from args
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache.cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = func(*args, **kwargs)
            
            if result is not None:  # Don't cache None results
                cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Convenience functions for common cache operations
def cache_news(ticker: str, articles: List[Dict], ttl: int = None):
    """Cache news articles for a ticker"""
    cache = get_cache_manager()
    key = cache.cache_key("news", ticker, datetime.now().date().isoformat())
    cache.set(key, articles, ttl)

def get_cached_news(ticker: str) -> Optional[List[Dict]]:
    """Get cached news for a ticker"""
    cache = get_cache_manager()
    key = cache.cache_key("news", ticker, datetime.now().date().isoformat())
    return cache.get(key)

def cache_stock_data(ticker: str, period: str, data: Any, ttl: int = None):
    """Cache stock data for a ticker"""
    cache = get_cache_manager()
    key = cache.cache_key("stock_data", ticker, period)
    cache.set(key, data, ttl)

def get_cached_stock_data(ticker: str, period: str) -> Optional[Any]:
    """Get cached stock data for a ticker"""
    cache = get_cache_manager()
    key = cache.cache_key("stock_data", ticker, period)
    return cache.get(key)

def cache_sentiment(text: str, sentiment: float, ttl: int = None):
    """Cache sentiment analysis result"""
    cache = get_cache_manager()
    # Use hash of text as key to avoid very long keys
    text_hash = hashlib.md5(text.encode()).hexdigest()
    key = cache.cache_key("sentiment", text_hash)
    cache.set(key, sentiment, ttl)

def get_cached_sentiment(text: str) -> Optional[float]:
    """Get cached sentiment for text"""
    cache = get_cache_manager()
    text_hash = hashlib.md5(text.encode()).hexdigest()
    key = cache.cache_key("sentiment", text_hash)
    return cache.get(key) 