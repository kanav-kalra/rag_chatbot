"""
Redis Session Storage - Handles Redis operations for session management
Decoupled from session manager for better separation of concerns.
"""
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import timedelta
import json
from urllib.parse import urlparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logging import logger
from app.core.config import settings

# Redis is required
try:
    import redis
except ImportError:
    raise ImportError(
        "Redis package is required for session management. "
        "Install it with: pip install redis"
    )


class RedisSessionStorage:
    """
    Redis-based storage for session management.
    Handles all Redis operations for sessions.
    """
    
    def __init__(self, redis_url: Optional[str] = None, redis_db: Optional[int] = None):
        """
        Initialize Redis session storage.
        
        Args:
            redis_url: Redis connection URL (default: from settings)
            redis_db: Redis database number (default: from settings)
        
        Raises:
            RuntimeError: If Redis connection fails
        """
        self.redis_url = redis_url or getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
        self.redis_db = redis_db if redis_db is not None else getattr(settings, 'REDIS_DB', 0)
        self._redis_client: Optional[redis.Redis] = None
        self._session_key_prefix = "session:"
        self._session_count_key = "session:count"
        
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection. Redis is required."""
        try:
            # Parse Redis URL and create client
            if self.redis_url.startswith('redis://') or self.redis_url.startswith('rediss://'):
                parsed = urlparse(self.redis_url)
                self._redis_client = redis.Redis(
                    host=parsed.hostname or 'localhost',
                    port=parsed.port or 6379,
                    db=self.redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            else:
                # Assume it's a host:port format
                parts = self.redis_url.split(':')
                host = parts[0] if len(parts) > 0 else 'localhost'
                port = int(parts[1]) if len(parts) > 1 else 6379
                self._redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=self.redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test connection
            self._redis_client.ping()
            logger.info(f"Redis session storage initialized: {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for session storage: {e}", exc_info=True)
            raise RuntimeError(
                f"Redis connection failed. Session management requires Redis. "
                f"Error: {str(e)}"
            ) from e
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for a session."""
        return f"{self._session_key_prefix}{session_id}"
    
    def serialize_session(self, session_data: dict) -> str:
        """Serialize session data to JSON string."""
        return json.dumps(session_data)
    
    def deserialize_session(self, data: str) -> dict:
        """Deserialize session data from JSON string."""
        return json.loads(data)
    
    def save_session(
        self,
        session_id: str,
        session_data: dict,
        ttl_seconds: int
    ) -> None:
        """
        Save session to Redis with TTL.
        
        Args:
            session_id: Session identifier
            session_data: Session data dictionary
            ttl_seconds: Time to live in seconds
        
        Raises:
            RuntimeError: If Redis operation fails
        """
        try:
            redis_key = self._get_session_key(session_id)
            serialized_data = self.serialize_session(session_data)
            self._redis_client.setex(redis_key, ttl_seconds, serialized_data)
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save session to Redis: {str(e)}") from e
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get session data from Redis.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data dictionary if found, None otherwise
        
        Raises:
            RuntimeError: If Redis operation fails
        """
        try:
            redis_key = self._get_session_key(session_id)
            session_data = self._redis_client.get(redis_key)
            if session_data:
                return self.deserialize_session(session_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get session from Redis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get session from Redis: {str(e)}") from e
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from Redis.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session was deleted, False if not found
        
        Raises:
            RuntimeError: If Redis operation fails
        """
        try:
            redis_key = self._get_session_key(session_id)
            deleted = self._redis_client.delete(redis_key) > 0
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete session from Redis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to delete session from Redis: {str(e)}") from e
    
    def increment_session_count(self) -> None:
        """Increment the session count counter."""
        try:
            self._redis_client.incr(self._session_count_key)
        except Exception as e:
            logger.error(f"Failed to increment session count: {e}", exc_info=True)
            raise RuntimeError(f"Failed to increment session count: {str(e)}") from e
    
    def decrement_session_count(self) -> None:
        """Decrement the session count counter."""
        try:
            self._redis_client.decr(self._session_count_key)
        except Exception as e:
            logger.error(f"Failed to decrement session count: {e}", exc_info=True)
            raise RuntimeError(f"Failed to decrement session count: {str(e)}") from e
    
    def get_session_count(self) -> int:
        """
        Get current session count from Redis counter.
        
        Returns:
            Number of active sessions
        
        Raises:
            RuntimeError: If Redis operation fails
        """
        try:
            count = self._redis_client.get(self._session_count_key)
            count_int = int(count) if count else 0
            # Ensure count is never negative (safeguard against drift)
            return max(0, count_int)
        except Exception as e:
            logger.error(f"Failed to get session count from Redis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get session count: {str(e)}") from e
    
    def scan_sessions(self, pattern: Optional[str] = None) -> list:
        """
        Scan all session keys from Redis.
        
        Args:
            pattern: Optional pattern to match (default: session:*)
        
        Returns:
            List of session keys (full keys, not just IDs)
        
        Raises:
            RuntimeError: If Redis operation fails
        """
        try:
            if pattern is None:
                pattern = f"{self._session_key_prefix}*"
            
            keys = []
            cursor = 0
            while True:
                cursor, batch = self._redis_client.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            
            # Filter out the counter key
            keys = [k for k in keys if k != self._session_count_key]
            return keys
        except Exception as e:
            logger.error(f"Failed to scan sessions from Redis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to scan sessions: {str(e)}") from e
    
    def extract_session_id_from_key(self, key: str) -> str:
        """
        Extract session ID from a Redis key.
        
        Args:
            key: Full Redis key (e.g., "session:abc123")
        
        Returns:
            Session ID (e.g., "abc123")
        """
        if key.startswith(self._session_key_prefix):
            return key[len(self._session_key_prefix):]
        return key

