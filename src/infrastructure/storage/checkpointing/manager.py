"""
Redis Checkpointer Manager for LangChain Agents
Manages Redis-based checkpointer for short-term memory storage
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.checkpoint.redis import RedisSaver

from src.shared.config.logging import logger
from src.shared.config.settings import settings


class CheckpointerManager:
    """
    Manages Redis checkpointer for LangChain agents.
    Provides singleton access to checkpointer instance.
    """
    
    _instance: Optional['CheckpointerManager'] = None
    _checkpointer: Optional[Any] = None
    _checkpointer_context: Optional[Any] = None
    _use_redis: bool = True
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize checkpointer manager"""
        if self._checkpointer is None:
            self._initialize_checkpointer()
    
    def _initialize_checkpointer(self) -> None:
        """Initialize Redis checkpointer. Raises exception if Redis is unavailable."""
        # Try to use Redis if available
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
        logger.info(f"Initializing Redis checkpointer with URL: {redis_url}")
        
        # RedisSaver.from_conn_string() returns a context manager
        # We need to enter it to get the actual checkpointer instance
        self._checkpointer_context = RedisSaver.from_conn_string(redis_url)
        self._checkpointer = self._checkpointer_context.__enter__()
        self._checkpointer.setup()
        self._use_redis = True
        logger.info("Redis checkpointer initialized successfully")
    
    @property
    def checkpointer(self):
        """Get the checkpointer instance"""
        if self._checkpointer is None:
            self._initialize_checkpointer()
        return self._checkpointer
    
    @property
    def is_redis(self) -> bool:
        """Check if using Redis checkpointer"""
        return self._use_redis
    
    def reset(self) -> None:
        """Reset checkpointer (reinitialize)"""
        # Clean up existing context manager if using Redis
        if self._checkpointer_context is not None and self._use_redis:
            try:
                self._checkpointer_context.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up Redis checkpointer context: {e}")
        
        self._checkpointer = None
        self._checkpointer_context = None
        self._initialize_checkpointer()
    
    def cleanup(self) -> None:
        """Clean up checkpointer resources (call on application shutdown)"""
        if self._checkpointer_context is not None and self._use_redis:
            try:
                self._checkpointer_context.__exit__(None, None, None)
                logger.info("Redis checkpointer context cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up Redis checkpointer context: {e}")
    
    def get_config(self, thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration dict for agent invocation with thread_id.
        
        Args:
            thread_id: Thread/session identifier
            user_id: Optional user identifier
            
        Returns:
            Configuration dict for agent.invoke()
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        if user_id:
            config["configurable"]["user_id"] = user_id
        return config


# Global singleton instance
_checkpointer_manager: Optional[CheckpointerManager] = None


def get_checkpointer_manager() -> CheckpointerManager:
    """
    Get the global checkpointer manager instance (singleton).
    
    Returns:
        CheckpointerManager instance
    """
    global _checkpointer_manager
    if _checkpointer_manager is None:
        _checkpointer_manager = CheckpointerManager()
    return _checkpointer_manager


def get_checkpointer():
    """
    Get the checkpointer instance.
    
    Returns:
        Checkpointer instance (RedisSaver or InMemorySaver)
    """
    return get_checkpointer_manager().checkpointer


@contextmanager
def get_checkpointer_context():
    """
    Context manager for checkpointer (useful for async operations).
    
    Usage:
        with get_checkpointer_context() as checkpointer:
            # use checkpointer
    """
    manager = get_checkpointer_manager()
    yield manager.checkpointer

