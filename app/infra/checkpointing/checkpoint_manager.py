"""
Redis Checkpointer Manager for LangChain Agents
Manages Redis-based checkpointer for short-term memory storage
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.logging import logger
from app.core.config import settings


class CheckpointerManager:
    """
    Manages Redis checkpointer for LangChain agents.
    Provides singleton access to checkpointer instance.
    """
    
    _instance: Optional['CheckpointerManager'] = None
    _checkpointer: Optional[Any] = None
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
        """Initialize Redis checkpointer or fallback to in-memory"""
        try:
            # Try to use Redis if available
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            logger.info(f"Initializing Redis checkpointer with URL: {redis_url}")
            
            self._checkpointer = RedisSaver.from_conn_string(redis_url)
            self._checkpointer.setup()
            self._use_redis = True
            logger.info("Redis checkpointer initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Redis checkpointer: {e}. "
                "Falling back to in-memory checkpointer."
            )
            # Fallback to in-memory checkpointer
            self._checkpointer = InMemorySaver()
            self._use_redis = False
            logger.info("Using in-memory checkpointer as fallback")
    
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
        self._checkpointer = None
        self._initialize_checkpointer()
    
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

