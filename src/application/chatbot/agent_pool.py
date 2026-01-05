"""
Agent Pool Manager - Manages a pool of agent instances for efficient resource usage.
Designed for multi-server, multi-user deployments.
"""
import sys
from pathlib import Path
from typing import Callable, Optional, Any, Dict
from threading import Lock

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.config.logging import logger
from src.shared.config.settings import settings


class AgentPool:
    """
    Manages a pool of agent instances for efficient resource usage.
    Thread-safe implementation for production use.
    Designed for multi-server, multi-user deployments.
    
    The agent pool ensures that agents are reused across requests rather than
    creating new instances for each request, which significantly reduces
    memory usage and improves performance.
    """
    
    def __init__(
        self,
        agent_factory: Callable,
        pool_size: Optional[int] = None
    ):
        """
        Initialize agent pool.
        
        Args:
            agent_factory: Function that creates a new agent instance
            pool_size: Number of agents in the pool (default: from settings)
        """
        self.agent_factory = agent_factory
        self.pool_size = pool_size or settings.AGENT_POOL_SIZE
        
        # Initialize agent pool
        self._shared_agent: Optional[Any] = None
        self._agent_pool: list = []
        self._pool_index = 0
        self._pool_lock = Lock()
        self._initialized = False
        
        self._initialize_pool()
        
        logger.info(
            f"Initialized AgentPool with pool_size={self.pool_size}"
        )
    
    def _initialize_pool(self) -> None:
        """Initialize the agent pool."""
        try:
            if self.pool_size == 1:
                # Single shared agent (most common case)
                self._shared_agent = self.agent_factory()
                logger.info("Initialized single shared agent")
            else:
                # Agent pool for higher concurrency
                self._agent_pool = [self.agent_factory() for _ in range(self.pool_size)]
                logger.info(f"Initialized agent pool with {self.pool_size} agents")
            
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize agent pool: {e}", exc_info=True)
            raise
    
    def get_agent(self) -> Any:
        """
        Get an agent from the pool (round-robin for pool, single agent for shared).
        Thread-safe.
        
        Returns:
            Agent instance from the pool
        """
        if not self._initialized:
            raise RuntimeError("Agent pool not initialized")
        
        with self._pool_lock:
            if self.pool_size == 1:
                # Single shared agent
                if self._shared_agent is None:
                    logger.warning("Shared agent is None, recreating...")
                    self._shared_agent = self.agent_factory()
                return self._shared_agent
            else:
                # Round-robin from pool
                if not self._agent_pool:
                    raise RuntimeError("Agent pool is empty")
                agent = self._agent_pool[self._pool_index]
                self._pool_index = (self._pool_index + 1) % len(self._agent_pool)
                return agent
    
    def get_pool_size(self) -> int:
        """Get the current pool size."""
        return self.pool_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent pool."""
        with self._pool_lock:
            return {
                "pool_size": self.pool_size,
                "initialized": self._initialized,
                "has_shared_agent": self._shared_agent is not None,
                "pool_count": len(self._agent_pool)
            }
    
    def reset(self) -> None:
        """
        Reset the agent pool (reinitialize all agents).
        Use with caution - this will recreate all agents.
        """
        logger.warning("Resetting agent pool...")
        with self._pool_lock:
            self._shared_agent = None
            self._agent_pool = []
            self._pool_index = 0
            self._initialized = False
        
        self._initialize_pool()
        logger.info("Agent pool reset complete")


# Global agent pools registry (for different chatbot types)
_agent_pools: Dict[str, AgentPool] = {}
_pools_lock = Lock()


def get_agent_pool(
    chatbot_type: str,
    agent_factory: Optional[Callable] = None,
    pool_size: Optional[int] = None
) -> AgentPool:
    """
    Get or create an agent pool for a specific chatbot type.
    Thread-safe singleton pattern per chatbot type.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        agent_factory: Function that creates a new agent instance (required for first call)
        pool_size: Number of agents in the pool (optional, uses settings if not provided)
        
    Returns:
        AgentPool instance for the specified chatbot type
        
    Raises:
        ValueError: If agent_factory is not provided and pool doesn't exist
    """
    with _pools_lock:
        if chatbot_type not in _agent_pools:
            if agent_factory is None:
                raise ValueError(
                    f"Agent pool for '{chatbot_type}' does not exist and "
                    "agent_factory is required for first initialization"
                )
            
            _agent_pools[chatbot_type] = AgentPool(
                agent_factory=agent_factory,
                pool_size=pool_size
            )
            logger.info(f"Created agent pool for chatbot type: {chatbot_type}")
        
        return _agent_pools[chatbot_type]


def reset_agent_pool(chatbot_type: str) -> None:
    """
    Reset an agent pool for a specific chatbot type.
    
    Args:
        chatbot_type: Type of chatbot to reset
    """
    with _pools_lock:
        if chatbot_type in _agent_pools:
            _agent_pools[chatbot_type].reset()
            logger.info(f"Reset agent pool for chatbot type: {chatbot_type}")
        else:
            logger.warning(f"Agent pool for '{chatbot_type}' does not exist")


def get_all_pool_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all agent pools."""
    with _pools_lock:
        return {
            chatbot_type: pool.get_stats()
            for chatbot_type, pool in _agent_pools.items()
        }

