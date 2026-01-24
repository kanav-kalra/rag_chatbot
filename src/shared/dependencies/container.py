"""
Application Dependency Container
Central place to wire all dependencies for the application.
"""
import sys
from pathlib import Path
from typing import Optional
from datetime import timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.llm.manager import (
    LLMManager,
    DefaultModelConfigRepository,
    SettingsAPIKeyProvider,
    ModelProviderFactory,
    UseCaseConfig,
    LLMCache,
    LLMManagerRegistry
)
from src.domain.session.manager import ChatbotSessionManager
from src.shared.config.settings import settings
from src.shared.config.logging import logger


class ApplicationContainer:
    """
    Container for all application dependencies.
    Wire dependencies here at application startup.
    Pure dependency injection - no service locator pattern.
    """
    
    def __init__(self):
        self._llm_manager: Optional[LLMManager] = None
        self._session_manager: Optional[ChatbotSessionManager] = None
        self._llm_registry: Optional[LLMManagerRegistry] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all dependencies."""
        if self._initialized:
            logger.warning("Application container already initialized")
            return
        
        # Create registry first
        self._llm_registry = LLMManagerRegistry()
        
        self._initialize_llm_manager()
        self._initialize_session_manager()
        self._initialized = True
        logger.info("Application container initialized")
    
    def _initialize_llm_manager(self) -> None:
        """Initialize LLM manager with default dependencies."""
        if self._llm_registry is None:
            raise RuntimeError("Registry must be created before initializing LLM manager")
        
        self._llm_manager = self._llm_registry.create_and_register(
            instance_id="default",
            config_repository=DefaultModelConfigRepository(),
            api_key_provider=SettingsAPIKeyProvider(),
            provider_factory=ModelProviderFactory(),
            use_case_config=UseCaseConfig(),
            cache=LLMCache()
        )
        logger.info("LLM manager initialized and registered")
    
    def _initialize_session_manager(self) -> None:
        """Initialize session manager."""
        self._session_manager = ChatbotSessionManager(
            session_timeout=timedelta(hours=settings.SESSION_TIMEOUT_HOURS),
            max_sessions=settings.MAX_CONCURRENT_SESSIONS
        )
        logger.info("Session manager initialized")
    
    def get_llm_manager(self) -> LLMManager:
        """Get the default LLM manager."""
        if self._llm_manager is None:
            raise RuntimeError(
                "Application container not initialized. Call initialize() first."
            )
        return self._llm_manager
    
    def get_session_manager(self) -> ChatbotSessionManager:
        """Get the session manager."""
        if self._session_manager is None:
            raise RuntimeError(
                "Application container not initialized. Call initialize() first."
            )
        return self._session_manager
    
    def get_llm_registry(self) -> LLMManagerRegistry:
        """Get the LLM manager registry."""
        if self._llm_registry is None:
            raise RuntimeError(
                "Application container not initialized. Call initialize() first."
            )
        return self._llm_registry
    
    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized
