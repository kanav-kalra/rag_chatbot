"""
LLM Manager Dependencies for FastAPI
Pure dependency injection - container is stored in app state.
"""
import sys
from pathlib import Path
from typing import Annotated
from fastapi import Depends, Request

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.llm.manager import LLMManager
from src.shared.dependencies.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    """
    FastAPI dependency that provides application container from app state.
    
    The container is stored in app.state.container during startup.
    
    Usage:
        @router.post("/chat")
        async def chat(
            container: ApplicationContainer = Depends(get_container)
        ):
            llm_manager = container.get_llm_manager()
    """
    if not hasattr(request.app.state, 'container'):
        raise RuntimeError(
            "Application container not found in app state. "
            "Ensure container is initialized in lifespan startup."
        )
    return request.app.state.container


def get_llm_manager(container: Annotated[ApplicationContainer, Depends(get_container)]) -> LLMManager:
    """
    FastAPI dependency that provides LLM manager.
    
    Usage:
        @router.post("/chat")
        async def chat(
            llm_manager: LLMManager = Depends(get_llm_manager)
        ):
            llm = llm_manager.get_llm(...)
    
    Returns:
        LLMManager instance from application container
    """
    return container.get_llm_manager()


def get_llm_manager_by_id_factory(instance_id: str):
    """
    Factory function for creating FastAPI dependency that gets LLM manager by ID.
    
    Usage:
        @router.post("/chat")
        async def chat(
            llm_manager: LLMManager = Depends(get_llm_manager_by_id_factory("custom"))
        ):
            ...
    
    Args:
        instance_id: Instance identifier
    
    Returns:
        FastAPI dependency function
    """
    def _get_llm_manager_by_id(
        container: Annotated[ApplicationContainer, Depends(get_container)]
    ) -> LLMManager:
        registry = container.get_llm_registry()
        return registry.get(instance_id)
    
    return _get_llm_manager_by_id
