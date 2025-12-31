"""
LangSmith Configuration Helper
Provides utilities for initializing and managing LangSmith tracing
"""
import os
from typing import Optional

from app.core.config import settings
from app.core.logging import logger


def initialize_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    endpoint: Optional[str] = None,
    workspace_id: Optional[str] = None,
    force_enable: bool = False
) -> bool:
    """
    Initialize LangSmith tracing by setting environment variables.
    
    Args:
        api_key: LangSmith API key (default: from settings)
        project: LangSmith project name (default: from settings)
        endpoint: LangSmith endpoint URL (default: from settings)
        workspace_id: LangSmith workspace ID (optional, only needed for org-scoped API keys)
        force_enable: Force enable tracing even if LANGCHAIN_TRACING_V2 is False in settings
        
    Returns:
        True if LangSmith was successfully initialized, False otherwise
    """
    try:
        # Check if tracing should be enabled
        if not force_enable and not settings.LANGCHAIN_TRACING_V2:
            logger.debug("LangSmith tracing is disabled in settings")
            return False
        
        # Set environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # Use provided values or fall back to settings
        # Support both LANGSMITH_API_KEY (preferred) and LANGCHAIN_API_KEY (legacy)
        api_key = api_key or settings.LANGCHAIN_API_KEY
        project = project or settings.LANGCHAIN_PROJECT
        endpoint = endpoint or settings.LANGCHAIN_ENDPOINT
        workspace_id = workspace_id or getattr(settings, 'LANGSMITH_WORKSPACE_ID', '')
        
        if api_key:
            # Set both for maximum compatibility
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_API_KEY"] = api_key
        else:
            logger.warning("LangSmith API key not provided. Tracing may not work.")
            return False
        
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        
        if endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        
        # Only set workspace_id if provided (required for org-scoped API keys)
        if workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
            logger.info(
                f"LangSmith tracing initialized - Project: {project}, "
                f"Workspace: {workspace_id}, Endpoint: {endpoint}"
            )
        else:
            logger.info(
                f"LangSmith tracing initialized - Project: {project}, "
                f"Endpoint: {endpoint} (no workspace_id - using personal API key)"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}", exc_info=True)
        return False


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is currently enabled.
    
    Returns:
        True if LangSmith tracing is enabled, False otherwise
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_langsmith_project() -> Optional[str]:
    """
    Get the current LangSmith project name.
    
    Returns:
        Project name if set, None otherwise
    """
    return os.getenv("LANGCHAIN_PROJECT") or settings.LANGCHAIN_PROJECT

