"""
Session Management Dependencies for FastAPI
Provides dependency injection for session management to reduce boilerplate.
"""
import sys
from pathlib import Path
from typing import Optional
from datetime import timedelta
from fastapi import Depends, HTTPException

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logger import logger
from app.core.config import settings
from app.chatbot.session_manager import ChatbotSession, ChatbotSessionManager


# Global session manager singleton
_session_manager: Optional[ChatbotSessionManager] = None


def get_session_manager() -> ChatbotSessionManager:
    """
    Get or create the global session manager singleton.
    Thread-safe singleton pattern.
    
    Returns:
        ChatbotSessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = ChatbotSessionManager(
            session_timeout=timedelta(hours=settings.SESSION_TIMEOUT_HOURS),
            max_sessions=settings.MAX_CONCURRENT_SESSIONS
        )
        logger.info("Session manager initialized")
    
    return _session_manager


def get_or_create_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ChatbotSession:
    """
    FastAPI dependency that gets or creates a session.
    
    Usage:
        @router.post("/chat")
        async def chat(
            request: ChatRequest,
            session: ChatbotSession = Depends(get_or_create_session)
        ):
            # session is automatically available
            ...
    
    Args:
        session_id: Optional session ID
        user_id: Optional user ID
        
    Returns:
        ChatbotSession instance
        
    Raises:
        HTTPException: If session creation fails
    """
    try:
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(
            session_id=session_id,
            user_id=user_id
        )
        session.update_activity()
        return session
    except RuntimeError as e:
        logger.error(f"Failed to get or create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Session management error: {str(e)}"
        )


def get_session(session_id: str) -> ChatbotSession:
    """
    FastAPI dependency that gets an existing session.
    Raises 404 if session doesn't exist.
    
    Usage:
        @router.get("/sessions/{session_id}")
        async def get_session_info(
            session: ChatbotSession = Depends(get_session)
        ):
            # session is automatically available or 404 raised
            ...
    
    Args:
        session_id: Session ID from path parameter
        
    Returns:
        ChatbotSession instance
        
    Raises:
        HTTPException: 404 if session not found
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return session


def get_session_manager_dependency() -> ChatbotSessionManager:
    """
    FastAPI dependency that returns the session manager.
    
    Usage:
        @router.get("/sessions/stats")
        async def get_stats(
            manager: ChatbotSessionManager = Depends(get_session_manager_dependency)
        ):
            return manager.get_session_stats()
    
    Returns:
        ChatbotSessionManager instance
    """
    return get_session_manager()

