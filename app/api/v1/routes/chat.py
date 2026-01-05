"""
HR Chatbot API Routes - Production-ready with session management
Uses FastAPI dependency injection for session management.
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.core.config import settings
from app.services.chatbot.hr_chatbot import get_hr_chatbot
from app.services.session.session_manager import ChatbotSession, ChatbotSessionManager
from app.core.session_dependencies import (
    get_session_from_headers,
    get_session_from_path,
    get_session_manager_dependency
)

router = APIRouter()


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint - only contains the message"""
    message: str = Field(..., description="User's message/question", min_length=1)


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    model_used: str = Field(..., description="Model that was used for the response")
    message_count: Optional[int] = Field(None, description="Number of messages in this session")


@router.post("/", response_model=ChatResponse, tags=["chat"])
async def chat_with_hr_chatbot(
    request: ChatRequest,
    session: ChatbotSession = Depends(get_session_from_headers)
):
    """
    Chat with the HR chatbot using Redis checkpointer for memory management.
    
    Send a message to the HR chatbot and receive a response based on HR policies and documents.
    Session management is handled automatically via dependency injection from headers/cookies.
    If X-Session-ID header or session_id cookie is provided, the conversation history is maintained.
    If not, a new session_id is generated.
    
    Headers (preferred):
        - X-Session-ID: Session identifier
        - X-User-ID: User identifier
    
    Cookies (fallback):
        - session_id: Session identifier
        - user_id: User identifier
    
    Args:
        request: Chat request containing only the message
        session: ChatbotSession automatically injected via dependency from headers/cookies
        
    Returns:
        ChatResponse with the chatbot's response and session_id
        
    Raises:
        HTTPException: If the chatbot fails to respond
    """
    try:
        # Get HR chatbot instance from agent pool at runtime
        chatbot = get_hr_chatbot()
        
        # Chat with the chatbot (checkpointer manages history automatically)
        # Use session_id as thread_id for checkpointer
        response_text = chatbot.chat(
            query=request.message,
            thread_id=session.session_id,
            user_id=session.user_id
        )
        
        # Get model name from chatbot instance (as per hr_chatbot.py pattern)
        model_used = chatbot.model_name
        
        logger.info(
            f"Chat completed for session {session.session_id} "
            f"(user_id={session.user_id})"
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session.session_id,
            model_used=model_used,
            message_count=None  # Message count is managed by checkpointer
        )
        
    except RuntimeError as e:
        # Handle initialization errors from get_hr_chatbot()
        logger.error(f"HR chatbot initialization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize HR chatbot: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.delete("/sessions/{session_id}", tags=["chat"])
async def delete_session(
    session_id: str,
    session: ChatbotSession = Depends(get_session_from_path),
    session_manager: ChatbotSessionManager = Depends(get_session_manager_dependency)
):
    """
    Delete a chat session from session manager.
    
    This removes the session from the session manager. Note that the conversation
    history in the checkpointer may persist until Redis TTL expires.
    
    Args:
        session_id: Session identifier from path
        session: ChatbotSession automatically injected via dependency (validates session exists)
        session_manager: ChatbotSessionManager automatically injected via dependency
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 404 if session not found, 500 on error
    """
    try:
        deleted = session_manager.delete_session(session_id)
        
        if deleted:
            logger.info(f"Session {session_id} deleted from session manager")
            return {
                "message": f"Session {session_id} deleted successfully.",
                "note": "Checkpointer data may persist in Redis until TTL expires."
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@router.get("/sessions/{session_id}", tags=["chat"])
async def get_session_info(
    session_id: str,
    session: ChatbotSession = Depends(get_session_from_path)
):
    """
    Get information about a session.
    
    Returns session metadata from the session manager and checkpointer status.
    
    Args:
        session_id: Session identifier from path
        session: ChatbotSession automatically injected via dependency (404 if not found)
        
    Returns:
        Session information including metadata and checkpointer status
        
    Raises:
        HTTPException: 404 if session not found, 500 on error
    """
    try:
        # Check checkpointer status
        from app.infra.checkpointing.checkpoint_manager import get_checkpointer_manager
        
        checkpointer_manager = get_checkpointer_manager()
        checkpointer = checkpointer_manager.checkpointer
        
        # Try to get thread state from checkpointer
        config = checkpointer_manager.get_config(session.session_id)
        checkpoint = checkpointer.get(config)
        
        session_info = session.to_dict()
        session_info.update({
            "exists": True,
            "checkpointer_has_data": checkpoint is not None,
            "checkpointer_type": "redis" if checkpointer_manager.is_redis else "memory"
        })
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session info: {str(e)}"
        )


@router.get("/sessions/stats", tags=["chat"])
async def get_session_stats(
    session_manager: ChatbotSessionManager = Depends(get_session_manager_dependency)
):
    """
    Get statistics about sessions and checkpointer system.
    
    Args:
        session_manager: ChatbotSessionManager automatically injected via dependency
        
    Returns:
        Combined statistics from session manager and checkpointer
    """
    try:
        session_stats = session_manager.get_session_stats()
        
        from app.infra.checkpointing.checkpoint_manager import get_checkpointer_manager
        from app.services.chatbot.agent_pool import get_all_pool_stats
        
        checkpointer_manager = get_checkpointer_manager()
        agent_pool_stats = get_all_pool_stats()
        
        return {
            **session_stats,
            "checkpointer": {
                "type": "redis" if checkpointer_manager.is_redis else "memory",
                "redis_url": getattr(settings, 'REDIS_URL', 'redis://localhost:6379') if checkpointer_manager.is_redis else None
            },
            "agent_pools": agent_pool_stats
        }
    except Exception as e:
        logger.error(f"Error getting session stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session stats: {str(e)}"
        )


@router.get("/health", tags=["chat"])
async def chat_health_check():
    """
    Health check for the HR chatbot service.
    
    Returns:
        Status of the chatbot service
    """
    try:
        from app.infra.checkpointing.checkpoint_manager import get_checkpointer_manager
        
        manager = get_checkpointer_manager()
        chatbot = get_hr_chatbot()  # Uses singleton from hr_chatbot module
        
        return {
            "status": "healthy",
            "service": "hr_chatbot",
            "checkpointer_type": "redis" if manager.is_redis else "memory",
            "model": chatbot.model_name
        }
    except RuntimeError as e:
        # Handle initialization errors
        logger.error(f"Chatbot health check failed - initialization error: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "hr_chatbot",
            "error": f"Initialization failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Chatbot health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "hr_chatbot",
            "error": str(e)
        }
