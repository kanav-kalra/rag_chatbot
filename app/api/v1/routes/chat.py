"""
HR Chatbot API Routes - Production-ready with session management
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from app.core.logger import logger
from app.core.config import settings
from app.chatbot.hr_chatbot import create_hr_chatbot, initialize_hr_chatbot_vector_store
from app.chatbot.session_manager import ChatbotSessionManager, ChatbotSession
from app.llm_manager import get_available_models

router = APIRouter()

# Global session manager instance (singleton)
_session_manager: Optional[ChatbotSessionManager] = None


def get_session_manager() -> ChatbotSessionManager:
    """
    Get or create the session manager instance (singleton).
    
    Returns:
        ChatbotSessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        try:
            # Factory function to create HR chatbot instances
            def agent_factory():
                return create_hr_chatbot(
                    initialize_vector_store=False,  # Already initialized on startup
                    verbose=False
                )
            
            _session_manager = ChatbotSessionManager(
                agent_factory=agent_factory,
                session_timeout=timedelta(hours=24),  # Sessions expire after 24 hours
                max_sessions=getattr(settings, 'MAX_CONCURRENT_SESSIONS', None),
                agent_pool_size=getattr(settings, 'AGENT_POOL_SIZE', 1)
            )
            logger.info("ChatbotSessionManager initialized for API")
        except Exception as e:
            logger.error(f"Failed to create session manager: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize session manager: {str(e)}"
            )
    return _session_manager


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation context")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    model_used: str = Field(..., description="Model that was used for the response")
    message_count: Optional[int] = Field(None, description="Number of messages in this session")


class ChatRequestWithConfig(ChatRequest):
    """Extended request model with agent configuration"""
    model_name: Optional[str] = Field(None, description="LLM model to use (optional)")
    temperature: Optional[float] = Field(None, description="Temperature for the model (optional)", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response (optional)", gt=0)
    api_key: Optional[str] = Field(None, description="API key for the model provider (optional)")
    base_url: Optional[str] = Field(None, description="Base URL for the model API (optional, mainly for Ollama)")
    verbose: bool = Field(False, description="Enable verbose logging")


@router.post("/", response_model=ChatResponse, tags=["chat"])
async def chat_with_hr_chatbot(
    request: ChatRequest,
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Chat with the HR chatbot with automatic session management.
    
    Send a message to the HR chatbot and receive a response based on HR policies and documents.
    If a session_id is provided, the conversation history is maintained. If not, a new session is created.
    
    Args:
        request: Chat request containing the message and optional session_id
        session_manager: Session manager dependency
        
    Returns:
        ChatResponse with the chatbot's response and session_id
        
    Raises:
        HTTPException: If the chatbot fails to respond
    """
    try:
        # Get or create session
        session = session_manager.get_or_create_session(
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Chat with the session (maintains history automatically)
        response_text = session.agent.chat(
            query=request.message,
            chat_history=session.get_chat_history(format_for_agent=True)[:-1]  # Exclude current message
        )
        
        # Add messages to session history
        session.add_message("user", request.message)
        session.add_message("assistant", response_text)
        
        # Get model name from settings
        model_used = settings.CHAT_MODEL
        
        logger.info(
            f"Chat completed for session {session.session_id} "
            f"(message_count={session.message_count})"
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session.session_id,
            model_used=model_used,
            message_count=session.message_count
        )
        
    except RuntimeError as e:
        # Handle max sessions error
        if "Maximum number of sessions" in str(e):
            logger.warning(f"Max sessions reached: {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable: maximum number of concurrent sessions reached"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/custom", response_model=ChatResponse, tags=["chat"])
async def chat_with_custom_agent(
    request: ChatRequestWithConfig,
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Chat with the HR chatbot using a custom agent configuration.
    
    This endpoint allows you to specify model parameters for a one-time chat.
    A new agent will be created with the specified configuration.
    
    Args:
        request: Chat request with custom agent configuration
        session_manager: Session manager dependency
        
    Returns:
        ChatResponse with the chatbot's response
        
    Raises:
        HTTPException: If the chatbot fails to respond or configuration is invalid
    """
    try:
        # Create a custom agent factory
        def custom_agent_factory():
            return create_hr_chatbot(
                model_name=request.model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                verbose=request.verbose,
                api_key=request.api_key,
                base_url=request.base_url,
                initialize_vector_store=False
            )
        
        # Get or create session with custom agent
        if request.session_id:
            # For existing sessions, we'd need to update the agent, which is complex
            # For now, create a new session with custom config
            session = session_manager.create_session(
                session_id=None,  # Create new session for custom config
                user_id=request.user_id
            )
            # Replace agent with custom one
            session.agent = custom_agent_factory()
        else:
            # Create new session with custom agent
            session = session_manager.create_session(
                user_id=request.user_id
            )
            session.agent = custom_agent_factory()
        
        # Chat with the session
        response_text = session.agent.chat(
            query=request.message,
            chat_history=session.get_chat_history(format_for_agent=True)[:-1]
        )
        
        # Add messages to session history
        session.add_message("user", request.message)
        session.add_message("assistant", response_text)
        
        # Get model name
        model_used = request.model_name or settings.CHAT_MODEL
        
        return ChatResponse(
            response=response_text,
            session_id=session.session_id,
            model_used=model_used,
            message_count=session.message_count
        )
        
    except ValueError as e:
        logger.error(f"Invalid configuration in chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid configuration: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in custom chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.delete("/sessions/{session_id}", tags=["chat"])
async def delete_session(
    session_id: str,
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Delete a chat session.
    
    Args:
        session_id: Session identifier
        session_manager: Session manager dependency
        
    Returns:
        Success message
    """
    try:
        deleted = session_manager.delete_session(session_id)
        if deleted:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@router.get("/sessions/{session_id}", tags=["chat"])
async def get_session_info(
    session_id: str,
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Get information about a session.
    
    Args:
        session_id: Session identifier
        session_manager: Session manager dependency
        
    Returns:
        Session information
    """
    try:
        session = session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
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
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Get statistics about active sessions.
    
    Args:
        session_manager: Session manager dependency
        
    Returns:
        Session statistics
    """
    try:
        return session_manager.get_session_stats()
    except Exception as e:
        logger.error(f"Error getting session stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting session stats: {str(e)}"
        )


@router.get("/models", response_model=List[str], tags=["chat"])
async def get_available_models():
    """
    Get list of available LLM models.
    
    Returns:
        List of available model names
    """
    try:
        models = get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error getting available models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available models: {str(e)}"
        )


@router.get("/health", tags=["chat"])
async def chat_health_check(
    session_manager: ChatbotSessionManager = Depends(get_session_manager)
):
    """
    Health check for the HR chatbot service.
    
    Returns:
        Status of the chatbot service
    """
    try:
        stats = session_manager.get_session_stats()
        return {
            "status": "healthy",
            "service": "hr_chatbot",
            "active_sessions": stats["total_sessions"],
            "total_messages": stats["total_messages"]
        }
    except Exception as e:
        logger.error(f"Chatbot health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "hr_chatbot",
            "error": str(e)
        }
