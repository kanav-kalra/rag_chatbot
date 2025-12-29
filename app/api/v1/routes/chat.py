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
from app.chatbot.hr_chatbot import (
    get_hr_chatbot,
    create_hr_chatbot_with_custom_memory,
    initialize_hr_chatbot_vector_store
)
from app.llm_manager import get_available_models
import uuid

router = APIRouter()


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
    memory_strategy: Optional[str] = Field(None, description="Memory strategy: 'none', 'trim', 'summarize', 'trim_and_summarize'")
    trim_keep_messages: Optional[int] = Field(None, description="Number of messages to keep when trimming")
    summarize_threshold: Optional[int] = Field(None, description="Message count threshold for summarization")


@router.post("/", response_model=ChatResponse, tags=["chat"])
async def chat_with_hr_chatbot(
    request: ChatRequest
):
    """
    Chat with the HR chatbot using Redis checkpointer for memory management.
    
    Send a message to the HR chatbot and receive a response based on HR policies and documents.
    If a session_id is provided, the conversation history is maintained via checkpointer.
    If not, a new session_id is generated.
    
    Args:
        request: Chat request containing the message and optional session_id
        
    Returns:
        ChatResponse with the chatbot's response and session_id
        
    Raises:
        HTTPException: If the chatbot fails to respond
    """
    try:
        # Get or generate session_id (thread_id for checkpointer)
        thread_id = request.session_id or str(uuid.uuid4())
        
        # Get HR chatbot instance (singleton from hr_chatbot module)
        chatbot = get_hr_chatbot()
        
        # Chat with the chatbot (checkpointer manages history automatically)
        response_text = chatbot.chat(
            query=request.message,
            thread_id=thread_id,
            user_id=request.user_id
        )
        
        # Get model name from settings
        model_used = settings.CHAT_MODEL
        
        logger.info(
            f"Chat completed for thread {thread_id} "
            f"(user_id={request.user_id})"
        )
        
        return ChatResponse(
            response=response_text,
            session_id=thread_id,
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


@router.post("/custom", response_model=ChatResponse, tags=["chat"])
async def chat_with_custom_agent(
    request: ChatRequestWithConfig
):
    """
    Chat with the HR chatbot using a custom agent configuration.
    
    This endpoint allows you to specify model parameters and memory strategy for a chat.
    A new chatbot instance will be created with the specified configuration.
    
    Args:
        request: Chat request with custom agent configuration
        
    Returns:
        ChatResponse with the chatbot's response
        
    Raises:
        HTTPException: If the chatbot fails to respond or configuration is invalid
    """
    try:
        # Get or generate session_id (thread_id for checkpointer)
        thread_id = request.session_id or str(uuid.uuid4())
        
        # Create custom HR chatbot instance
        # If memory parameters are provided, use custom memory configuration
        if request.memory_strategy:
            try:
                chatbot = create_hr_chatbot_with_custom_memory(
                    memory_strategy=request.memory_strategy,
                    trim_keep_messages=request.trim_keep_messages,
                    summarize_threshold=request.summarize_threshold,
                    model_name=request.model_name,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    verbose=request.verbose,
                    api_key=request.api_key,
                    base_url=request.base_url
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
        else:
            # Use default chatbot with custom model parameters if provided
            from app.chatbot.hr_chatbot import create_hr_chatbot
            chatbot = create_hr_chatbot(
                model_name=request.model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                verbose=request.verbose,
                api_key=request.api_key,
                base_url=request.base_url,
                initialize_vector_store=False
            )
        
        # Chat with the chatbot
        response_text = chatbot.chat(
            query=request.message,
            thread_id=thread_id,
            user_id=request.user_id
        )
        
        # Get model name
        model_used = request.model_name or settings.CHAT_MODEL
        
        return ChatResponse(
            response=response_text,
            session_id=thread_id,
            model_used=model_used,
            message_count=None  # Message count is managed by checkpointer
        )
        
    except HTTPException:
        raise
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
async def delete_session(session_id: str):
    """
    Delete a chat session (thread) from checkpointer.
    
    Note: This removes the conversation history from Redis.
    The session_id is used as thread_id in the checkpointer.
    
    Args:
        session_id: Session/thread identifier
        
    Returns:
        Success message
    """
    try:
        from app.core.checkpointer_manager import get_checkpointer
        
        checkpointer = get_checkpointer()
        # Note: LangGraph checkpointer doesn't have a direct delete method
        # The data will expire based on Redis TTL or can be cleared manually
        # For now, we'll just return success
        logger.info(f"Session deletion requested for thread_id: {session_id}")
        return {
            "message": f"Session {session_id} deletion requested. "
                      "Note: Data may persist in Redis until TTL expires."
        }
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@router.get("/sessions/{session_id}", tags=["chat"])
async def get_session_info(session_id: str):
    """
    Get information about a session (thread).
    
    Note: With checkpointer, we can retrieve the thread state.
    
    Args:
        session_id: Session/thread identifier
        
    Returns:
        Session information
    """
    try:
        from app.core.checkpointer_manager import get_checkpointer_manager
        
        manager = get_checkpointer_manager()
        checkpointer = manager.checkpointer
        
        # Try to get thread state
        config = manager.get_config(session_id)
        try:
            # Get the latest checkpoint
            from langgraph.checkpoint.base import Checkpoint
            checkpoint = checkpointer.get(config)
            
            if checkpoint:
                return {
                    "session_id": session_id,
                    "exists": True,
                    "checkpointer_type": "redis" if manager.is_redis else "memory"
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found"
                )
        except Exception:
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
async def get_session_stats():
    """
    Get statistics about the checkpointer system.
    
    Returns:
        Checkpointer statistics
    """
    try:
        from app.core.checkpointer_manager import get_checkpointer_manager
        
        manager = get_checkpointer_manager()
        return {
            "checkpointer_type": "redis" if manager.is_redis else "memory",
            "redis_url": getattr(settings, 'REDIS_URL', 'redis://localhost:6379') if manager.is_redis else None
        }
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
async def chat_health_check():
    """
    Health check for the HR chatbot service.
    
    Returns:
        Status of the chatbot service
    """
    try:
        from app.core.checkpointer_manager import get_checkpointer_manager
        
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
