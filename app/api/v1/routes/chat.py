"""
HR Chatbot API Routes
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logger import logger
from app.chatbot.hr_chatbot import create_hr_chatbot_agent
from app.chatbot.chatbot import chat_with_agent
from app.llm_manager import get_available_models

router = APIRouter()

# Global agent instance (singleton pattern)
_hr_chatbot_agent: Optional[Any] = None


def get_hr_chatbot_agent():
    """
    Get or create the HR chatbot agent instance (singleton).
    
    Returns:
        HR chatbot agent instance
    """
    global _hr_chatbot_agent
    if _hr_chatbot_agent is None:
        try:
            _hr_chatbot_agent = create_hr_chatbot_agent(
                initialize_vector_store=False,  # Already initialized on startup
                verbose=False
            )
            logger.info("HR chatbot agent created for API")
        except Exception as e:
            logger.error(f"Failed to create HR chatbot agent: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize HR chatbot agent: {str(e)}"
            )
    return _hr_chatbot_agent


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question", min_length=1)
    model_name: Optional[str] = Field(None, description="LLM model to use (optional)")
    temperature: Optional[float] = Field(None, description="Temperature for the model (optional)", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response (optional)", gt=0)
    chat_history: Optional[List[ChatMessage]] = Field(None, description="Previous chat messages for context")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot's response")
    model_used: str = Field(..., description="Model that was used for the response")
    message_id: Optional[str] = Field(None, description="Optional message ID for tracking")


class ChatRequestWithConfig(ChatRequest):
    """Extended request model with agent configuration"""
    api_key: Optional[str] = Field(None, description="API key for the model provider (optional)")
    base_url: Optional[str] = Field(None, description="Base URL for the model API (optional, mainly for Ollama)")
    verbose: bool = Field(False, description="Enable verbose logging")


@router.post("/", response_model=ChatResponse, tags=["chat"])
async def chat_with_hr_chatbot(request: ChatRequest):
    """
    Chat with the HR chatbot.
    
    Send a message to the HR chatbot and receive a response based on HR policies and documents.
    
    Args:
        request: Chat request containing the message and optional parameters
        
    Returns:
        ChatResponse with the chatbot's response
        
    Raises:
        HTTPException: If the chatbot fails to respond
    """
    try:
        # Get or create agent
        agent = get_hr_chatbot_agent()
        
        # Convert chat history if provided
        chat_history = None
        if request.chat_history:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        # Get response from chatbot
        response_text = chat_with_agent(
            agent=agent,
            query=request.message,
            chat_history=chat_history
        )
        
        # Get model name from settings (since we're using default agent)
        from app.core.config import settings
        model_used = request.model_name or settings.CHAT_MODEL
        
        return ChatResponse(
            response=response_text,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/custom", response_model=ChatResponse, tags=["chat"])
async def chat_with_custom_agent(request: ChatRequestWithConfig):
    """
    Chat with the HR chatbot using a custom agent configuration.
    
    This endpoint allows you to specify model parameters for a one-time chat.
    A new agent will be created with the specified configuration.
    
    Args:
        request: Chat request with custom agent configuration
        
    Returns:
        ChatResponse with the chatbot's response
        
    Raises:
        HTTPException: If the chatbot fails to respond or configuration is invalid
    """
    try:
        # Create a custom agent with specified configuration
        agent = create_hr_chatbot_agent(
            model_name=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            verbose=request.verbose,
            api_key=request.api_key,
            base_url=request.base_url,
            initialize_vector_store=False  # Already initialized on startup
        )
        
        # Convert chat history if provided
        chat_history = None
        if request.chat_history:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        # Get response from chatbot
        response_text = chat_with_agent(
            agent=agent,
            query=request.message,
            chat_history=chat_history
        )
        
        # Get model name
        from app.core.config import settings
        model_used = request.model_name or settings.CHAT_MODEL
        
        return ChatResponse(
            response=response_text,
            model_used=model_used
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
        # Try to get the agent to verify it's working
        agent = get_hr_chatbot_agent()
        
        return {
            "status": "healthy",
            "service": "hr_chatbot",
            "agent_initialized": agent is not None
        }
    except Exception as e:
        logger.error(f"Chatbot health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "hr_chatbot",
            "error": str(e)
        }

