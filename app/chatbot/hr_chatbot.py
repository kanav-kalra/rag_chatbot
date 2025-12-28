"""
HR Chatbot - Wrapper around generic chatbot with HR-specific prompts and tools
"""
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool

from app.core.config import settings
from app.core.logger import logger
from app.chatbot.chatbot import create_chatbot_agent, chat_with_agent
from app.llm_manager import get_llm, get_available_models
from app.chatbot.prompts import HR_CHATBOT_SYSTEM_PROMPT, AGENT_INSTRUCTIONS
from app.tools.retrieval_tool import retrieve_documents
from app.document_loader.memory_builder import build_memory_from_pdfs
from app.tools.vector_store_manager import set_vector_store


def initialize_hr_chatbot_vector_store():
    """
    Initialize the vector store for HR chatbot on application startup.
    This should be called during application startup to load HR documents into the vector store.
    
    Raises:
        Exception: If vector store initialization fails
    """
    try:
        logger.info("Initializing HR chatbot vector store on startup...")
        vector_store = build_memory_from_pdfs()
        set_vector_store(vector_store)
        logger.info("HR chatbot vector store initialized successfully.")
        logger.info("Vector store set globally for retrieval tools.")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing HR chatbot vector store: {e}", exc_info=True)
        raise


def create_hr_chatbot_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    initialize_vector_store: bool = False
):
    """
    Create an HR chatbot agent with HR-specific prompts and tools.
    
    Args:
        model_name: Name of the LLM model to use (default: from settings)
        temperature: Temperature for the model (default: from settings)
        max_tokens: Maximum tokens for responses (default: from settings)
        tools: List of additional tools for the agent (default: [retrieve_documents])
        verbose: Whether to enable verbose logging (default: False)
        api_key: API key for the model provider (optional)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        initialize_vector_store: Whether to initialize vector store (default: True)
        
    Returns:
        CompiledStateGraph instance (LangChain 1.0+ agent) configured for HR chatbot
        
    Raises:
        ValueError: If model configuration is invalid
        RuntimeError: If agent creation fails
    """
    try:
        # Initialize vector store if requested
        if initialize_vector_store:
            initialize_hr_chatbot_vector_store()
        
        # Set up tools (default to retrieve_documents if not provided)
        if tools is None:
            tools = [retrieve_documents]
        elif retrieve_documents not in tools:
            # Always include retrieve_documents for HR chatbot
            tools = [retrieve_documents] + tools
        
        # Create system prompt for HR chatbot
        system_prompt = HR_CHATBOT_SYSTEM_PROMPT + "\n\n" + AGENT_INSTRUCTIONS
        
        # Create agent using generic chatbot function
        agent = create_chatbot_agent(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(
            f"Created HR chatbot agent with model: {model_name or settings.CHAT_MODEL}"
        )
        return agent
        
    except Exception as e:
        error_msg = f"Error creating HR chatbot agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# Re-export chat_with_agent and get_available_models for convenience
__all__ = [
    "initialize_hr_chatbot_vector_store",
    "create_hr_chatbot_agent",
    "chat_with_agent",
    "get_available_models",
    "get_llm",
]

