"""
HR Chatbot - OOP implementation
Wrapper around generic chatbot with HR-specific prompts and tools
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
from app.core.logging import logger
from app.services.chatbot.chatbot_service import ChatbotAgent
from app.infra.llm.llm_manager import get_llm
from app.services.chatbot.prompts import HR_CHATBOT_SYSTEM_PROMPT, AGENT_INSTRUCTIONS
from app.services.retrieval.retrieval_service import retrieve_documents
from ingestion.embedder import build_memory_from_pdfs
from vectorstore.client import set_vector_store
from app.core.memory_config import get_memory_config
from app.services.chatbot.agent_pool import get_agent_pool


class HRChatbot(ChatbotAgent):
    """
    HR-specific chatbot that extends the generic ChatbotAgent.
    Includes HR-specific prompts and retrieval tools.
    """
    
    def _get_chatbot_type(self) -> str:
        """Get the chatbot type identifier."""
        return "hr"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        base_url: Optional[str] = None
    ):
        """
        Initialize HR chatbot.
        
        Args:
            model_name: Name of the LLM model to use (default: from settings)
            temperature: Temperature for the model (default: from settings)
            max_tokens: Maximum tokens for responses (default: from settings)
            tools: List of additional tools for the agent (default: [retrieve_documents])
            verbose: Whether to enable verbose logging (default: False)
            base_url: Base URL for the model API (optional, mainly for Ollama)
        """
        # Set up tools (default to retrieve_documents if not provided)
        if tools is None:
            tools = [retrieve_documents]
        elif retrieve_documents not in tools:
            # Always include retrieve_documents for HR chatbot
            tools = [retrieve_documents] + tools
        
        # Create system prompt for HR chatbot
        system_prompt = HR_CHATBOT_SYSTEM_PROMPT + "\n\n" + AGENT_INSTRUCTIONS
        
        # Get default HR memory config (defaults to TRIM strategy, can be overridden via config/env)
        memory_config = get_memory_config("hr", use_settings=True)
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            base_url=base_url,
            memory_config=memory_config,
            chatbot_type="hr"
        )
        
        logger.info(
            f"Initialized HRChatbot with model: {self.model_name}"
        )
    
    @staticmethod
    def _initialize_vector_store() -> None:
        """
        Initialize the vector store for HR chatbot.
        This should be called during application startup.
        
        Raises:
            Exception: If vector store initialization fails
        """
        try:
            logger.info("Initializing HR chatbot vector store...")
            vector_store = build_memory_from_pdfs()
            set_vector_store(vector_store)
            logger.info("HR chatbot vector store initialized successfully.")
            logger.info("Vector store set globally for retrieval tools.")
        except Exception as e:
            logger.error(f"Error initializing HR chatbot vector store: {e}", exc_info=True)
            raise


def _initialize_hr_chatbot_vector_store():
    """
    Initialize the vector store for HR chatbot on application startup.
    This should be called during application startup to load HR documents into the vector store.
    Private function - use HRChatbot._initialize_vector_store() directly if needed.
    
    Raises:
        Exception: If vector store initialization fails
    """
    HRChatbot._initialize_vector_store()


def _create_hr_chatbot(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    base_url: Optional[str] = None
) -> HRChatbot:
    """
    Create an HR chatbot instance (private factory function).
    Agents should be fetched from agent pool, not created directly.
    
    Args:
        model_name: Name of the LLM model to use (default: from settings)
        temperature: Temperature for the model (default: from settings)
        max_tokens: Maximum tokens for responses (default: from settings)
        tools: List of additional tools for the agent (default: [retrieve_documents])
        verbose: Whether to enable verbose logging (default: False)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        
    Returns:
        HRChatbot instance
    """
    return HRChatbot(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        verbose=verbose,
        base_url=base_url
    )


def _get_default_hr_chatbot() -> HRChatbot:
    """
    Get or create a default HR chatbot instance with configuration from settings.
    Private function - use get_hr_chatbot() to get from agent pool instead.
    
    Returns:
        HRChatbot instance configured with settings
    """
    # Memory config is automatically loaded from settings via get_memory_config("hr")
    # which uses get_memory_config_from_settings internally
    return _create_hr_chatbot(
        verbose=False
    )


def get_hr_chatbot() -> HRChatbot:
    """
    Get an HR chatbot instance from the agent pool.
    Uses default configuration from settings.
    
    This is the recommended way to get the HR chatbot for API use.
    The agent pool ensures efficient resource usage across multiple requests.
    Thread-safe for concurrent API requests.
    
    Returns:
        HRChatbot instance from agent pool
        
    Raises:
        RuntimeError: If chatbot initialization fails
    """
    try:
        # Get agent pool for HR chatbot type
        # The pool will create instances using the factory if needed
        agent_pool = get_agent_pool(
            chatbot_type="hr",
            agent_factory=_get_default_hr_chatbot
        )
        
        # Get agent from pool
        chatbot = agent_pool.get_agent()
        return chatbot
    except Exception as e:
        logger.error(f"Failed to get HR chatbot from pool: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get HR chatbot from pool: {str(e)}") from e


# Re-export for convenience
__all__ = [
    "HRChatbot",
    "get_hr_chatbot",
    "_initialize_hr_chatbot_vector_store",  # Private, but needed for main.py
]
