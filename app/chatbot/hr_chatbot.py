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
from app.core.logger import logger
from app.chatbot.chatbot import ChatbotAgent, chat_with_agent
from app.llm_manager import get_llm, get_available_models
from app.chatbot.prompts import HR_CHATBOT_SYSTEM_PROMPT, AGENT_INSTRUCTIONS
from app.tools.retrieval_tool import retrieve_documents
from app.document_loader.memory_builder import build_memory_from_pdfs
from app.tools.vector_store_manager import set_vector_store
from app.core.memory_config import MemoryConfig, MemoryStrategy, get_memory_config


class HRChatbot(ChatbotAgent):
    """
    HR-specific chatbot that extends the generic ChatbotAgent.
    Includes HR-specific prompts and retrieval tools.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        initialize_vector_store: bool = False,
        memory_config: Optional[MemoryConfig] = None,
        memory_strategy: Optional[str] = None,
        trim_keep_messages: Optional[int] = None,
        summarize_threshold: Optional[int] = None
    ):
        """
        Initialize HR chatbot.
        
        Args:
            model_name: Name of the LLM model to use (default: from settings)
            temperature: Temperature for the model (default: from settings)
            max_tokens: Maximum tokens for responses (default: from settings)
            tools: List of additional tools for the agent (default: [retrieve_documents])
            verbose: Whether to enable verbose logging (default: False)
            api_key: API key for the model provider (optional)
            base_url: Base URL for the model API (optional, mainly for Ollama)
            initialize_vector_store: Whether to initialize vector store (default: False)
            memory_config: Memory configuration object (optional, overrides other memory params)
            memory_strategy: Memory strategy - "none", "trim", "summarize", "trim_and_summarize" (optional)
            trim_keep_messages: Number of messages to keep when trimming (optional)
            summarize_threshold: Message count threshold for summarization (optional)
        """
        # Initialize vector store if requested
        if initialize_vector_store:
            self._initialize_vector_store()
        
        # Set up tools (default to retrieve_documents if not provided)
        if tools is None:
            tools = [retrieve_documents]
        elif retrieve_documents not in tools:
            # Always include retrieve_documents for HR chatbot
            tools = [retrieve_documents] + tools
        
        # Create system prompt for HR chatbot
        system_prompt = HR_CHATBOT_SYSTEM_PROMPT + "\n\n" + AGENT_INSTRUCTIONS
        
        # Configure memory management
        if memory_config is None:
            # Get default HR memory config
            memory_config = get_memory_config("hr")
            
            # Override with provided parameters if any
            if memory_strategy:
                try:
                    memory_config.strategy = MemoryStrategy(memory_strategy)
                except ValueError:
                    logger.warning(f"Invalid memory_strategy: {memory_strategy}, using default")
            if trim_keep_messages is not None:
                memory_config.trim_keep_messages = trim_keep_messages
            if summarize_threshold is not None:
                memory_config.summarize_threshold = summarize_threshold
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            api_key=api_key,
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


# Factory functions for backward compatibility and convenience
def initialize_hr_chatbot_vector_store():
    """
    Initialize the vector store for HR chatbot on application startup.
    This should be called during application startup to load HR documents into the vector store.
    
    Raises:
        Exception: If vector store initialization fails
    """
    HRChatbot._initialize_vector_store()


def create_hr_chatbot_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    initialize_vector_store: bool = False,
    memory_config: Optional[MemoryConfig] = None,
    memory_strategy: Optional[str] = None,
    trim_keep_messages: Optional[int] = None,
    summarize_threshold: Optional[int] = None
):
    """
    Create an HR chatbot agent (factory function for backward compatibility).
    
    Args:
        model_name: Name of the LLM model to use (default: from settings)
        temperature: Temperature for the model (default: from settings)
        max_tokens: Maximum tokens for responses (default: from settings)
        tools: List of additional tools for the agent (default: [retrieve_documents])
        verbose: Whether to enable verbose logging (default: False)
        api_key: API key for the model provider (optional)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        initialize_vector_store: Whether to initialize vector store (default: False)
        
    Returns:
        CompiledStateGraph instance (LangChain 1.0+ agent) configured for HR chatbot
        
    Raises:
        ValueError: If model configuration is invalid
        RuntimeError: If agent creation fails
    """
    chatbot = HRChatbot(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url,
        initialize_vector_store=initialize_vector_store,
        memory_config=memory_config,
        memory_strategy=memory_strategy,
        trim_keep_messages=trim_keep_messages,
        summarize_threshold=summarize_threshold
    )
    return chatbot.agent  # Return the underlying agent for backward compatibility


def create_hr_chatbot(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    initialize_vector_store: bool = False,
    memory_config: Optional[MemoryConfig] = None,
    memory_strategy: Optional[str] = None,
    trim_keep_messages: Optional[int] = None,
    summarize_threshold: Optional[int] = None
) -> HRChatbot:
    """
    Create an HR chatbot instance (new OOP interface).
    
    Args:
        model_name: Name of the LLM model to use (default: from settings)
        temperature: Temperature for the model (default: from settings)
        max_tokens: Maximum tokens for responses (default: from settings)
        tools: List of additional tools for the agent (default: [retrieve_documents])
        verbose: Whether to enable verbose logging (default: False)
        api_key: API key for the model provider (optional)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        initialize_vector_store: Whether to initialize vector store (default: False)
        
    Returns:
        HRChatbot instance
    """
    return HRChatbot(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url,
        initialize_vector_store=initialize_vector_store,
        memory_config=memory_config,
        memory_strategy=memory_strategy,
        trim_keep_messages=trim_keep_messages,
        summarize_threshold=summarize_threshold
    )


# Re-export for convenience
__all__ = [
    "HRChatbot",
    "initialize_hr_chatbot_vector_store",
    "create_hr_chatbot_agent",
    "create_hr_chatbot",
    "chat_with_agent",
    "get_available_models",
    "get_llm",
]
