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
        summarize_threshold: Optional[int] = None,
        summarize_model: Optional[str] = None
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
            summarize_model: Model name for summarization (should have high context window, optional)
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
            if summarize_model is not None:
                memory_config.summarize_model = summarize_model
        
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
    summarize_threshold: Optional[int] = None,
    summarize_model: Optional[str] = None
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
        summarize_threshold=summarize_threshold,
        summarize_model=summarize_model
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
    summarize_threshold: Optional[int] = None,
    summarize_model: Optional[str] = None
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
        memory_config: Memory configuration object (optional, overrides other memory params)
        memory_strategy: Memory strategy - "none", "trim", "summarize", "trim_and_summarize" (optional)
        trim_keep_messages: Number of messages to keep when trimming (optional)
        summarize_threshold: Message count threshold for summarization (optional)
        
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
        summarize_threshold=summarize_threshold,
        summarize_model=summarize_model
    )


def get_default_hr_chatbot() -> HRChatbot:
    """
    Get or create a default HR chatbot instance with configuration from settings.
    This is a convenience function that creates an HR chatbot with default settings.
    For API use, this creates a singleton instance.
    
    Returns:
        HRChatbot instance configured with settings
    """
    # Memory config is automatically loaded from settings via get_memory_config("hr")
    # which uses get_memory_config_from_settings internally
    return create_hr_chatbot(
        initialize_vector_store=False,  # Vector store should be initialized on startup
        verbose=False
    )


def create_hr_chatbot_with_custom_memory(
    memory_strategy: str,
    trim_keep_messages: Optional[int] = None,
    summarize_threshold: Optional[int] = None,
    summarize_model: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> HRChatbot:
    """
    Create an HR chatbot instance with custom memory configuration.
    This is a convenience function for creating chatbots with specific memory strategies.
    
    Args:
        memory_strategy: Memory strategy - "none", "trim", "summarize", "trim_and_summarize"
        trim_keep_messages: Number of messages to keep when trimming (optional, uses default if None)
        summarize_threshold: Message count threshold for summarization (optional, uses default if None)
        model_name: Name of the LLM model to use (optional)
        temperature: Temperature for the model (optional)
        max_tokens: Maximum tokens for responses (optional)
        verbose: Whether to enable verbose logging (default: False)
        api_key: API key for the model provider (optional)
        base_url: Base URL for the model API (optional)
        
    Returns:
        HRChatbot instance with custom memory configuration
        
    Raises:
        ValueError: If memory_strategy is invalid
    """
    try:
        MemoryStrategy(memory_strategy)  # Validate strategy
    except ValueError:
        raise ValueError(
            f"Invalid memory_strategy: {memory_strategy}. "
            "Must be one of: none, trim, summarize, trim_and_summarize"
        )
    
    return create_hr_chatbot(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url,
        initialize_vector_store=False,
        memory_strategy=memory_strategy,
        trim_keep_messages=trim_keep_messages,
        summarize_threshold=summarize_threshold,
        summarize_model=summarize_model
    )


# Re-export for convenience
__all__ = [
    "HRChatbot",
    "initialize_hr_chatbot_vector_store",
    "create_hr_chatbot_agent",
    "create_hr_chatbot",
    "get_default_hr_chatbot",
    "create_hr_chatbot_with_custom_memory",
    "chat_with_agent",
    "get_available_models",
    "get_llm",
]
