"""
Generic Chatbot with RAG - OOP implementation
Creates an agent using LangChain's create_agent
Uses LLMManager for managing multiple LLM instances

Refactored to follow SOLID principles:
- Single Responsibility: Configuration, tools, and prompts handled by separate classes
- Open/Closed: Extensible through subclassing without modification
- Liskov Substitution: Subclasses can replace base class
- Interface Segregation: Focused abstract methods
- Dependency Inversion: Uses abstractions (config manager, tool factory, prompt builder)
"""
import sys
from pathlib import Path
from typing import Optional, List, Any
from abc import ABC, abstractmethod

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from src.shared.config.settings import settings
from src.shared.config.logging import logger
from src.infrastructure.llm.manager import get_llm_manager
from src.infrastructure.storage.checkpointing.manager import get_checkpointer
from src.shared.memory.config import MemoryConfig, MemoryStrategy, MemoryConfigFactory
from src.domain.memory.manager import MemoryManager
from src.application.chatbot.agent_pool import get_agent_pool
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys
from src.domain.chatbot.core.tools import ChatbotToolFactory
from src.domain.chatbot.core.prompts import ChatbotPromptBuilder


# Class-level cache for chatbot types to avoid creating instances just to get the type
_chatbot_type_cache: dict = {}


class ChatbotAgent(ABC):
    """
    Abstract base class for chatbot agents with RAG capabilities.
    Encapsulates agent creation and chat functionality.
    
    This class cannot be instantiated directly. Subclasses must implement
    the required methods and provide chatbot-specific configuration.
    
    Extension Points:
    -----------------
    Subclasses can customize behavior by overriding these methods:
    
    1. _get_chatbot_type() - REQUIRED: Return chatbot type identifier
    2. _get_config_filename() - Return YAML config filename (optional, enables auto-loading)
    
    Note: If _get_config_filename() is defined, the base class automatically loads
    the YAML config and uses it for ALL configuration:
    - model_name, temperature, max_tokens, base_url, verbose
    - memory configuration (strategy, trim_keep_messages, etc.)
    - tools configuration (enable_retrieval, vector_store.type)
    - system_prompt (template and agent_instructions_template)
    - agent_pool.size (for agent pool configuration)
    
    If YAML config doesn't have values, it falls back to:
    - settings.CHAT_MODEL, settings.CHAT_MODEL_TEMPERATURE, etc. for model settings
    - Default prompts from default_prompts.yaml (or custom prompts file if _get_prompts_filename() is overridden)
    - Generic defaults for memory and other settings
    
    For YAML-based chatbots (like HRChatbot), ALL configuration comes from the YAML file.
    No need to override any hook methods!
    
    Example Extension (YAML-based configuration):
    -------------------------------------------
    To create a new chatbot with YAML configuration:
    
    1. Create a YAML config file in config/chatbot/ (e.g., "support_chatbot_config.yaml"):
       ```yaml
       model:
         name: "gpt-4"
         temperature: 0.7
         max_tokens: 2000
       vector_store:
         type: "support"
       tools:
         enable_retrieval: true
       memory:
         strategy: "trim"
         trim_keep_messages: 5
       agent_pool:
         size: 2
       ```
    
    2. Create a subclass:
       ```python
       class SupportChatbot(ChatbotAgent):
           def _get_chatbot_type(self) -> str:
               return "support"
           
           @classmethod
           def _get_config_filename(cls) -> str:
               return "support_chatbot_config.yaml"
           
           # That's it! The base class automatically loads the YAML config
           # and uses it for ALL configuration (model, memory, tools, system_prompt).
           # No need to override any hook methods - everything comes from YAML!
       ```
    
    3. Use it:
       ```python
       chatbot = SupportChatbot.get_from_pool()
       response = chatbot.chat("Hello", thread_id="thread-123")
       ```
    
    That's it! The base class handles all the configuration loading, agent creation,
    and memory management. You only need to specify the chatbot type and config filename.
    
    Core Logic:
    ----------
    The core chat functionality, memory management, and agent initialization
    are handled by this base class. Subclasses only need to customize
    configuration and behavior through the extension hooks above.
    """
    
    @abstractmethod
    def _get_chatbot_type(self) -> str:
        """
        Get the chatbot type identifier.
        Must be implemented by subclasses.
        
        Returns:
            Chatbot type string (e.g., "hr", "default")
        """
        pass
    
    @classmethod
    def _get_chatbot_type_class(cls) -> Optional[str]:
        """
        Get the chatbot type identifier without creating an instance.
        Optional class method that subclasses can override to avoid instantiation.
        
        If not overridden, falls back to creating a temporary instance.
        
        Returns:
            Chatbot type string (e.g., "hr", "default") or None to use instance method
        """
        return None
    
    @classmethod
    @abstractmethod
    def _get_default_instance(cls):
        """
        Get or create a default instance of this chatbot type.
        Must be implemented by subclasses.
        
        This is used by the agent pool to create new instances.
        
        Returns:
            Instance of the chatbot subclass with default configuration
        """
        pass
    
    # Note: _get_pool_size() has been removed. Pool size is now read from YAML config
    # (agent_pool.size) in the get_from_pool() method.
    
    @classmethod
    def _get_config_filename(cls) -> Optional[str]:
        """
        Get the YAML config filename for this chatbot type.
        Override in subclasses to specify a config file.
        
        Returns:
            Config filename (e.g., "hr_chatbot_config.yaml") or None if no config file
        """
        return None
    
    @classmethod
    def _get_prompts_filename(cls) -> Optional[str]:
        """
        Get the prompts YAML filename for this chatbot type.
        Override in subclasses to specify a custom prompts file.
        
        Returns:
            Prompts filename (e.g., "hr_chatbot_prompts.yaml") or None to use default_prompts.yaml
        """
        return None
    
    def _create_memory_config(self, config_manager: ChatbotConfigManager) -> MemoryConfig:
        """
        Create memory configuration from config manager.
        
        Args:
            config_manager: Config manager instance
        
        Returns:
            MemoryConfig instance
            
        Raises:
            ValueError: If memory configuration is missing or invalid
        """
        # Get memory config from YAML - all fields are required
        memory_config_dict = config_manager.get_nested_dict(ConfigKeys.MEMORY, {})
        
        if not memory_config_dict:
            raise ValueError(
                f"Memory configuration is required for chatbot type '{self.chatbot_type}'. "
                f"Please provide memory configuration in the YAML config file."
            )
        
        # Use factory to create MemoryConfig - validates all required fields
        return MemoryConfigFactory.from_dict(memory_config_dict)
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None
    ):
        """
        Initialize chatbot agent.
        
        Args:
            model_name: Name of the LLM model to use (default: from YAML config or settings.CHAT_MODEL)
            temperature: Temperature for the model (default: from YAML config or settings.CHAT_MODEL_TEMPERATURE)
            max_tokens: Maximum tokens for responses (default: from YAML config or settings.CHAT_MODEL_MAX_TOKENS)
            tools: List of tools for the agent to use (default: from tool factory)
            system_prompt: System prompt for the agent (default: from YAML config)
            verbose: Whether to enable verbose logging (default: from YAML config or False)
            api_key: API key for the model provider (optional)
            base_url: Base URL for the model API (default: from YAML config or None)
            memory_config: Memory configuration for managing chat history (optional)
        """
        # Initialize config manager
        config_filename = self.__class__._get_config_filename()
        try:
            self._config_manager = ChatbotConfigManager(config_filename) if config_filename else None
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to load YAML config {config_filename}: {e}. Using defaults.")
            self._config_manager = None
        
        # Get chatbot type from subclass implementation
        self.chatbot_type = self._get_chatbot_type()
        
        # Initialize model parameters from config or provided values
        self.model_name = model_name or self._get_model_name()
        self.temperature = temperature if temperature is not None else self._get_temperature()
        self.max_tokens = max_tokens if max_tokens is not None else self._get_max_tokens()
        self.base_url = base_url or self._get_base_url()
        self.verbose = verbose if verbose is not None else self._get_verbose()
        self.api_key = api_key
        
        # Initialize tool factory and create tools
        tool_factory = ChatbotToolFactory(self._config_manager)
        self.tools = tool_factory.create_tools(self.chatbot_type, provided_tools=tools)
        
        # Initialize prompt builder and create system prompt
        if system_prompt is None:
            prompt_builder = ChatbotPromptBuilder(self._config_manager)
            system_prompt = prompt_builder.build_system_prompt()
        self.system_prompt = system_prompt
        
        # Get memory config
        if memory_config is not None:
            self.memory_config = memory_config
        else:
            self.memory_config = self._create_memory_config(
                self._config_manager or ChatbotConfigManager()
            )
        
        # Create the agent
        self._agent = None
        self._initialize_agent()
    
    def _get_model_name(self) -> str:
        """Get model name from config or settings."""
        if self._config_manager:
            model_name = self._config_manager.get(ConfigKeys.MODEL_NAME)
            if model_name:
                return model_name
        return settings.CHAT_MODEL
    
    def _get_temperature(self) -> float:
        """Get temperature from config or settings."""
        if self._config_manager:
            temp = self._config_manager.get(ConfigKeys.MODEL_TEMPERATURE)
            if temp is not None:
                return float(temp)
        return settings.CHAT_MODEL_TEMPERATURE
    
    def _get_max_tokens(self) -> int:
        """Get max tokens from config or settings."""
        if self._config_manager:
            max_tok = self._config_manager.get(ConfigKeys.MODEL_MAX_TOKENS)
            if max_tok is not None:
                return int(max_tok)
        return settings.CHAT_MODEL_MAX_TOKENS
    
    def _get_base_url(self) -> Optional[str]:
        """Get base URL from config."""
        if self._config_manager:
            return self._config_manager.get(ConfigKeys.MODEL_BASE_URL)
        return None
    
    def _get_verbose(self) -> bool:
        """Get verbose flag from config or default."""
        if self._config_manager:
            return self._config_manager.get(ConfigKeys.VERBOSE, False)
        return False
    
    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with checkpointer and memory management."""
        try:
            # Get LLM using LLM manager
            llm = get_llm_manager().get_llm(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Get checkpointer for short-term memory
            checkpointer = get_checkpointer()
            
            # Create memory manager for processing messages
            self._memory_manager = MemoryManager(self.memory_config) if self.memory_config.strategy.value != "none" else None
            
            # Create agent using create_agent (LangChain 1.0+ API)
            self._agent = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
                checkpointer=checkpointer,
                debug=self.verbose
            )
            
            logger.info(
                f"Initialized ChatbotAgent with model: {self.model_name}, "
                f"memory_strategy: {self.memory_config.strategy.value}"
            )
            
        except Exception as e:
            error_msg = f"Error initializing chatbot agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    @property
    def agent(self):
        """Get the underlying agent instance."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        return self._agent
    
    def chat(
        self,
        query: str,
        thread_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Chat with the chatbot agent using checkpointer for memory management.
        
        Args:
            query: User's question
            thread_id: Thread/session identifier for maintaining conversation context
            user_id: Optional user identifier
        
        Returns:
            Agent's response as a string
        """
        try:
            # Prepare input messages
            from langchain_core.messages import HumanMessage
            from src.infrastructure.storage.checkpointing.manager import get_checkpointer_manager
            
            messages = [HumanMessage(content=query)]
            inputs = {"messages": messages}
            
            # Create config with thread_id for checkpointer
            config = get_checkpointer_manager().get_config(thread_id, user_id)
            
            # Invoke agent with checkpointer config
            result = self.agent.invoke(inputs, config=config)
            
            # Extract response from the result
            response = self._extract_response(result)
            
            logger.debug(
                f"Chat response generated (length: {len(response)} chars, "
                f"thread_id: {thread_id})"
            )
            return response
            
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _extract_response(self, result: Any) -> str:
        """
        Extract response text from agent result.
        Compatible with both OpenAI and Gemini (Google) model response formats.
        
        Handles:
        - OpenAI: String content in messages
        - Gemini: List of content blocks like [{'type': 'text', 'text': '...'}]
        - LangChain message objects with content attribute
        - Dictionary representations of messages
        
        Args:
            result: Agent invocation result
        
        Returns:
            Response text as string
        """
        def extract_text_from_content(content: Any) -> str:
            """
            Extract text from content, handling various formats from different LLM providers.
            
            Formats handled:
            - String (OpenAI, most providers)
            - List of strings (some providers)
            - List of content blocks (Gemini): [{'type': 'text', 'text': '...'}]
            - Other types: converted to string
            """
            if isinstance(content, str):
                # OpenAI and most providers return string directly
                return content
            elif isinstance(content, list):
                # Handle list of content blocks (Gemini format: [{'type': 'text', 'text': '...'}])
                # or list of strings
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        # Gemini format: {'type': 'text', 'text': '...'}
                        if block.get('type') == 'text' and 'text' in block:
                            text_parts.append(str(block['text']))
                        # Fallback: any dict with 'text' key
                        elif 'text' in block:
                            text_parts.append(str(block['text']))
                        # Fallback: any dict with 'content' key
                        elif 'content' in block:
                            text_parts.append(str(block['content']))
                    elif isinstance(block, str):
                        # List of strings
                        text_parts.append(block)
                    else:
                        # Unknown format, convert to string
                        text_parts.append(str(block))
                return ''.join(text_parts) if text_parts else str(content)
            else:
                # Fallback: convert to string
                return str(content)
        
        # Handle LangChain message objects (from langchain_core.messages)
        try:
            from langchain_core.messages import BaseMessage
            if isinstance(result, BaseMessage):
                return extract_text_from_content(result.content)
        except (ImportError, AttributeError):
            pass
        
        # Handle dictionary results (most common format from agent.invoke())
        if isinstance(result, dict):
            # Try different possible response formats
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                
                # Handle LangChain message objects
                if hasattr(last_message, 'content'):
                    return extract_text_from_content(last_message.content)
                # Handle dictionary representation of messages
                elif isinstance(last_message, dict):
                    if "content" in last_message:
                        return extract_text_from_content(last_message["content"])
                    # Fallback: try to find any text-like field
                    for key in ["text", "message", "output"]:
                        if key in last_message:
                            return extract_text_from_content(last_message[key])
            
            # Try direct output field
            if "output" in result:
                return extract_text_from_content(result["output"])
            
            # Fallback: convert entire dict to string
            return str(result)
        
        # Fallback: convert to string
        return str(result)
    
    def update_tools(self, tools: List[BaseTool]) -> None:
        """
        Update agent tools (requires reinitialization).
        
        Args:
            tools: New list of tools
        """
        self.tools = tools
        self._initialize_agent()
        logger.info(f"Updated agent tools (count: {len(tools)})")
    
    def update_system_prompt(self, system_prompt: str) -> None:
        """
        Update system prompt (requires reinitialization).
        
        Args:
            system_prompt: New system prompt
        """
        self.system_prompt = system_prompt
        self._initialize_agent()
        logger.info("Updated agent system prompt")
    
    def update_memory_config(self, memory_config: MemoryConfig) -> None:
        """
        Update memory configuration (requires reinitialization).
        
        Args:
            memory_config: New memory configuration
        """
        self.memory_config = memory_config
        self._initialize_agent()
        logger.info(f"Updated memory config: {memory_config.strategy.value}")
    
    @classmethod
    def get_from_pool(cls):
        """
        Get a chatbot instance from the agent pool.
        
        Generic method that works for any chatbot subclass.
        Uses the agent pool for efficient resource usage across multiple requests.
        Thread-safe for concurrent API requests.
        
        The chatbot type is cached per class to avoid creating temporary instances
        on every call, which would trigger unnecessary initialization.
        
        Returns:
            ChatbotAgent instance from agent pool
            
        Raises:
            RuntimeError: If chatbot initialization fails
        """
        try:
            # Get chatbot type from cache or by using class method or creating a temporary instance
            class_key = cls.__name__
            if class_key not in _chatbot_type_cache:
                # Try class method first (avoids instantiation)
                chatbot_type = cls._get_chatbot_type_class()
                if chatbot_type is None:
                    # Fall back to creating a temporary instance (only once per class)
                    temp_instance = cls._get_default_instance()
                    chatbot_type = temp_instance._get_chatbot_type()
                _chatbot_type_cache[class_key] = chatbot_type
            chatbot_type = _chatbot_type_cache[class_key]
            
            # Get pool size from YAML config if available
            pool_size = 1  # Default
            config_filename = cls._get_config_filename()
            if config_filename:
                try:
                    config_manager = ChatbotConfigManager(config_filename)
                    pool_size = config_manager.get(ConfigKeys.AGENT_POOL_SIZE, 1)
                except (FileNotFoundError, ValueError, RuntimeError):
                    # If config loading fails, use default
                    pass
            
            # Get agent pool for this chatbot type
            # The pool will create instances using the factory if needed
            agent_pool = get_agent_pool(
                chatbot_type=chatbot_type,
                agent_factory=cls._get_default_instance,
                pool_size=pool_size
            )
            
            # Get agent from pool
            chatbot = agent_pool.get_agent()
            return chatbot
        except Exception as e:
            logger.error(f"Failed to get chatbot from pool: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get chatbot from pool: {str(e)}") from e


# Export for convenience
__all__ = [
    "ChatbotAgent",
]
