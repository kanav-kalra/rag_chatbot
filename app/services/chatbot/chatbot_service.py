"""
Generic Chatbot with RAG - OOP implementation
Creates an agent using LangChain's create_agent
Uses LLMManager for managing multiple LLM instances
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from app.core.config import settings
from app.core.logging import logger
from app.infra.llm.llm_manager import get_llm_manager, get_llm
from app.infra.checkpointing.checkpoint_manager import get_checkpointer
from app.core.memory_config import MemoryConfig, get_memory_config
from app.services.memory.memory_manager import MemoryManager


class ChatbotAgent(ABC):
    """
    Abstract base class for chatbot agents with RAG capabilities.
    Encapsulates agent creation and chat functionality.
    
    This class cannot be instantiated directly. Subclasses must implement
    the required methods and provide chatbot-specific configuration.
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
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None,
        chatbot_type: str = "default"
    ):
        """
        Initialize chatbot agent.
        
        Args:
            model_name: Name of the LLM model to use (default: from settings)
            temperature: Temperature for the model (default: from settings)
            max_tokens: Maximum tokens for responses (default: from settings)
            tools: List of tools for the agent to use (default: empty list)
            system_prompt: System prompt for the agent (optional)
            verbose: Whether to enable verbose logging (default: False)
            api_key: API key for the model provider (optional)
            base_url: Base URL for the model API (optional, mainly for Ollama)
            memory_config: Memory configuration for managing chat history (optional)
            chatbot_type: Type of chatbot for default memory config (default: "default")
        """
        self.model_name = model_name or settings.CHAT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.api_key = api_key
        self.base_url = base_url
        
        # Memory configuration
        # Use provided chatbot_type or get from subclass implementation
        self.chatbot_type = chatbot_type if chatbot_type else self._get_chatbot_type()
        self.memory_config = memory_config or get_memory_config(self.chatbot_type)
        
        # Create the agent
        self._agent = None
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with checkpointer and memory management."""
        try:
            # Get LLM using LLM manager
            llm = get_llm(
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
            from app.infra.checkpointing.checkpoint_manager import get_checkpointer_manager
            
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
        
        Args:
            result: Agent invocation result
        
        Returns:
            Response text as string
        """
        if isinstance(result, dict):
            # Try different possible response formats
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
            return result.get("output", str(result))
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


# Export for convenience
__all__ = [
    "ChatbotAgent",
]
