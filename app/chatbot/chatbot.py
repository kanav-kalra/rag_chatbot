"""
Generic Chatbot with RAG - OOP implementation
Creates an agent using LangChain's create_agent
Uses LLMManager for managing multiple LLM instances
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from app.core.config import settings
from app.core.logger import logger
from app.llm_manager import get_llm_manager, get_llm, get_available_models


class ChatbotAgent:
    """
    Generic chatbot agent class with RAG capabilities.
    Encapsulates agent creation and chat functionality.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
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
        """
        self.model_name = model_name or settings.CHAT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.api_key = api_key
        self.base_url = base_url
        
        # Create the agent
        self._agent = None
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent."""
        try:
            # Get LLM using LLM manager
            llm = get_llm(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Create agent using create_agent (LangChain 1.0+ API)
            self._agent = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
                debug=self.verbose
            )
            
            logger.info(
                f"Initialized ChatbotAgent with model: {self.model_name}"
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
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Chat with the chatbot agent.
        
        Args:
            query: User's question
            chat_history: Optional chat history (list of dicts with 'role' and 'content')
        
        Returns:
            Agent's response as a string
        """
        try:
            # Prepare input messages
            messages = [{"role": "user", "content": query}]
            if chat_history:
                messages = chat_history + messages
            
            inputs = {"messages": messages}
            
            # Invoke agent
            result = self.agent.invoke(inputs)
            
            # Extract response from the result
            response = self._extract_response(result)
            
            logger.debug(f"Chat response generated (length: {len(response)} chars)")
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


# Factory functions for backward compatibility
def create_chatbot_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    """
    Create a generic chatbot agent (factory function for backward compatibility).
    
    Args:
        model_name: Name of the LLM model to use (default: from settings)
        temperature: Temperature for the model (default: from settings)
        max_tokens: Maximum tokens for responses (default: from settings)
        tools: List of tools for the agent to use (default: empty list)
        system_prompt: System prompt for the agent (optional)
        verbose: Whether to enable verbose logging (default: False)
        api_key: API key for the model provider (optional)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        
    Returns:
        ChatbotAgent instance
    """
    agent = ChatbotAgent(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url
    )
    return agent.agent  # Return the underlying agent for backward compatibility


def chat_with_agent(
    agent,
    query: str,
    chat_history: Optional[List] = None
) -> str:
    """
    Chat with the chatbot agent (backward compatibility function).
    
    Args:
        agent: The agent instance (CompiledStateGraph from LangChain 1.0+)
        query: User's question
        chat_history: Optional chat history (list of messages)
        
    Returns:
        Agent's response as a string
    """
    try:
        # LangChain 1.0+ CompiledStateGraph
        inputs = {"messages": [{"role": "user", "content": query}]}
        if chat_history:
            inputs["messages"] = chat_history + inputs["messages"]
        
        result = agent.invoke(inputs)
        
        # Extract response from the result
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
        
    except Exception as e:
        error_msg = f"Error during chat: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"I encountered an error while processing your question: {str(e)}"


# Export for convenience
__all__ = [
    "ChatbotAgent",
    "create_chatbot_agent",
    "chat_with_agent",
    "get_available_models",
]
