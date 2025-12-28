"""
Generic Chatbot with RAG - Creates an agent using LangChain's create_agent
Uses LLMManager for managing multiple LLM instances
"""
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from app.core.config import settings
from app.core.logger import logger
from app.llm_manager import get_llm_manager, get_llm, get_available_models


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
    Create a generic chatbot agent using LangChain's create_agent.
    
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
        CompiledStateGraph instance (LangChain 1.0+ agent)
        
    Raises:
        ValueError: If model configuration is invalid
        RuntimeError: If agent creation fails
    """
    try:
        # Get LLM using LLM manager
        llm = get_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url
        )
        
        # Set up tools (default to empty list if not provided)
        if tools is None:
            tools = []
        
        # Create agent using create_agent (LangChain 1.0+ API)
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=verbose
        )
        
        logger.info(
            f"Created chatbot agent with model: {model_name or settings.CHAT_MODEL}"
        )
        return agent
        
    except Exception as e:
        error_msg = f"Error creating chatbot agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e




def chat_with_agent(
    agent,
    query: str,
    chat_history: Optional[List] = None
) -> str:
    """
    Chat with the chatbot agent.
    
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
