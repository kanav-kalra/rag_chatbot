"""
Graph Factory for LangGraph Studio
Provides graph instances that can be visualized and debugged in LangGraph Studio

LangGraph Studio requires variables holding compiled graphs, not functions.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure environment variables are loaded before importing settings
from dotenv import load_dotenv
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from langchain.agents import create_agent

import yaml

from src.shared.config.settings import settings
from src.shared.config.logging import logger
from src.infrastructure.llm.manager import LLMManager
from src.shared.dependencies.container import ApplicationContainer
from src.domain.retrieval.service import RetrievalService
from src.infrastructure.vectorstore import get_vector_store

# Initialize LangSmith tracing if enabled (required for LangGraph Studio)
# This ensures environment variables are set before graph creation
try:
    from src.shared.config.langsmith import initialize_langsmith
    initialize_langsmith()
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith in graph_factory: {e}. Continuing without tracing.")


def _create_hr_chatbot_graph(llm_manager: LLMManager):
    """
    Create HR chatbot graph for LangGraph Studio visualization.
    
    This function creates an agent graph that can be visualized and debugged
    in LangGraph Studio. It uses the same configuration as the production HR chatbot.
    
    Args:
        llm_manager: LLM manager instance (REQUIRED - pure dependency injection)
    
    Returns:
        LangGraph agent instance (compiled graph)
    """
    try:
        
        # Get LLM using LLM manager
        llm = llm_manager.get_llm(
            model_name=settings.CHAT_MODEL,
            temperature=settings.CHAT_MODEL_TEMPERATURE,
            max_tokens=settings.CHAT_MODEL_MAX_TOKENS
        )
        
        # Get vector store and create retrieval tool
        vector_store = get_vector_store("hr")
        retrieval_service = RetrievalService(vector_store)
        retrieve_documents_tool = retrieval_service.create_tool()
        
        # Load prompts from config/chatbot/prompts/hr_chatbot.yaml
        # Path from src/application/chatbot/graph_factory.py -> config/chatbot/prompts/
        # Go up: chatbot -> application -> src -> project_root -> config/chatbot/prompts
        prompts_file = Path(__file__).parent.parent.parent.parent / "config" / "chatbot" / "prompts" / "hr_chatbot.yaml"
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f)
        
        system_prompt_text = prompts_data.get("system_prompt", "")
        agent_instructions = prompts_data.get("agent_instructions", "")
        
        # Create system prompt
        system_prompt = system_prompt_text
        if agent_instructions:
            system_prompt = system_prompt + "\n\n" + agent_instructions
        
        # Note: No checkpointer needed for LangGraph Studio/API
        # The platform handles persistence automatically
        
        # Create and compile agent graph
        agent = create_agent(
            model=llm,
            tools=[retrieve_documents_tool],
            system_prompt=system_prompt,
            debug=True  # Enable debug mode for Studio
        )
        
        logger.info(f"Created HR chatbot graph for LangGraph Studio (model: {settings.CHAT_MODEL})")
        return agent
        
    except Exception as e:
        logger.error(f"Error creating HR chatbot graph: {e}", exc_info=True)
        raise


def _create_default_chatbot_graph(llm_manager: LLMManager):
    """
    Create default chatbot graph for LangGraph Studio visualization.
    
    This function creates a basic agent graph without retrieval tools
    that can be visualized and debugged in LangGraph Studio.
    
    Args:
        llm_manager: LLM manager instance (REQUIRED - pure dependency injection)
    
    Returns:
        LangGraph agent instance (compiled graph)
    """
    try:
        
        # Get LLM using LLM manager
        llm = llm_manager.get_llm(
            model_name=settings.CHAT_MODEL,
            temperature=settings.CHAT_MODEL_TEMPERATURE,
            max_tokens=settings.CHAT_MODEL_MAX_TOKENS
        )
        
        # Note: No checkpointer needed for LangGraph Studio/API
        # The platform handles persistence automatically
        
        # Create and compile agent graph without tools
        agent = create_agent(
            model=llm,
            tools=[],  # No tools for default chatbot
            system_prompt=None,
            debug=True  # Enable debug mode for Studio
        )
        
        logger.info(f"Created default chatbot graph for LangGraph Studio (model: {settings.CHAT_MODEL})")
        return agent
        
    except Exception as e:
        logger.error(f"Error creating default chatbot graph: {e}", exc_info=True)
        raise


# LangGraph Studio requires variables holding compiled graphs, not functions
# These are created at module import time
# Note: For LangGraph Studio, we create container explicitly (pure DI, no service locator)
try:
    # Create container explicitly for LangGraph Studio
    container = ApplicationContainer()
    container.initialize()
    llm_manager = container.get_llm_manager()
    
    # Create graphs with injected dependencies
    hr_chatbot = _create_hr_chatbot_graph(llm_manager)
    default_chatbot = _create_default_chatbot_graph(llm_manager)
except Exception as e:
    logger.error(f"Failed to create graphs at import time: {e}", exc_info=True)
    raise

