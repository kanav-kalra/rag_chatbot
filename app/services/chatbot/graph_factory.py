"""
Graph Factory for LangGraph Studio
Provides graph instances that can be visualized and debugged in LangGraph Studio

LangGraph Studio requires variables holding compiled graphs, not functions.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure environment variables are loaded before importing settings
from dotenv import load_dotenv
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from langchain.agents import create_agent

import yaml

from app.core.config import settings
from app.core.logging import logger
from app.infra.llm.llm_manager import get_llm_manager
from app.services.retrieval.retrieval_service import RetrievalService
from app.infra.vectorstore import get_vector_store

# Initialize LangSmith tracing if enabled (required for LangGraph Studio)
# This ensures environment variables are set before graph creation
try:
    from app.core.langsmith_config import initialize_langsmith
    initialize_langsmith()
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith in graph_factory: {e}. Continuing without tracing.")


def _create_hr_chatbot_graph():
    """
    Create HR chatbot graph for LangGraph Studio visualization.
    
    This function creates an agent graph that can be visualized and debugged
    in LangGraph Studio. It uses the same configuration as the production HR chatbot.
    
    Returns:
        LangGraph agent instance (compiled graph)
    """
    try:
        # Get LLM using settings
        llm = get_llm_manager().get_llm(
            model_name=settings.CHAT_MODEL,
            temperature=settings.CHAT_MODEL_TEMPERATURE,
            max_tokens=settings.CHAT_MODEL_MAX_TOKENS
        )
        
        # Get vector store and create retrieval tool
        vector_store = get_vector_store("hr")
        retrieval_service = RetrievalService(vector_store)
        retrieve_documents_tool = retrieval_service.create_tool()
        
        # Load prompts from hr_chatbot_prompts.yaml
        prompts_file = Path(__file__).parent / "hr_chatbot_prompts.yaml"
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


def _create_default_chatbot_graph():
    """
    Create default chatbot graph for LangGraph Studio visualization.
    
    This function creates a basic agent graph without retrieval tools
    that can be visualized and debugged in LangGraph Studio.
    
    Returns:
        LangGraph agent instance (compiled graph)
    """
    try:
        # Get LLM using settings
        llm = get_llm_manager().get_llm(
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
# Note: Ensure .env file is loaded and dependencies are available
try:
    hr_chatbot = _create_hr_chatbot_graph()
except Exception as e:
    logger.error(f"Failed to create hr_chatbot graph at import time: {e}")
    raise

try:
    default_chatbot = _create_default_chatbot_graph()
except Exception as e:
    logger.error(f"Failed to create default_chatbot graph at import time: {e}")
    raise

