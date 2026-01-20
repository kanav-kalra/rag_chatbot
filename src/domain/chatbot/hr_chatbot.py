"""
HR Chatbot - Minimal implementation using refactored ChatbotAgent architecture.
All configuration comes from hr_chatbot_config.yaml via ChatbotConfigManager.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infrastructure.llm.manager import LLMManager

from src.domain.chatbot.core.chatbot_agent import ChatbotAgent


class HRChatbot(ChatbotAgent):
    """
    HR-specific chatbot implementation.
    
    This is a minimal subclass that only defines:
    1. Chatbot type identifier ("hr")
    2. Config filename ("hr_chatbot_config.yaml")
    
    All other functionality (config loading, tool creation, prompt building, etc.)
    is handled by the base ChatbotAgent class using the new architecture:
    - ChatbotConfigManager for configuration
    - ChatbotToolFactory for tools
    - ChatbotPromptBuilder for prompts
    
    Usage:
        chatbot = HRChatbot.get_from_pool()
        response = chatbot.chat("Hello", thread_id="thread-123")
    """
    
    def _get_chatbot_type(self) -> str:
        """Return the chatbot type identifier."""
        return "hr"
    
    @classmethod
    def _get_chatbot_type_class(cls) -> str:
        """Return the chatbot type identifier without instantiation."""
        return "hr"
    
    @classmethod
    def _get_config_filename(cls) -> str:
        """Return the YAML config filename."""
        return "hr_chatbot_config.yaml"
    
    @classmethod
    def _get_default_instance(cls, llm_manager: "LLMManager") -> "HRChatbot":
        """
        Create a default HR chatbot instance for the agent pool.
        
        Args:
            llm_manager: LLM manager instance (REQUIRED)
        
        Returns:
            HRChatbot instance with configuration from hr_chatbot_config.yaml
        """
        return HRChatbot(llm_manager=llm_manager)


def get_hr_chatbot(llm_manager: "LLMManager") -> HRChatbot:
    """
    Get an HR chatbot instance from the agent pool.
    
    This is the recommended way to get the HR chatbot for API use.
    The agent pool ensures efficient resource usage across multiple requests.
    Thread-safe for concurrent API requests.
    
    Args:
        llm_manager: LLM manager instance (REQUIRED - from dependency injection)
    
    Returns:
        HRChatbot instance from agent pool
        
    Raises:
        RuntimeError: If chatbot initialization fails
        ValueError: If llm_manager is not provided
    """
    return HRChatbot.get_from_pool(llm_manager=llm_manager)


__all__ = [
    "HRChatbot",
    "get_hr_chatbot",
]
