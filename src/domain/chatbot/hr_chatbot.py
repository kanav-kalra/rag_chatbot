"""
HR Chatbot - Minimal implementation using refactored ChatbotAgent architecture.
All configuration comes from hr_chatbot.yaml via ChatbotConfigManager.
"""
from src.domain.chatbot.core.chatbot_agent import ChatbotAgent


class HRChatbot(ChatbotAgent):
    """
    HR-specific chatbot implementation.
    
    This is a minimal subclass that only defines:
    1. Chatbot type identifier ("hr")
    2. Config filename ("hr_chatbot.yaml")
    
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
        return "hr_chatbot.yaml"
    
    @classmethod
    def _get_default_instance(cls) -> "HRChatbot":
        """
        Create a default HR chatbot instance for the agent pool.
        
        Returns:
            HRChatbot instance with configuration from hr_chatbot.yaml
        """
        return HRChatbot()


def get_hr_chatbot() -> HRChatbot:
    """
    Get an HR chatbot instance from the agent pool.
    
    This is the recommended way to get the HR chatbot for API use.
    The agent pool ensures efficient resource usage across multiple requests.
    Thread-safe for concurrent API requests.
    
    Returns:
        HRChatbot instance from agent pool
        
    Raises:
        RuntimeError: If chatbot initialization fails
    """
    return HRChatbot.get_from_pool()


__all__ = [
    "HRChatbot",
    "get_hr_chatbot",
]
