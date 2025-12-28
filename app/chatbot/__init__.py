"""
Chatbot Module - Generic chatbot and HR-specific chatbot implementations
"""
# Generic chatbot exports
from app.chatbot.chatbot import (
    ChatbotAgent,
    create_chatbot_agent,
    chat_with_agent,
)

# HR chatbot exports
from app.chatbot.hr_chatbot import (
    HRChatbot,
    create_hr_chatbot_agent,
    create_hr_chatbot,
    initialize_hr_chatbot_vector_store,
)

# Session management exports
from app.chatbot.session_manager import (
    ChatbotSession,
    ChatbotSessionManager,
)

# LLM Manager exports
from app.llm_manager import (
    LLMManager,
    get_llm_manager,
    get_llm,
    get_available_models,
)

# Prompts exports
from app.chatbot.prompts import (
    HR_CHATBOT_SYSTEM_PROMPT,
    AGENT_INSTRUCTIONS,
    get_hr_chatbot_prompt,
    HR_TOPIC_PROMPTS,
    get_rag_prompt_template,
)

# For backward compatibility
create_agent = create_hr_chatbot_agent

__all__ = [
    # Generic chatbot classes
    "ChatbotAgent",
    "create_chatbot_agent",
    "chat_with_agent",
    # HR chatbot classes
    "HRChatbot",
    "create_hr_chatbot_agent",
    "create_hr_chatbot",
    "initialize_hr_chatbot_vector_store",
    "create_agent",  # Backward compatibility
    # Session management
    "ChatbotSession",
    "ChatbotSessionManager",
    # LLM Manager
    "LLMManager",
    "get_llm_manager",
    "get_llm",
    "get_available_models",
    # Prompts
    "HR_CHATBOT_SYSTEM_PROMPT",
    "AGENT_INSTRUCTIONS",
    "get_hr_chatbot_prompt",
    "HR_TOPIC_PROMPTS",
    "get_rag_prompt_template",
]

