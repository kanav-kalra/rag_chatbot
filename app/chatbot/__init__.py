"""
Chatbot Module - Generic chatbot and HR-specific chatbot implementations
"""
# Generic chatbot exports
from app.chatbot.chatbot import (
    create_chatbot_agent,
    chat_with_agent,
)

# LLM Manager exports
from app.llm_manager import (
    LLMManager,
    get_llm_manager,
    get_llm,
    get_available_models,
)

# HR chatbot exports
from app.chatbot.hr_chatbot import (
    create_hr_chatbot_agent,
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
    # Generic chatbot
    "create_chatbot_agent",
    "chat_with_agent",
    # LLM Manager
    "LLMManager",
    "get_llm_manager",
    "get_llm",
    "get_available_models",
    # HR chatbot
    "create_hr_chatbot_agent",
    "create_agent",  # Backward compatibility
    # Prompts
    "HR_CHATBOT_SYSTEM_PROMPT",
    "AGENT_INSTRUCTIONS",
    "get_hr_chatbot_prompt",
    "HR_TOPIC_PROMPTS",
    "get_rag_prompt_template",
]

