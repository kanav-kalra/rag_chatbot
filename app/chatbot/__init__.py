"""
Chatbot Module - Generic chatbot and HR-specific chatbot implementations
"""
# Generic chatbot exports
from app.chatbot.chatbot import ChatbotAgent

# HR chatbot exports
from app.chatbot.hr_chatbot import HRChatbot, get_hr_chatbot

# Agent pool exports
from app.chatbot.agent_pool import AgentPool, get_agent_pool

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
)

# Prompts exports
from app.chatbot.prompts import (
    HR_CHATBOT_SYSTEM_PROMPT,
    AGENT_INSTRUCTIONS,
    get_hr_chatbot_prompt,
    HR_TOPIC_PROMPTS,
    get_rag_prompt_template,
)

__all__ = [
    # Generic chatbot classes
    "ChatbotAgent",
    # HR chatbot classes
    "HRChatbot",
    "get_hr_chatbot",
    # Agent pool
    "AgentPool",
    "get_agent_pool",
    # Session management
    "ChatbotSession",
    "ChatbotSessionManager",
    # LLM Manager
    "LLMManager",
    "get_llm_manager",
    "get_llm",
    # Prompts
    "HR_CHATBOT_SYSTEM_PROMPT",
    "AGENT_INSTRUCTIONS",
    "get_hr_chatbot_prompt",
    "HR_TOPIC_PROMPTS",
    "get_rag_prompt_template",
]

