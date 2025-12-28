"""
LLM Manager Package - Factory for creating and managing multiple LLM instances
"""
from app.llm_manager.llm_manager import (
    LLMManager,
    get_llm_manager,
    get_llm,
    get_available_models,
    MODEL_CONFIGS,
)

__all__ = [
    "LLMManager",
    "get_llm_manager",
    "get_llm",
    "get_available_models",
    "MODEL_CONFIGS",
]
