"""
Memory Configuration for Chatbots
Defines memory management strategies (trim, summarize, or combination)
"""
from typing import Literal, Optional
from enum import Enum
from dataclasses import dataclass


class MemoryStrategy(str, Enum):
    """Memory management strategies"""
    NONE = "none"
    TRIM = "trim"
    SUMMARIZE = "summarize"
    TRIM_AND_SUMMARIZE = "trim_and_summarize"


@dataclass
class MemoryConfig:
    """
    Configuration for memory management in chatbots.
    
    Attributes:
        strategy: Memory management strategy
            - "none": No memory management (keep all messages)
            - "trim": Trim old messages, keep only recent ones
            - "summarize": Summarize old messages when threshold is reached
            - "trim_and_summarize": Both trim and summarize
        trim_keep_messages: Number of recent messages to keep when trimming (default: 10)
        summarize_threshold: Number of messages before summarizing (default: 20)
        summarize_model: Optional model name for summarization (uses chat model if None)
    """
    strategy: MemoryStrategy = MemoryStrategy.NONE
    trim_keep_messages: int = 10
    summarize_threshold: int = 20
    summarize_model: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.trim_keep_messages < 1:
            raise ValueError("trim_keep_messages must be at least 1")
        if self.summarize_threshold < 1:
            raise ValueError("summarize_threshold must be at least 1")
        if self.strategy == MemoryStrategy.TRIM_AND_SUMMARIZE:
            if self.trim_keep_messages >= self.summarize_threshold:
                raise ValueError(
                    "For trim_and_summarize strategy, trim_keep_messages should be "
                    "less than summarize_threshold"
                )


# Default memory configurations for different chatbots
# These can be overridden via environment variables or API parameters
# Default summarize_model uses a model with higher context window for better summarization
DEFAULT_MEMORY_CONFIGS = {
    "hr": MemoryConfig(
        strategy=MemoryStrategy.SUMMARIZE,  # Can be overridden via settings or API
        trim_keep_messages=10,
        summarize_threshold=2,
        summarize_model="gpt-3.5-turbo-16k",  # Use high-context model for summarization
    ),
    "default": MemoryConfig(
        strategy=MemoryStrategy.NONE,
        trim_keep_messages=10,
        summarize_threshold=20,
        summarize_model="gpt-3.5-turbo-16k",  # Use high-context model for summarization
    ),
}


def get_memory_config_from_settings(chatbot_type: str = "default") -> MemoryConfig:
    """
    Get memory configuration from settings (environment variables).
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        
    Returns:
        MemoryConfig instance with values from settings
    """
    import os
    from app.core.config import settings
    
    # Get default config
    config = DEFAULT_MEMORY_CONFIGS.get(chatbot_type, DEFAULT_MEMORY_CONFIGS["default"])
    
    # Override with settings if available
    strategy_str = getattr(settings, 'DEFAULT_MEMORY_STRATEGY', None)
    if strategy_str:
        try:
            config.strategy = MemoryStrategy(strategy_str)
        except ValueError:
            pass  # Keep default if invalid
    
    config.trim_keep_messages = getattr(settings, 'MEMORY_TRIM_KEEP_MESSAGES', config.trim_keep_messages)
    config.summarize_threshold = getattr(settings, 'MEMORY_SUMMARIZE_THRESHOLD', config.summarize_threshold)
    
    # Override summarize_model from settings if provided
    summarize_model = getattr(settings, 'MEMORY_SUMMARIZE_MODEL', None)
    if summarize_model:
        config.summarize_model = summarize_model
    
    return config


def get_memory_config(chatbot_type: str = "default", use_settings: bool = True) -> MemoryConfig:
    """
    Get memory configuration for a chatbot type.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        use_settings: Whether to override with settings (default: True)
        
    Returns:
        MemoryConfig instance
    """
    if use_settings:
        return get_memory_config_from_settings(chatbot_type)
    return DEFAULT_MEMORY_CONFIGS.get(chatbot_type, DEFAULT_MEMORY_CONFIGS["default"])

