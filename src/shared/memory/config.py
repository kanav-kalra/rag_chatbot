"""
Memory Configuration for Chatbots
Defines memory management strategies (trim, summarize, or combination)

Following SOLID principles:
- Single Responsibility: MemoryConfig holds data, MemoryConfigFactory creates instances
- No default configurations - all values must be explicitly provided
"""
from typing import Optional
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
    
    All fields are required - no default values.
    Configuration must be explicitly provided.
    
    Attributes:
        strategy: Memory management strategy
            - "none": No memory management (keep all messages)
            - "trim": Trim old messages, keep only recent ones
            - "summarize": Summarize old messages when threshold is reached
            - "trim_and_summarize": Both trim and summarize
        trim_keep_messages: Number of recent messages to keep when trimming
        summarize_threshold: Number of messages before summarizing
        summarize_model: Optional model name for summarization (uses chat model if None)
    """
    strategy: MemoryStrategy
    trim_keep_messages: int
    summarize_threshold: int
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
                    "For trim_and_summarize strategy, trim_keep_messages must be "
                    "less than summarize_threshold"
                )


class MemoryConfigFactory:
    """
    Factory for creating MemoryConfig instances from dictionaries.
    
    Follows Single Responsibility Principle - only responsible for
    creating MemoryConfig instances from various sources.
    """
    
    @staticmethod
    def from_dict(config_dict: dict) -> MemoryConfig:
        """
        Create MemoryConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with keys:
                - strategy: str (required)
                - trim_keep_messages: int (required)
                - summarize_threshold: int (required)
                - summarize_model: Optional[str]
        
        Returns:
            MemoryConfig instance
            
        Raises:
            ValueError: If required fields are missing or invalid
            KeyError: If required fields are missing
        """
        if not config_dict:
            raise ValueError("Memory configuration dictionary cannot be empty")
        
        # Validate required fields
        required_fields = ["strategy", "trim_keep_messages", "summarize_threshold"]
        missing_fields = [field for field in required_fields if field not in config_dict]
        if missing_fields:
            raise ValueError(
                f"Missing required memory configuration fields: {', '.join(missing_fields)}"
            )
        
        # Parse strategy
        strategy_str = config_dict["strategy"]
        try:
            strategy = MemoryStrategy(strategy_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid memory strategy: '{strategy_str}'. "
                f"Must be one of: {[s.value for s in MemoryStrategy]}"
            ) from e
        
        # Parse numeric fields
        try:
            trim_keep_messages = int(config_dict["trim_keep_messages"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"trim_keep_messages must be an integer, got: {config_dict['trim_keep_messages']}"
            ) from e
        
        try:
            summarize_threshold = int(config_dict["summarize_threshold"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"summarize_threshold must be an integer, got: {config_dict['summarize_threshold']}"
            ) from e
        
        # Optional field
        summarize_model = config_dict.get("summarize_model")
        if summarize_model is not None and not isinstance(summarize_model, str):
            raise ValueError(f"summarize_model must be a string or None, got: {type(summarize_model)}")
        
        return MemoryConfig(
            strategy=strategy,
            trim_keep_messages=trim_keep_messages,
            summarize_threshold=summarize_threshold,
            summarize_model=summarize_model
        )
