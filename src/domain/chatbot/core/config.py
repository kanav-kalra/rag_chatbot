"""
Configuration Manager for Chatbot Agents
Handles YAML configuration loading, caching, and value access.
Follows Single Responsibility Principle.
"""
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from src.shared.config.logging import logger


class ConfigKeys:
    """Constants for configuration keys to avoid magic strings."""
    MODEL_NAME = "model.name"
    MODEL_TEMPERATURE = "model.temperature"
    MODEL_MAX_TOKENS = "model.max_tokens"
    MODEL_BASE_URL = "model.base_url"
    VECTOR_STORE_TYPE = "vector_store.type"
    TOOLS_ENABLE_RETRIEVAL = "tools.enable_retrieval"
    TOOLS_ADDITIONAL = "tools.additional"
    SYSTEM_PROMPT_TEMPLATE = "system_prompt.template"
    SYSTEM_PROMPT_AGENT_INSTRUCTIONS = "system_prompt.agent_instructions_template"
    SYSTEM_PROMPT_PROMPTS_FILE = "system_prompt.prompts_file"
    MEMORY = "memory"
    MEMORY_STRATEGY = "memory.strategy"
    MEMORY_TRIM_KEEP_MESSAGES = "memory.trim_keep_messages"
    MEMORY_SUMMARIZE_THRESHOLD = "memory.summarize_threshold"
    MEMORY_SUMMARIZE_MODEL = "memory.summarize_model"
    AGENT_POOL_SIZE = "agent_pool.size"
    VERBOSE = "verbose"


class ChatbotConfigManager:
    """
    Manages chatbot configuration loading and access.
    Handles YAML file loading, caching, and nested value retrieval.
    """
    
    _config_cache: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, config_filename: Optional[str] = None, config_dir: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_filename: Name of the YAML config file (e.g., "hr_chatbot_config.yaml")
            config_dir: Directory containing config files (default: config/chatbot/)
        """
        self.config_filename = config_filename
        self.config_dir = config_dir or self._get_default_config_dir()
        self._config: Optional[Dict[str, Any]] = None
        
        if config_filename:
            self._load_config()
    
    @staticmethod
    def _get_default_config_dir() -> Path:
        """Get default config directory (config/chatbot/)."""
        # Path from src/domain/chatbot/core/config.py -> config/chatbot/
        # Go up: core -> chatbot -> domain -> src -> project_root -> config/chatbot
        return Path(__file__).parent.parent.parent.parent.parent / "config" / "chatbot"
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with caching."""
        if not self.config_filename:
            return
        
        # Check cache first
        cache_key = str(self.config_dir / self.config_filename)
        if cache_key in self._config_cache:
            self._config = self._config_cache[cache_key]
            return
        
        # Load from file
        config_path = self.config_dir / self.config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            # Cache the config
            self._config_cache[cache_key] = self._config
            logger.debug(f"Loaded config from {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the config using dot notation (e.g., "model.name").
        
        Args:
            key: Dot-separated key path (e.g., "model.name", "tools.enable_retrieval")
            default: Default value if key not found
        
        Returns:
            Config value or default
        """
        if self._config is None:
            return default
        
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value if value is not None else default
    
    def get_nested_dict(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a nested dictionary from config.
        
        Args:
            key: Dot-separated key path
            default: Default dictionary if key not found
        
        Returns:
            Dictionary from config or default
        """
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default or {}
    
    def has_config(self) -> bool:
        """Check if config is loaded."""
        return self._config is not None
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the config cache (useful for testing)."""
        cls._config_cache.clear()

