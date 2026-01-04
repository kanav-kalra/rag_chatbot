"""
LLM Manager - Factory class for creating and managing multiple LLM instances
Supports different LLM providers and allows multiple instances for different use cases
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Optional imports
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    ChatGoogleGenerativeAI = None

from app.core.config import settings
from app.core.logging import logger


# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "class": ChatOpenAI,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "class": ChatOpenAI,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "class": ChatOpenAI,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "class": ChatOpenAI,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229",
        "class": ChatAnthropic if ANTHROPIC_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-anthropic",
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "class": ChatAnthropic if ANTHROPIC_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-anthropic",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "class": ChatAnthropic if ANTHROPIC_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-anthropic",
    },
    "llama2": {
        "provider": "ollama",
        "model_name": "llama2",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "llama3": {
        "provider": "ollama",
        "model_name": "llama3",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "llama3.1": {
        "provider": "ollama",
        "model_name": "llama3.1",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "llama3.2": {
        "provider": "ollama",
        "model_name": "llama3.2",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "mistral": {
        "provider": "ollama",
        "model_name": "mistral",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "phi3": {
        "provider": "ollama",
        "model_name": "phi3",
        "class": ChatOllama if OLLAMA_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-ollama",
    },
    "gemini-pro": {
        "provider": "google",
        "model_name": "gemini-pro",
        "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-google-genai",
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "model_name": "gemini-1.5-flash",
        "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-google-genai",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "requires_package": "langchain-google-genai",
    },
}


class LLMManager:
    """
    Manager class for creating and managing multiple LLM instances.
    Allows different LLMs for different use cases (e.g., Q&A, summarization, classification).
    """
    
    def __init__(self):
        """Initialize the LLM Manager with an empty cache."""
        self._llm_cache: Dict[str, BaseChatModel] = {}
        self._default_model: Optional[str] = None
    
    def get_llm(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> BaseChatModel:
        """
        Create or retrieve an LLM instance.
        
        Args:
            model_name: Name of the model to use (default: from settings)
            temperature: Temperature for the model (default: from settings or model default)
            max_tokens: Maximum tokens for the model (default: from settings or model default)
            api_key: API key for the model provider (optional, uses env vars if not provided)
            base_url: Base URL for the model API (optional, mainly for Ollama)
            use_cache: Whether to cache the LLM instance (default: True)
            cache_key: Custom cache key (default: model_name). Use different keys for same model with different params
            
        Returns:
            BaseChatModel instance
            
        Raises:
            ValueError: If model is not supported or API key is missing
        """
        model_name = model_name or self._default_model or settings.CHAT_MODEL
        
        if model_name not in MODEL_CONFIGS:
            available_models = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Available models: {available_models}"
            )
        
        # Create cache key
        if cache_key is None:
            cache_key = f"{model_name}_{temperature}_{max_tokens}_{base_url}"
        
        # Return cached instance if available
        if use_cache and cache_key in self._llm_cache:
            logger.debug(f"Returning cached LLM instance: {cache_key}")
            return self._llm_cache[cache_key]
        
        config = MODEL_CONFIGS[model_name]
        provider = config["provider"]
        model_class = config["class"]
        
        # Check if model class is available
        if model_class is None:
            required_package = config.get("requires_package", "unknown")
            raise ValueError(
                f"Model '{model_name}' requires the '{required_package}' package. "
                f"Please install it using: pip install {required_package}"
            )
        
        # Get API key (not needed for Ollama)
        if provider == "openai":
            api_key = api_key or settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
                )
        elif provider == "anthropic":
            api_key = api_key or settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
                )
        elif provider == "google":
            api_key = api_key or settings.GOOGLE_API_KEY
            if not api_key:
                raise ValueError(
                    "Google API key is required. Set GOOGLE_API_KEY environment variable."
                )
        elif provider == "ollama":
            # Ollama doesn't need API key, but may need base_url
            if base_url is None:
                base_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Get model parameters
        temp = temperature if temperature is not None else (
            settings.CHAT_MODEL_TEMPERATURE if hasattr(settings, 'CHAT_MODEL_TEMPERATURE') 
            else config["default_temperature"]
        )
        max_toks = max_tokens if max_tokens is not None else (
            settings.CHAT_MODEL_MAX_TOKENS if hasattr(settings, 'CHAT_MODEL_MAX_TOKENS')
            else config["default_max_tokens"]
        )
        
        # Create model instance
        model_kwargs = {
            "model": config["model_name"],
            "temperature": temp,
        }
        
        if provider == "openai":
            model_kwargs["openai_api_key"] = api_key
            model_kwargs["max_tokens"] = max_toks
        elif provider == "anthropic":
            model_kwargs["anthropic_api_key"] = api_key
            model_kwargs["max_tokens"] = max_toks
        elif provider == "google":
            model_kwargs["google_api_key"] = api_key
            model_kwargs["max_output_tokens"] = max_toks
        elif provider == "ollama":
            model_kwargs["base_url"] = base_url
            if max_toks:
                model_kwargs["num_predict"] = max_toks
        
        try:
            llm = model_class(**model_kwargs)
            logger.info(
                f"Created {provider} model: {config['model_name']} "
                f"(temperature={temp}, max_tokens={max_toks})"
            )
            
            # Cache the instance if requested
            if use_cache:
                self._llm_cache[cache_key] = llm
            
            return llm
        except Exception as e:
            logger.error(f"Error creating LLM model: {e}", exc_info=True)
            raise
    
    def get_llm_for_use_case(
        self,
        use_case: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Get an LLM instance for a specific use case.
        Different use cases can use different models or configurations.
        
        Args:
            use_case: Name of the use case (e.g., "qa", "summarization", "classification")
            model_name: Optional model name override for this use case
            **kwargs: Additional parameters passed to get_llm()
            
        Returns:
            BaseChatModel instance configured for the use case
        """
        # Use case specific configurations
        use_case_configs = {
            "qa": {
                "temperature": 0.3,  # Lower temperature for more focused answers
                "max_tokens": 1000,
            },
            "summarization": {
                "temperature": 0.5,
                "max_tokens": 500,
            },
            "classification": {
                "temperature": 0.1,  # Very low for consistent classification
                "max_tokens": 100,
            },
            "creative": {
                "temperature": 0.9,  # Higher for creative tasks
                "max_tokens": 2000,
            },
            "default": {
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        }
        
        config = use_case_configs.get(use_case, use_case_configs["default"])
        
        # Merge use case config with provided kwargs
        merged_kwargs = {**config, **kwargs}
        if model_name:
            merged_kwargs["model_name"] = model_name
        
        # Use use_case in cache key to separate instances
        cache_key = f"{use_case}_{merged_kwargs.get('model_name', 'default')}"
        merged_kwargs["cache_key"] = cache_key
        
        return self.get_llm(**merged_kwargs)
    
    def set_default_model(self, model_name: str):
        """
        Set the default model name for this manager.
        
        Args:
            model_name: Name of the model to use as default
        """
        if model_name not in MODEL_CONFIGS:
            available_models = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Available models: {available_models}"
            )
        self._default_model = model_name
        logger.info(f"Set default model to: {model_name}")
    
    def clear_cache(self):
        """Clear the LLM instance cache."""
        self._llm_cache.clear()
        logger.info("Cleared LLM cache")
    
    def get_cached_instances(self) -> Dict[str, str]:
        """
        Get information about cached LLM instances.
        
        Returns:
            Dictionary mapping cache keys to model names
        """
        return {
            key: str(type(llm).__name__) 
            for key, llm in self._llm_cache.items()
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of available model names
        """
        return list(MODEL_CONFIGS.keys())


# Global LLM manager instance
_global_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """
    Get the global LLM manager instance (singleton pattern).
    
    Returns:
        Global LLMManager instance
    """
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = LLMManager()
    return _global_llm_manager


# Convenience function for backward compatibility
def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseChatModel:
    """
    Convenience function to get an LLM instance using the global manager.
    
    Args:
        model_name: Name of the model to use (default: from settings)
        temperature: Temperature for the model (default: from settings or model default)
        max_tokens: Maximum tokens for the model (default: from settings or model default)
        api_key: API key for the model provider (optional, uses env vars if not provided)
        base_url: Base URL for the model API (optional, mainly for Ollama)
        
    Returns:
        BaseChatModel instance
    """
    return get_llm_manager().get_llm(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url
    )


# Convenience function for getting available models
def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of available model names
    """
    return LLMManager.get_available_models()

