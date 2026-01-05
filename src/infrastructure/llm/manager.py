"""
LLM Manager - Factory class for creating and managing multiple LLM instances
Refactored to follow SOLID principles:
- Single Responsibility: Each class has one reason to change
- Open/Closed: Extensible via provider registry pattern
- Liskov Substitution: Abstractions allow interchangeable implementations
- Interface Segregation: Focused interfaces for specific responsibilities
- Dependency Inversion: Depend on abstractions, not concrete implementations
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Protocol

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
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

from src.shared.config.settings import settings
from src.shared.config.logging import logger


# ============================================================================
# Protocols (Abstractions) - Interface Segregation & Dependency Inversion
# ============================================================================

class APIKeyProvider(Protocol):
    """Protocol for API key providers - allows different implementations"""
    
    def get_api_key(self, provider: str, override_key: Optional[str] = None) -> Optional[str]:
        """Get API key for a provider"""
        ...
    
    def validate_api_key(self, provider: str, api_key: Optional[str]) -> None:
        """Validate that API key exists for providers that require it"""
        ...


class ModelConfigRepository(Protocol):
    """Protocol for model configuration repositories"""
    
    def get_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        ...
    
    def has_model(self, model_name: str) -> bool:
        """Check if model exists"""
        ...
    
    def list_models(self) -> List[str]:
        """List all available models"""
        ...


# ============================================================================
# Model Configuration Repository - Single Responsibility
# ============================================================================

class DefaultModelConfigRepository:
    """
    Manages model configurations.
    Single Responsibility: Only responsible for storing and retrieving model configs.
    """
    
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = self._load_default_configs()
    
    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load default model configurations"""
        return {
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
            "gemini-2.5-flash": {
                "provider": "google",
                "model_name": "gemini-2.5-flash",
                "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
                "requires_package": "langchain-google-genai",
            },
            "gemini-2.5-flash-lite": {
                "provider": "google",
                "model_name": "gemini-2.5-flash-lite",
                "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
                "requires_package": "langchain-google-genai",
            },
            "gemini-2.5-pro": {
                "provider": "google",
                "model_name": "gemini-2.5-pro",
                "class": ChatGoogleGenerativeAI if GOOGLE_AVAILABLE else None,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
                "requires_package": "langchain-google-genai",
            },
        }
    
    def get_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        if not self.has_model(model_name):
            available_models = ", ".join(self.list_models())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Available models: {available_models}"
            )
        return self._configs[model_name]
    
    def has_model(self, model_name: str) -> bool:
        """Check if model exists"""
        return model_name in self._configs
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self._configs.keys())
    
    def register_model(self, model_name: str, config: Dict[str, Any]) -> None:
        """Register a new model configuration (Open/Closed Principle)"""
        self._configs[model_name] = config
        logger.info(f"Registered model: {model_name}")


# ============================================================================
# API Key Provider - Single Responsibility & Dependency Inversion
# ============================================================================

class SettingsAPIKeyProvider:
    """
    Provides API keys from settings.
    Single Responsibility: Only responsible for retrieving API keys.
    """
    
    def get_api_key(self, provider: str, override_key: Optional[str] = None) -> Optional[str]:
        """Get API key for a provider"""
        if override_key:
            return override_key
        
        provider_key_map = {
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
            "google": settings.GOOGLE_API_KEY,
            "ollama": None,  # Ollama doesn't need API key
        }
        
        return provider_key_map.get(provider)
    
    def validate_api_key(self, provider: str, api_key: Optional[str]) -> None:
        """Validate that API key exists for providers that require it"""
        if provider == "ollama":
            return  # Ollama doesn't need API key
        
        if not api_key:
            provider_names = {
                "openai": "OpenAI",
                "anthropic": "Anthropic",
                "google": "Google",
            }
            provider_name = provider_names.get(provider, provider)
            raise ValueError(
                f"{provider_name} API key is required. "
                f"Set {provider.upper()}_API_KEY environment variable."
            )


# ============================================================================
# Model Provider Factory - Open/Closed Principle & Dependency Inversion
# ============================================================================

class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    Open/Closed Principle: New providers can be added by extending this class.
    """
    
    @abstractmethod
    def create_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        """Create a model instance"""
        pass


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models"""
    
    def create_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens,
        )


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models"""
    
    def create_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        if not ANTHROPIC_AVAILABLE:
            raise ValueError(
                "Anthropic models require the 'langchain-anthropic' package. "
                "Please install it using: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            anthropic_api_key=api_key,
            max_tokens=max_tokens,
        )


class GoogleProvider(ModelProvider):
    """Provider for Google models"""
    
    def create_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        if not GOOGLE_AVAILABLE:
            raise ValueError(
                "Google models require the 'langchain-google-genai' package. "
                "Please install it using: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
        )


class OllamaProvider(ModelProvider):
    """Provider for Ollama models"""
    
    def create_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        if not OLLAMA_AVAILABLE:
            raise ValueError(
                "Ollama models require the 'langchain-ollama' package. "
                "Please install it using: pip install langchain-ollama"
            )
        default_base_url = base_url or getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "base_url": default_base_url,
        }
        if max_tokens:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(**kwargs)


class ModelProviderFactory:
    """
    Factory for creating model providers.
    Open/Closed Principle: New providers can be registered without modifying this class.
    Dependency Inversion: Depends on ModelProvider abstraction.
    """
    
    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
            "ollama": OllamaProvider(),
        }
    
    def get_provider(self, provider_name: str) -> ModelProvider:
        """Get a provider by name"""
        if provider_name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' is not supported. "
                f"Available providers: {available}"
            )
        return self._providers[provider_name]
    
    def register_provider(self, provider_name: str, provider: ModelProvider) -> None:
        """Register a new provider (Open/Closed Principle)"""
        self._providers[provider_name] = provider
        logger.info(f"Registered provider: {provider_name}")


# ============================================================================
# Use Case Configuration - Single Responsibility
# ============================================================================

class UseCaseConfig:
    """
    Manages use case specific configurations.
    Single Responsibility: Only responsible for use case configurations.
    """
    
    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {
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
    
    def get_config(self, use_case: str) -> Dict[str, Any]:
        """Get configuration for a use case"""
        return self._configs.get(use_case, self._configs["default"])
    
    def register_use_case(self, use_case: str, config: Dict[str, Any]) -> None:
        """Register a new use case configuration"""
        self._configs[use_case] = config
        logger.info(f"Registered use case: {use_case}")


# ============================================================================
# LLM Cache - Single Responsibility
# ============================================================================

class LLMCache:
    """
    Manages caching of LLM instances.
    Single Responsibility: Only responsible for caching operations.
    """
    
    def __init__(self):
        self._cache: Dict[str, BaseChatModel] = {}
    
    def get(self, key: str) -> Optional[BaseChatModel]:
        """Get cached LLM instance"""
        return self._cache.get(key)
    
    def set(self, key: str, llm: BaseChatModel) -> None:
        """Cache an LLM instance"""
        self._cache[key] = llm
        logger.debug(f"Cached LLM instance: {key}")
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self._cache
    
    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        logger.info("Cleared LLM cache")
    
    def get_cached_instances(self) -> Dict[str, str]:
        """Get information about cached instances"""
        return {
            key: str(type(llm).__name__)
            for key, llm in self._cache.items()
        }
    
    def generate_cache_key(
        self,
        model_name: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        base_url: Optional[str],
    ) -> str:
        """Generate a cache key from parameters"""
        return f"{model_name}_{temperature}_{max_tokens}_{base_url}"


# ============================================================================
# LLM Manager - Orchestrator (Single Responsibility)
# ============================================================================

class LLMManager:
    """
    Manager class for creating and managing multiple LLM instances.
    Single Responsibility: Orchestrates components to provide LLM instances.
    Dependency Inversion: Depends on abstractions (repositories, providers, etc.)
    """
    
    def __init__(
        self,
        config_repository: Optional[ModelConfigRepository] = None,
        api_key_provider: Optional[APIKeyProvider] = None,
        provider_factory: Optional[ModelProviderFactory] = None,
        use_case_config: Optional[UseCaseConfig] = None,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the LLM Manager with dependencies.
        
        Args:
            config_repository: Repository for model configurations (default: DefaultModelConfigRepository)
            api_key_provider: Provider for API keys (default: SettingsAPIKeyProvider)
            provider_factory: Factory for model providers (default: ModelProviderFactory)
            use_case_config: Configuration for use cases (default: UseCaseConfig)
            cache: Cache for LLM instances (default: LLMCache)
        """
        self._config_repository = config_repository or DefaultModelConfigRepository()
        self._api_key_provider = api_key_provider or SettingsAPIKeyProvider()
        self._provider_factory = provider_factory or ModelProviderFactory()
        self._use_case_config = use_case_config or UseCaseConfig()
        self._cache = cache or LLMCache()
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
            cache_key: Custom cache key (default: auto-generated)
            
        Returns:
            BaseChatModel instance
            
        Raises:
            ValueError: If model is not supported or API key is missing
        """
        # Resolve model name
        model_name = model_name or self._default_model or settings.CHAT_MODEL
        
        # Get model configuration
        config = self._config_repository.get_config(model_name)
        provider_name = config["provider"]
        model_class = config["class"]
        
        # Check if model class is available
        if model_class is None:
            required_package = config.get("requires_package", "unknown")
            raise ValueError(
                f"Model '{model_name}' requires the '{required_package}' package. "
                f"Please install it using: pip install {required_package}"
            )
        
        # Get API key
        resolved_api_key = self._api_key_provider.get_api_key(provider_name, api_key)
        self._api_key_provider.validate_api_key(provider_name, resolved_api_key)
        
        # Get model parameters with fallback chain
        temp = temperature if temperature is not None else (
            settings.CHAT_MODEL_TEMPERATURE if hasattr(settings, 'CHAT_MODEL_TEMPERATURE')
            else config["default_temperature"]
        )
        max_toks = max_tokens if max_tokens is not None else (
            settings.CHAT_MODEL_MAX_TOKENS if hasattr(settings, 'CHAT_MODEL_MAX_TOKENS')
            else config["default_max_tokens"]
        )
        
        # Generate cache key
        if cache_key is None:
            cache_key = self._cache.generate_cache_key(model_name, temp, max_toks, base_url)
        
        # Return cached instance if available
        if use_cache and self._cache.has(cache_key):
            logger.debug(f"Returning cached LLM instance: {cache_key}")
            return self._cache.get(cache_key)
        
        # Create model instance using provider factory
        provider = self._provider_factory.get_provider(provider_name)
        try:
            llm = provider.create_model(
                model_name=config["model_name"],
                temperature=temp,
                max_tokens=max_toks,
                api_key=resolved_api_key,
                base_url=base_url,
            )
            logger.info(
                f"Created {provider_name} model: {config['model_name']} "
                f"(temperature={temp}, max_tokens={max_toks})"
            )
            
            # Cache the instance if requested
            if use_cache:
                self._cache.set(cache_key, llm)
            
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
        # Get use case configuration
        use_case_config = self._use_case_config.get_config(use_case)
        
        # Merge use case config with provided kwargs
        merged_kwargs = {**use_case_config, **kwargs}
        if model_name:
            merged_kwargs["model_name"] = model_name
        
        # Use use_case in cache key to separate instances
        cache_key = f"{use_case}_{merged_kwargs.get('model_name', 'default')}"
        merged_kwargs["cache_key"] = cache_key
        
        return self.get_llm(**merged_kwargs)
    
    def set_default_model(self, model_name: str) -> None:
        """
        Set the default model name for this manager.
        
        Args:
            model_name: Name of the model to use as default
        """
        if not self._config_repository.has_model(model_name):
            available_models = ", ".join(self._config_repository.list_models())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Available models: {available_models}"
            )
        self._default_model = model_name
        logger.info(f"Set default model to: {model_name}")
    
    def clear_cache(self) -> None:
        """Clear the LLM instance cache."""
        self._cache.clear()
    
    def get_cached_instances(self) -> Dict[str, str]:
        """
        Get information about cached LLM instances.
        
        Returns:
            Dictionary mapping cache keys to model names
        """
        return self._cache.get_cached_instances()
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of available model names
        """
        return self._config_repository.list_models()


# ============================================================================
# Global Instance - Singleton Pattern
# ============================================================================

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
