"""
Vector Store Manager - Generic vector store management for different chatbot types
Supports loading and caching vector stores (ChromaDB, InMemory, etc.) per chatbot type

Supports multiple embedding providers/models by auto-generating collection names
that include provider and model information (e.g., hr_chatbot_openai_text-embedding-3-small)
"""
import re
import sys
from pathlib import Path
from typing import Optional, Dict
from threading import Lock

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

from src.shared.config.settings import settings
from src.shared.config.logging import logger
from src.domain.chatbot.core.config import ChatbotConfigManager, ConfigKeys

# Optional imports for embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    OpenAIEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GOOGLE_EMBEDDINGS_AVAILABLE = False
    GoogleGenerativeAIEmbeddings = None


# Global cache for vector stores per chatbot type
_vector_stores: Dict[str, VectorStore] = {}
_vector_stores_lock = Lock()


def get_default_embedding_model(provider: str, model: Optional[str] = None) -> str:
    """
    Get the actual embedding model name that will be used, applying provider defaults.
    
    Args:
        provider: Embedding provider ("openai" or "google")
        model: Optional model name override
        
    Returns:
        Model name to use (with provider defaults applied)
    """
    if provider == "openai":
        return model or "text-embedding-3-small"
    elif provider == "google":
        return model or "models/embedding-001"
    else:
        return model or "default"


def generate_collection_name(
    base_name: str,
    provider: str,
    model: Optional[str] = None
) -> str:
    """
    Generate a collection name that includes provider and model information.
    This ensures different embeddings are stored in separate collections.
    Used by both create_vectorstore.py and vector_store_manager.py.
    
    Args:
        base_name: Base collection name (e.g., "hr_chatbot")
        provider: Embedding provider (e.g., "openai", "google")
        model: Embedding model name (e.g., "text-embedding-3-small", "models/embedding-001")
        
    Returns:
        Collection name like: "hr_chatbot_openai_text-embedding-3-small"
    """
    # Create a slug from the model name (remove special chars, replace / with _)
    if model:
        # Remove "models/" prefix if present (for Google models)
        model_slug = model.replace("models/", "")
        # Replace special characters with underscores
        model_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', model_slug)
        # Remove multiple consecutive underscores
        model_slug = re.sub(r'_+', '_', model_slug).strip('_')
        return f"{base_name}_{provider}_{model_slug}"
    else:
        # Use provider default model indicator
        return f"{base_name}_{provider}_default"


def get_vector_store_config(
    chatbot_type: str,
    config_manager: Optional[ChatbotConfigManager] = None
) -> Dict[str, str]:
    """
    Get vector store configuration for a chatbot type.
    
    Requires config_manager to be provided. Fails if config cannot be loaded.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default", "support")
        config_manager: ChatbotConfigManager instance with loaded config (required)
        
    Returns:
        Dictionary with vector store configuration:
            - persist_dir: Directory path for ChromaDB persistence
            - collection_name: ChromaDB collection name
            - store_type: Type of vector store ("chroma" or "memory")
            - embedding_provider: Embedding provider ("openai" or "google", required)
            - embedding_model: Embedding model name (empty = use provider default)
            
    Raises:
        ValueError: If config_manager is not provided or config cannot be loaded
    """
    if config_manager is None:
        raise ValueError(
            f"config_manager is required for chatbot type '{chatbot_type}'. "
            f"Vector store configuration must be loaded from config file."
        )
    
    if not config_manager.has_config():
        raise ValueError(
            f"Config manager for '{chatbot_type}' has no loaded config. "
            f"Ensure the config file exists and was loaded successfully."
        )
    
    # Read vector store config from YAML
    persist_dir = config_manager.get("vector_store.persist_dir")
    collection_name = config_manager.get("vector_store.collection_name")
    embedding_provider = config_manager.get("vector_store.embedding_provider")
    embedding_model = config_manager.get("vector_store.embedding_model")
    
    if not persist_dir:
        raise ValueError(
            f"vector_store.persist_dir is required in config for '{chatbot_type}'"
        )
    if not collection_name:
        raise ValueError(
            f"vector_store.collection_name is required in config for '{chatbot_type}'"
        )
    if embedding_provider is None:
        raise ValueError(
            f"vector_store.embedding_provider is required in config for '{chatbot_type}'"
        )
    
    # Validate embedding_provider value
    valid_providers = ["openai", "google"]
    if embedding_provider not in valid_providers:
        raise ValueError(
            f"Invalid embedding_provider '{embedding_provider}' for '{chatbot_type}'. "
            f"Must be one of: {', '.join(valid_providers)}. "
            f"'auto' is not supported - set embedding_provider explicitly."
        )
    
    logger.debug(
        f"Reading vector store config for {chatbot_type}: "
        f"persist_dir={persist_dir}, collection_name={collection_name}, "
        f"embedding_provider={embedding_provider}, embedding_model={embedding_model}"
    )
    
    config = {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "store_type": "chroma",
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model or ""
    }
    
    # Always auto-generate collection name with embedding suffix
    # This ensures different embedding providers/models use separate collections,
    # allowing users to switch providers without recreating embeddings.
    # The explicit collection_name from config is treated as a base name.
    base_collection_name = collection_name
    
    # Get the actual model name that will be used
    actual_model = get_default_embedding_model(
        config["embedding_provider"],
        config["embedding_model"]
    )
    
    # Always generate collection name with embedding suffix
    # This allows multiple embedding providers to coexist
    config["collection_name"] = generate_collection_name(
        base_collection_name,
        config["embedding_provider"],
        actual_model
    )
    logger.info(
        f"Auto-generated collection name with embedding suffix: {config['collection_name']} "
        f"(base: {base_collection_name}, provider: {config['embedding_provider']}, model: {actual_model})"
    )
    
    logger.debug(f"Loaded vector store config from config_manager for {chatbot_type}")
    
    return config


def create_embeddings(
    provider: str = "openai",
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Embeddings:
    """
    Create embeddings instance based on provider.
    
    Args:
        provider: Embedding provider ("openai" or "google")
        embedding_model: Embedding model name (default: provider-specific default)
        api_key: API key for the provider (if not provided, uses env vars)
        
    Returns:
        Embeddings instance
        
    Raises:
        ValueError: If provider is not supported or required package is missing
    """
    if provider == "openai":
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            raise ValueError(
                "OpenAI embeddings require 'langchain-openai' package. "
                "Please install it using: pip install langchain-openai"
            )
        model = embedding_model or "text-embedding-3-small"
        api_key = api_key or (settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None)
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key
        return OpenAIEmbeddings(model=model, **embeddings_kwargs)
    
    elif provider == "google":
        if not GOOGLE_EMBEDDINGS_AVAILABLE:
            raise ValueError(
                "Google embeddings require 'langchain-google-genai' package. "
                "Please install it using: pip install langchain-google-genai"
            )
        api_key = api_key or (settings.GOOGLE_API_KEY if settings.GOOGLE_API_KEY else None)
        if not api_key:
            raise ValueError(
                "Google API key is required for Google embeddings. "
                "Set GOOGLE_API_KEY environment variable."
            )
        # Google embeddings model parameter is required
        # Default to "models/embedding-001" or use newer "models/text-embedding-004" if available
        model = embedding_model or "models/embedding-001"
        embeddings_kwargs = {
            "google_api_key": api_key,
            "model": model
        }
        return GoogleGenerativeAIEmbeddings(**embeddings_kwargs)
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported: 'openai', 'google'")


def load_chroma_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Chroma:
    """
    Load a ChromaDB vector store.
    
    Args:
        persist_directory: Directory path where ChromaDB persists the data
        collection_name: Name of the ChromaDB collection
        embedding_provider: Embedding provider ("openai" or "google", default: "openai")
        embedding_model: Embedding model name (default: provider-specific default)
        api_key: API key for the embedding provider (if not provided, uses env vars)
        
    Returns:
        Chroma vector store instance
        
    Raises:
        FileNotFoundError: If the persist_directory does not exist
        ValueError: If the collection does not exist or is empty, or provider is not supported
        RuntimeError: If embedding dimensions don't match the collection
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: {persist_directory}. "
            f"Please run the vector store creation job first."
        )
    
    # Create embeddings based on provider
    embeddings = create_embeddings(
        provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key
    )
    
    # Check embedding dimension before loading
    try:
        # Get the actual embedding dimension from the embeddings
        test_embedding = embeddings.embed_query("test")
        actual_dimension = len(test_embedding)
        
        # Try to get the expected dimension from existing collection metadata
        # We'll use a temporary Chroma instance to check metadata
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("chromadb package is required for dimension validation")
        client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            existing_collection = client.get_collection(name=collection_name)
            # Get metadata from collection
            if hasattr(existing_collection, 'metadata') and existing_collection.metadata:
                expected_dimension = existing_collection.metadata.get('dimension')
            else:
                # Try to get dimension from collection info
                # If collection exists, try to peek at a sample embedding
                expected_dimension = None
                if existing_collection.count() > 0:
                    # Get a sample to check dimension
                    sample = existing_collection.peek(limit=1)
                    if sample and 'embeddings' in sample and len(sample['embeddings']) > 0:
                        expected_dimension = len(sample['embeddings'][0])
        except Exception:
            # Collection doesn't exist yet or can't access metadata
            expected_dimension = None
        
        # If we found an expected dimension and it doesn't match, raise error
        if expected_dimension is not None and expected_dimension != actual_dimension:
            # Determine which provider was likely used to create the collection
            if expected_dimension == 1536:
                likely_provider = "openai"
                likely_model = "text-embedding-3-small or text-embedding-ada-002"
            elif expected_dimension == 768:
                likely_provider = "google"
                likely_model = "models/embedding-001"
            else:
                likely_provider = "unknown"
                likely_model = "unknown"
            
            raise RuntimeError(
                f"Embedding dimension mismatch!\n"
                f"  Collection expects: {expected_dimension} dimensions (likely created with {likely_provider}: {likely_model})\n"
                f"  Current provider ({embedding_provider}) produces: {actual_dimension} dimensions\n\n"
                f"To fix this, either:\n"
                f"  1. Set vector_store.embedding_provider={likely_provider} in your chatbot config YAML file, or\n"
                f"  2. Recreate the vector store with the correct embedding provider"
            )
    except RuntimeError:
        # Re-raise our custom error
        raise
    except Exception as e:
        # If dimension check fails, log warning but continue
        # The actual error will be caught when trying to use the vector store
        logger.warning(f"Could not verify embedding dimensions: {e}")
    
    # Check if collection exists before trying to load it
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("chromadb package is required")
    
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        existing_collection = client.get_collection(name=collection_name)
        collection_exists = True
    except Exception:
        collection_exists = False
    
    if not collection_exists:
        # List available collections to help diagnose
        try:
            available_collections = client.list_collections()
            collections_with_data = [c for c in available_collections if c.count() > 0]
            
            error_msg = (
                f"ChromaDB collection '{collection_name}' does not exist in persist directory: {persist_directory}\n"
                f"Available collections:\n"
            )
            for coll in available_collections:
                coll_count = coll.count()
                error_msg += f"  - {coll.name}: {coll_count} documents\n"
            
            if collections_with_data:
                error_msg += (
                    f"\nCollections with data:\n"
                )
                for coll in collections_with_data:
                    error_msg += f"  - {coll.name}: {coll.count()} documents\n"
                error_msg += (
                    f"\nPossible solutions:\n"
                    f"  1. Re-index documents to create collection '{collection_name}'\n"
                    f"  2. Check if embedding_provider/model in config matches the indexed collection\n"
                )
            else:
                error_msg += "\nNo collections with data found. Please run the vector store creation job to populate it."
            
            raise ValueError(error_msg)
        except ValueError:
            # Re-raise our custom error
            raise
        except Exception as list_error:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' does not exist in persist directory: {persist_directory}. "
                f"Please run the vector store creation job to create it."
            ) from list_error
    
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        # Catch dimension mismatch errors from ChromaDB and provide helpful message
        error_msg = str(e)
        if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
            # Try to extract expected dimension from error
            dim_match = re.search(r'(\d+)', error_msg)
            if dim_match:
                expected_dim = dim_match.group(1)
                # Get actual dimension if not already calculated
                try:
                    test_embedding = embeddings.embed_query("test")
                    actual_dim = len(test_embedding)
                except Exception:
                    actual_dim = "unknown"
                
                if expected_dim == "1536":
                    likely_provider = "openai"
                elif expected_dim == "768":
                    likely_provider = "google"
                else:
                    likely_provider = "unknown"
                
                raise RuntimeError(
                    f"Embedding dimension mismatch!\n"
                    f"  Collection expects: {expected_dim} dimensions\n"
                    f"  Current provider ({embedding_provider}) produces: {actual_dim} dimensions\n\n"
                    f"To fix this, either:\n"
                    f"  1. Set vector_store.embedding_provider={likely_provider} in your chatbot config YAML file, or\n"
                    f"  2. Recreate the vector store with the correct embedding provider"
                ) from e
        raise
    
    # Verify the collection has data
    try:
        count = vector_store._collection.count()
        if count == 0:
            # List available collections to help diagnose the issue
            try:
                client = chromadb.PersistentClient(path=persist_directory)
                available_collections = client.list_collections()
                collection_names = [c.name for c in available_collections]
                collections_with_data = [c for c in available_collections if c.count() > 0]
                
                error_msg = (
                    f"ChromaDB collection '{collection_name}' exists but is empty (0 documents).\n"
                    f"Available collections in this persist directory:\n"
                )
                for coll in available_collections:
                    coll_count = coll.count()
                    error_msg += f"  - {coll.name}: {coll_count} documents\n"
                
                if collections_with_data:
                    error_msg += (
                        f"\nCollections with data:\n"
                    )
                    for coll in collections_with_data:
                        error_msg += f"  - {coll.name}: {coll.count()} documents\n"
                    error_msg += (
                        f"\nPossible solutions:\n"
                        f"  1. Re-index documents to create collection '{collection_name}'\n"
                        f"  2. Check if embedding_provider/model in config matches the indexed collection\n"
                        f"  3. Clear vector store cache if you re-indexed: clear_vector_store_cache('{persist_directory.split('/')[-1]}')\n"
                    )
                else:
                    error_msg += "\nNo collections with data found. Please run the vector store creation job to populate it."
                
                raise ValueError(error_msg)
            except Exception as list_error:
                # If we can't list collections, just raise the original error
                raise ValueError(
                    f"ChromaDB collection '{collection_name}' exists but is empty. "
                    f"Please run the vector store creation job to populate it."
                ) from list_error
        logger.info(
            f"Loaded ChromaDB vector store with {count} document chunks "
            f"(embedding provider: {embedding_provider}, dimension: {actual_dimension})"
        )
    except ValueError:
        # Re-raise ValueError (our custom error messages)
        raise
    except Exception as e:
        logger.warning(f"Could not verify collection count: {e}")
    
    return vector_store


def get_vector_store(
    chatbot_type: str,
    config_manager: Optional[ChatbotConfigManager] = None
) -> VectorStore:
    """
    Get or load the vector store for a chatbot type.
    Uses lazy loading - loads on first access and caches the result.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        config_manager: ChatbotConfigManager instance with loaded config (required).
                       Vector store config will be read from it.
        
    Returns:
        VectorStore instance (Chroma or InMemory)
        
    Raises:
        ValueError: If config_manager is not provided or config cannot be loaded
        FileNotFoundError: If ChromaDB persist directory doesn't exist
        ValueError: If collection is empty or not found
        RuntimeError: If vector store loading fails
    """
    if config_manager is None:
        raise ValueError(
            f"config_manager is required for chatbot type '{chatbot_type}'. "
            f"Cannot load vector store without configuration."
        )
    
    global _vector_stores
    
    with _vector_stores_lock:
        # Check cache first
        if chatbot_type in _vector_stores:
            return _vector_stores[chatbot_type]
        
        try:
            # Get configuration for this chatbot type
            config = get_vector_store_config(chatbot_type, config_manager=config_manager)
            store_type = config.get("store_type", "chroma")
            
            if store_type == "chroma":
                # Load ChromaDB vector store
                persist_dir = config["persist_dir"]
                collection_name = config["collection_name"]
                embedding_provider = config.get("embedding_provider", "openai")
                embedding_model = config.get("embedding_model") or None
                
                logger.info(
                    f"Loading {chatbot_type} ChromaDB vector store from: {persist_dir} "
                    f"(embedding provider: {embedding_provider})"
                )
                vector_store = load_chroma_vector_store(
                    persist_directory=persist_dir,
                    collection_name=collection_name,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model
                )
                logger.info(f"{chatbot_type} ChromaDB vector store loaded successfully")
            else:
                # Future: Support for other vector store types (InMemory, etc.)
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            # Cache the vector store
            _vector_stores[chatbot_type] = vector_store
            return vector_store
            
        except Exception as e:
            logger.error(
                f"Failed to load vector store for chatbot type '{chatbot_type}': {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to load vector store for chatbot type '{chatbot_type}': {str(e)}"
            ) from e


def clear_vector_store_cache(chatbot_type: Optional[str] = None) -> None:
    """
    Clear the vector store cache.
    
    Args:
        chatbot_type: If provided, clear cache for this chatbot type only.
                     If None, clear all cached vector stores.
    """
    global _vector_stores
    
    with _vector_stores_lock:
        if chatbot_type:
            if chatbot_type in _vector_stores:
                del _vector_stores[chatbot_type]
                logger.info(f"Cleared vector store cache for chatbot type: {chatbot_type}")
        else:
            _vector_stores.clear()
            logger.info("Cleared all vector store caches")


def is_vector_store_available(chatbot_type: str) -> bool:
    """
    Check if a vector store is available for a chatbot type.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        
    Returns:
        True if vector store is available, False otherwise
    """
    try:
        get_vector_store(chatbot_type)
        return True
    except Exception:
        return False

