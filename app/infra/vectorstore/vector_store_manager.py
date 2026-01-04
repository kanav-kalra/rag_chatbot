"""
Vector Store Manager - Generic vector store management for different chatbot types
Supports loading and caching vector stores (ChromaDB, InMemory, etc.) per chatbot type
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
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

from app.core.config import settings
from app.core.logging import logger

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


def get_vector_store_config(chatbot_type: str) -> Dict[str, str]:
    """
    Get vector store configuration for a chatbot type.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        
    Returns:
        Dictionary with vector store configuration:
            - persist_dir: Directory path for ChromaDB persistence
            - collection_name: ChromaDB collection name
            - store_type: Type of vector store ("chroma" or "memory")
            - embedding_provider: Embedding provider ("openai", "google", or "auto")
            - embedding_model: Embedding model name (empty = use provider default)
    """
    # Default configuration
    config = {
        "persist_dir": f"./chroma_db/{chatbot_type}",
        "collection_name": chatbot_type,
        "store_type": "chroma",
        "embedding_provider": "openai",
        "embedding_model": ""
    }
    
    # Override with chatbot-specific settings if available
    # For HR chatbot
    if chatbot_type == "hr":
        config["persist_dir"] = getattr(settings, "HR_CHROMA_PERSIST_DIR", config["persist_dir"])
        config["collection_name"] = getattr(settings, "HR_CHROMA_COLLECTION_NAME", config["collection_name"])
        config["embedding_provider"] = getattr(settings, "HR_EMBEDDING_PROVIDER", "auto")
        config["embedding_model"] = getattr(settings, "HR_EMBEDDING_MODEL", "")
        
        # Auto-detect embedding provider based on chat model if set to "auto"
        if config["embedding_provider"] == "auto":
            hr_chat_model = getattr(settings, "HR_CHAT_MODEL", "")
            if hr_chat_model and hr_chat_model.startswith("gemini"):
                config["embedding_provider"] = "google"
            else:
                config["embedding_provider"] = "openai"
    
    # Future: Add support for other chatbot types
    # elif chatbot_type == "support":
    #     config["persist_dir"] = getattr(settings, "SUPPORT_CHROMA_PERSIST_DIR", config["persist_dir"])
    #     config["collection_name"] = getattr(settings, "SUPPORT_CHROMA_COLLECTION_NAME", config["collection_name"])
    
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
                f"  1. Set HR_EMBEDDING_PROVIDER={likely_provider} in your environment variables, or\n"
                f"  2. Recreate the vector store with: python jobs/create_hr_vectorstore.py --embedding-provider {embedding_provider} --clear-existing"
            )
    except RuntimeError:
        # Re-raise our custom error
        raise
    except Exception as e:
        # If dimension check fails, log warning but continue
        # The actual error will be caught when trying to use the vector store
        logger.warning(f"Could not verify embedding dimensions: {e}")
    
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
                    f"  1. Set HR_EMBEDDING_PROVIDER={likely_provider} in your environment variables, or\n"
                    f"  2. Recreate the vector store with: python jobs/create_hr_vectorstore.py --embedding-provider {embedding_provider} --clear-existing"
                ) from e
        raise
    
    # Verify the collection has data
    try:
        count = vector_store._collection.count()
        if count == 0:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' exists but is empty. "
                f"Please run the vector store creation job to populate it."
            )
        logger.info(
            f"Loaded ChromaDB vector store with {count} document chunks "
            f"(embedding provider: {embedding_provider}, dimension: {actual_dimension})"
        )
    except Exception as e:
        logger.warning(f"Could not verify collection count: {e}")
    
    return vector_store


def get_vector_store(chatbot_type: str) -> VectorStore:
    """
    Get or load the vector store for a chatbot type.
    Uses lazy loading - loads on first access and caches the result.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "default")
        
    Returns:
        VectorStore instance (Chroma or InMemory)
        
    Raises:
        FileNotFoundError: If ChromaDB persist directory doesn't exist
        ValueError: If collection is empty
        RuntimeError: If vector store loading fails
    """
    global _vector_stores
    
    with _vector_stores_lock:
        # Check cache first
        if chatbot_type in _vector_stores:
            return _vector_stores[chatbot_type]
        
        try:
            # Get configuration for this chatbot type
            config = get_vector_store_config(chatbot_type)
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

