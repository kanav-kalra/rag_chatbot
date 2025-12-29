"""
Vector Store Manager - Generic vector store management for different chatbot types
Supports loading and caching vector stores (ChromaDB, InMemory, etc.) per chatbot type
"""
import sys
from pathlib import Path
from typing import Optional, Dict
from threading import Lock

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

from app.core.config import settings
from app.core.logging import logger


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
    """
    # Default configuration
    config = {
        "persist_dir": f"./chroma_db/{chatbot_type}",
        "collection_name": chatbot_type,
        "store_type": "chroma"
    }
    
    # Override with chatbot-specific settings if available
    # For HR chatbot
    if chatbot_type == "hr":
        config["persist_dir"] = getattr(settings, "HR_CHROMA_PERSIST_DIR", config["persist_dir"])
        config["collection_name"] = getattr(settings, "HR_CHROMA_COLLECTION_NAME", config["collection_name"])
    
    # Future: Add support for other chatbot types
    # elif chatbot_type == "support":
    #     config["persist_dir"] = getattr(settings, "SUPPORT_CHROMA_PERSIST_DIR", config["persist_dir"])
    #     config["collection_name"] = getattr(settings, "SUPPORT_CHROMA_COLLECTION_NAME", config["collection_name"])
    
    return config


def load_chroma_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Chroma:
    """
    Load a ChromaDB vector store.
    
    Args:
        persist_directory: Directory path where ChromaDB persists the data
        collection_name: Name of the ChromaDB collection
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        Chroma vector store instance
        
    Raises:
        FileNotFoundError: If the persist_directory does not exist
        ValueError: If the collection does not exist or is empty
    """
    from langchain_openai import OpenAIEmbeddings
    
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: {persist_directory}. "
            f"Please run the vector store creation job first."
        )
    
    embedding_model = embedding_model or "text-embedding-3-small"
    api_key = openai_api_key or (settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None)
    embeddings_kwargs = {}
    if api_key:
        embeddings_kwargs["openai_api_key"] = api_key
        
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        **embeddings_kwargs
    )
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Verify the collection has data
    try:
        count = vector_store._collection.count()
        if count == 0:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' exists but is empty. "
                f"Please run the vector store creation job to populate it."
            )
        logger.info(f"Loaded ChromaDB vector store with {count} document chunks")
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
                
                logger.info(
                    f"Loading {chatbot_type} ChromaDB vector store from: {persist_dir}"
                )
                vector_store = load_chroma_vector_store(
                    persist_directory=persist_dir,
                    collection_name=collection_name
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

