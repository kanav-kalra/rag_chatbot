"""
Vector Store Manager - Global vector store storage
Supports both InMemoryVectorStore and ChromaDB
"""
from typing import Optional, Union
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import InMemoryVectorStore, Chroma

# Global vector store instance (can be InMemoryVectorStore or Chroma)
_vector_store: Optional[VectorStore] = None


def set_vector_store(vector_store: Union[InMemoryVectorStore, Chroma]) -> None:
    """
    Set the global vector store instance.
    
    Args:
        vector_store: VectorStore instance (InMemoryVectorStore or Chroma) to store globally
    """
    global _vector_store
    _vector_store = vector_store


def get_vector_store() -> Optional[VectorStore]:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore instance (InMemoryVectorStore or Chroma) if available, None otherwise
    """
    return _vector_store


def is_vector_store_available() -> bool:
    """
    Check if the vector store is available.
    
    Returns:
        True if vector store is available, False otherwise
    """
    return _vector_store is not None

