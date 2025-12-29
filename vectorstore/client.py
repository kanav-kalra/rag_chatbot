"""
Vector Store Manager - Global vector store storage
"""
from typing import Optional
from langchain_community.vectorstores import InMemoryVectorStore

# Global vector store instance
_vector_store: Optional[InMemoryVectorStore] = None


def set_vector_store(vector_store: InMemoryVectorStore) -> None:
    """
    Set the global vector store instance.
    
    Args:
        vector_store: InMemoryVectorStore instance to store globally
    """
    global _vector_store
    _vector_store = vector_store


def get_vector_store() -> Optional[InMemoryVectorStore]:
    """
    Get the global vector store instance.
    
    Returns:
        InMemoryVectorStore instance if available, None otherwise
    """
    return _vector_store


def is_vector_store_available() -> bool:
    """
    Check if the vector store is available.
    
    Returns:
        True if vector store is available, False otherwise
    """
    return _vector_store is not None

