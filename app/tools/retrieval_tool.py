"""
Document Retrieval Tool - Retrieves relevant documents from vector store based on query
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.core.logger import logger
from app.tools.vector_store_manager import get_vector_store, is_vector_store_available
# Call memory_builder to build and initialize the in-memory vector store
from app.document_loader.memory_builder import build_memory_from_pdfs
# Set up global vector store for tools
from app.tools.vector_store_manager import set_vector_store

@tool(response_format="content_and_artifact")
def retrieve_documents(
    query: str,
    k: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant documents from the vector store based on a query.
    This tool searches through the document vector store to find the most relevant documents
    matching the given query.
    
    Args:
        query: The search query string
        k: Number of documents to retrieve (default: 4)
        
    Returns:
        Tuple of (content, artifact):
            - content: Formatted string with retrieved documents
            - artifact: List of dictionaries with detailed document information (rank, content, metadata)
            
    Raises:
        ValueError: If vector store is not available
        RuntimeError: If retrieval fails
    """
    try:
        # Get vector store from global manager
        vs = get_vector_store()
        
        if vs is None:
            error_msg = "Vector store is not available. Please initialize it first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Perform similarity search
        logger.info(f"Retrieving documents for query: '{query}' (k={k})")
        documents = vs.similarity_search(query, k=k)
        
        # Format results as artifact (detailed structured data)
        artifact = []
        content_parts = []
        
        for i, doc in enumerate(documents):
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            artifact.append(result)
            
            # Format content string
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            file_name = doc.metadata.get("file_name", "Unknown")
            
            content_parts.append(
                f"Document {i + 1} (Source: {file_name}, Page: {page}):\n"
                f"{doc.page_content}\n"
            )
        
        # Combine content parts
        content = "\n---\n\n".join(content_parts)
        
        logger.info(f"Retrieved {len(artifact)} documents for query: '{query}'")
        
        # Return tuple (content, artifact) for response_format="content_and_artifact"
        return (content, artifact)
        
    except Exception as e:
        error_msg = f"Error retrieving documents: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def retrieve_documents_with_scores(
    query: str,
    k: int = 4,
    vector_store=None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents with similarity scores from the vector store.
    
    Args:
        query: The search query string
        k: Number of documents to retrieve (default: 4)
        vector_store: Optional vector store instance. If not provided, uses the global vector store.
        
    Returns:
        List of dictionaries containing:
            - content: The document content
            - metadata: Document metadata (source, page, etc.)
            - score: Similarity score
            
    Raises:
        ValueError: If vector store is not available and not provided
        RuntimeError: If retrieval fails
    """
    try:
        # Get vector store (use provided one or global one)
        vs = vector_store if vector_store is not None else get_vector_store()
        
        if vs is None:
            error_msg = "Vector store is not available. Please initialize it first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Perform similarity search with scores
        logger.info(f"Retrieving documents with scores for query: '{query}' (k={k})")
        documents_with_scores = vs.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for i, (doc, score) in enumerate(documents_with_scores):
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} documents with scores for query: '{query}'")
        return results
        
    except Exception as e:
        error_msg = f"Error retrieving documents with scores: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def retrieve_documents_as_string(
    query: str,
    k: int = 4,
    separator: str = "\n\n---\n\n",
    vector_store=None
) -> str:
    """
    Retrieve relevant documents and return them as a formatted string.
    This is a helper function that wraps the tool for backward compatibility.
    
    Args:
        query: The search query string
        k: Number of documents to retrieve (default: 4)
        separator: String to separate documents (default: "\n\n---\n\n")
        vector_store: Optional vector store instance (for backward compatibility, not used in tool)
        
    Returns:
        Formatted string containing all retrieved documents
        
    Raises:
        ValueError: If vector store is not available
        RuntimeError: If retrieval fails
    """
    try:
        # Call the underlying function directly for simplicity
        content, artifact = retrieve_documents.func(query=query, k=k)
        
        # Return the content string
        return content
        
    except Exception as e:
        error_msg = f"Error formatting documents as string: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    vector_store = build_memory_from_pdfs()
    set_vector_store(vector_store)
    # Example usage
    from app.tools.vector_store_manager import is_vector_store_available
    
    if not is_vector_store_available():
        logger.warning("Vector store is not available. Please initialize it first.")
        logger.info("Example: from app.tools.vector_store_manager import set_vector_store")
        logger.info("         from app.document_loader.memory_builder import build_memory_from_pdfs")
        logger.info("         vs = build_memory_from_pdfs()")
        logger.info("         set_vector_store(vs)")
    else:
        try:
            query = "What is notice period of grade 8 employees?"
            logger.info(f"Testing retrieval with query: '{query}'")
            
            # Use the tool's underlying function directly for testing
            # The tool.invoke() might wrap the result differently
            content, artifact = retrieve_documents.func(query=query, k=3)
            
            logger.info(f"\nRetrieved {len(artifact)} documents:")
            logger.info(f"Content:\n{content[:500] if content else 'No content'}...")
            
            for doc in artifact:
                logger.info(f"\nRank {doc['rank']}:")
                logger.info(f"Content preview: {doc['content'][:200]}...")
                logger.info(f"Metadata: {doc['metadata']}")
                
        except Exception as e:
            logger.error(f"Error in example: {e}", exc_info=True)

