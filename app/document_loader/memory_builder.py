"""
Memory Builder for loading documents, splitting them, and adding to InMemoryVectorStore
"""
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

from app.core.config import settings
from app.core.logger import logger
from app.document_loader.pdf_loader import load_pdf_documents
from app.document_loader.text_splitter import split_documents


def create_vector_store(
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> InMemoryVectorStore:
    """
    Create an InMemoryVectorStore with OpenAIEmbeddings.
    
    Args:
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        InMemoryVectorStore instance configured with OpenAIEmbeddings
    """
    try:
        embedding_model = embedding_model or "text-embedding-3-small"
        
        # Initialize OpenAI Embeddings
        # Use provided key, or fall back to settings, or let it use env var
        embeddings_kwargs = {}
        # Only pass API key if explicitly provided or from settings (and not empty)
        api_key = openai_api_key or (settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None)
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key
        # If no key provided, OpenAIEmbeddings will check OPENAI_API_KEY env var automatically
            
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            **embeddings_kwargs
        )
        
        # Create InMemoryVectorStore with embeddings
        vector_store = InMemoryVectorStore(embedding=embeddings)
        
        logger.info(f"Created InMemoryVectorStore with embedding model: {embedding_model}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise


def build_memory_from_pdfs(
    folder_path: str = "/Users/kanavkalra/Data/genAI/projects/policies",
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    vector_store: Optional[InMemoryVectorStore] = None
) -> InMemoryVectorStore:
    """
    Load PDF documents, split them, and add to InMemoryVectorStore.
    
    Args:
        folder_path: Path to the folder containing PDF files
        recursive: If True, search for PDFs recursively in subdirectories
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        vector_store: Optional existing vector store to add documents to. If None, creates a new one.
        
    Returns:
        InMemoryVectorStore with documents added
    """
    try:
        # Use provided key, or fall back to settings (if not empty)
        api_key = openai_api_key or (settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None)
        
        # Create or use existing vector store
        if vector_store is None:
            vector_store = create_vector_store(
                embedding_model=embedding_model,
                openai_api_key=api_key
            )
        
        # Load documents from PDFs
        logger.info(f"Loading PDF documents from: {folder_path}")
        documents = load_pdf_documents(folder_path=folder_path, recursive=recursive)
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        split_docs = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # Add documents to vector store
        logger.info(f"Adding {len(split_docs)} document chunks to vector store")
        vector_store.add_documents(split_docs)
        
        logger.info(f"Successfully built memory with {len(split_docs)} document chunks")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building memory from PDFs: {e}", exc_info=True)
        raise


def add_documents_to_vector_store(
    documents: List[Document],
    vector_store: InMemoryVectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> InMemoryVectorStore:
    """
    Split documents and add them to an existing InMemoryVectorStore.
    
    Args:
        documents: List of Document objects to add
        vector_store: InMemoryVectorStore to add documents to
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        
    Returns:
        InMemoryVectorStore with new documents added
    """
    try:
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return vector_store
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        split_docs = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # Add documents to vector store
        logger.info(f"Adding {len(split_docs)} document chunks to vector store")
        vector_store.add_documents(split_docs)
        
        logger.info(f"Successfully added {len(split_docs)} document chunks to vector store")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Building memory from PDF documents...")
        vector_store = build_memory_from_pdfs(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        logger.info("Memory built successfully!")
        logger.info(f"Vector store type: {type(vector_store).__name__}")
        
        # Example: Perform a similarity search
        query = "Is 1st Jan 2026 a holiday?"
        logger.info(f"Performing similarity search for: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        logger.info(f"Found {len(results)} similar documents:")
        for i, doc in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Content preview: {doc.page_content[:200]}...")
            logger.info(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)

