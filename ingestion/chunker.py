"""
Text Splitter for Document Chunking using RecursiveCharacterTextSplitter
"""
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import logger


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with specified parameters.
    
    Args:
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        add_start_index: If True, includes start_index in metadata
        separators: List of separators to use for splitting. If None, uses default.
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
        separators=separators,
    )
    return splitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True
) -> List[Document]:
    """
    Split a list of documents into smaller chunks.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        add_start_index: If True, includes start_index in metadata
        
    Returns:
        List of split Document objects
    """
    if not documents:
        logger.warning("No documents provided for splitting")
        return []
    
    logger.info(f"Splitting {len(documents)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    text_splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index
    )
    
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        logger.error(f"Error splitting documents: {e}", exc_info=True)
        raise


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split a single text string into chunks.
    
    Args:
        text: Text string to split
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        logger.warning("Empty text provided for splitting")
        return []
    
    logger.debug(f"Splitting text (length: {len(text)}) with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    text_splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=False
    )
    
    try:
        chunks = text_splitter.split_text(text)
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    from ingestion.loader import load_pdf_documents
    
    try:
        # Load documents
        logger.info("Loading PDF documents...")
        docs = load_pdf_documents()
        
        # Split documents
        logger.info("Splitting documents...")
        split_docs = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        
        logger.info(f"Original documents: {len(docs)}")
        logger.info(f"Split documents: {len(split_docs)}")
        
        # Show first chunk as example
        if split_docs:
            logger.info("\nFirst chunk preview:")
            logger.info(f"Content (first 200 chars): {split_docs[0].page_content[:200]}...")
            logger.info(f"Metadata: {split_docs[0].metadata}")
            
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)

