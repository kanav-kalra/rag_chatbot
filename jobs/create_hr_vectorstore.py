"""
CLI job for creating HR chatbot vector store using ChromaDB
This job can be run on-demand to build and persist the HR vector store.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings
from app.core.logging import logger
from ingestion.loader import load_pdf_documents
from ingestion.chunker import split_documents


def create_chroma_vector_store(
    persist_directory: str,
    collection_name: str = "hr_chatbot",
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Chroma:
    """
    Create or load a ChromaDB vector store with OpenAIEmbeddings.
    If the persist_directory exists, it will load existing data; otherwise creates new.
    
    Args:
        persist_directory: Directory path where ChromaDB will persist the data
        collection_name: Name of the ChromaDB collection (default: "hr_chatbot")
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        Chroma vector store instance configured with OpenAIEmbeddings
    """
    try:
        embedding_model = embedding_model or "text-embedding-3-small"
        
        # Initialize OpenAI Embeddings
        api_key = openai_api_key or (settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else None)
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key
            
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            **embeddings_kwargs
        )
        
        # Create or load ChromaDB vector store
        # If persist_directory exists, it will load existing data; otherwise creates new
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(
            f"Created/loaded ChromaDB vector store with embedding model: {embedding_model}, "
            f"collection: {collection_name}, persist_dir: {persist_directory}"
        )
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating/loading ChromaDB vector store: {e}", exc_info=True)
        raise


def load_hr_chroma_vector_store(
    persist_directory: str = "./chroma_db/hr_chatbot",
    collection_name: str = "hr_chatbot",
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Chroma:
    """
    Load an existing HR ChromaDB vector store.
    This function can be used by the HR chatbot service to load the persisted vector store.
    
    Args:
        persist_directory: Directory path where ChromaDB persists the data
        collection_name: Name of the ChromaDB collection (default: "hr_chatbot")
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        
    Returns:
        Chroma vector store instance
        
    Raises:
        FileNotFoundError: If the persist_directory does not exist
        ValueError: If the collection does not exist or is empty
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: {persist_directory}. "
            f"Please run the create_hr_vectorstore job first."
        )
    
    vector_store = create_chroma_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key
    )
    
    # Verify the collection has data
    try:
        count = vector_store._collection.count()
        if count == 0:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' exists but is empty. "
                f"Please run the create_hr_vectorstore job to populate it."
            )
        logger.info(f"Loaded HR ChromaDB vector store with {count} document chunks")
    except Exception as e:
        logger.warning(f"Could not verify collection count: {e}")
    
    return vector_store


def build_hr_vectorstore_from_pdfs(
    folder_path: str,
    persist_directory: str,
    collection_name: str = "hr_chatbot",
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    clear_existing: bool = False
) -> Chroma:
    """
    Load PDF documents, split them, and add to ChromaDB vector store.
    
    Args:
        folder_path: Path to the folder containing PDF files
        persist_directory: Directory path where ChromaDB will persist the data
        collection_name: Name of the ChromaDB collection (default: "hr_chatbot")
        recursive: If True, search for PDFs recursively in subdirectories
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-small")
        openai_api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        clear_existing: If True, delete existing collection before adding documents
        
    Returns:
        Chroma vector store with documents added
    """
    try:
        # Create ChromaDB vector store
        vector_store = create_chroma_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Clear existing collection if requested
        if clear_existing:
            logger.info(f"Clearing existing collection: {collection_name}")
            # Delete the collection by recreating it
            vector_store.delete_collection()
            vector_store = create_chroma_vector_store(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_model=embedding_model,
                openai_api_key=openai_api_key
            )
        
        # Load documents from PDFs
        logger.info(f"Loading PDF documents from: {folder_path}")
        documents = load_pdf_documents(folder_path=folder_path, recursive=recursive)
        
        if not documents:
            logger.warning(f"No PDF documents found in: {folder_path}")
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
        logger.info(f"Adding {len(split_docs)} document chunks to ChromaDB vector store")
        vector_store.add_documents(split_docs)
        
        # Persist the vector store
        vector_store.persist()
        
        logger.info(
            f"Successfully built HR vector store with {len(split_docs)} document chunks. "
            f"Persisted to: {persist_directory}"
        )
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building HR vector store from PDFs: {e}", exc_info=True)
        raise


def main():
    """Main entrypoint for HR vector store creation"""
    parser = argparse.ArgumentParser(
        description="Create HR chatbot vector store using ChromaDB"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/Users/kanavkalra/Data/genAI/projects/policies",
        help="Path to folder containing PDF files"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db/hr_chatbot",
        help="Directory path where ChromaDB will persist the data"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="hr_chatbot",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name"
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing collection before adding documents"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search for PDFs recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    try:
        # Create persist directory if it doesn't exist
        persist_dir = Path(args.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting HR vector store creation from: {args.folder}")
        logger.info(f"Persist directory: {args.persist_dir}")
        logger.info(f"Collection name: {args.collection_name}")
        
        vector_store = build_hr_vectorstore_from_pdfs(
            folder_path=args.folder,
            persist_directory=str(persist_dir),
            collection_name=args.collection_name,
            recursive=not args.no_recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            clear_existing=args.clear_existing
        )
        
        # Verify the vector store
        logger.info("Verifying vector store...")
        collection_count = vector_store._collection.count()
        logger.info(f"Vector store contains {collection_count} document chunks")
        
        logger.info("HR vector store creation completed successfully")
        
    except Exception as e:
        logger.error(f"HR vector store creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

