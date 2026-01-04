"""
CLI job for creating chatbot vector store using ChromaDB
This job can be run on-demand to build and persist vector stores for any chatbot type.
Uses the new architecture with ChatbotConfigManager and {chatbot_type}_chatbot_config.yaml
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import Chroma

from app.core.logging import logger
from app.services.chatbot.config_manager import ChatbotConfigManager
from app.infra.vectorstore.vector_store_manager import (
    get_vector_store_config,
    create_embeddings
)
from ingestion.loader import load_pdf_documents
from ingestion.chunker import split_documents


def build_vectorstore_from_pdfs(
    chatbot_type: str,
    folder_path: str,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None,
    clear_existing: bool = False
) -> Chroma:
    """
    Load PDF documents, split them, and add to ChromaDB vector store.
    Uses configuration from {chatbot_type}_chatbot_config.yaml.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "support", "default")
        folder_path: Path to the folder containing PDF files
        persist_directory: Override persist directory (default: from config)
        collection_name: Override collection name (default: from config)
        recursive: If True, search for PDFs recursively in subdirectories
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        embedding_provider: Override embedding provider (default: from config)
        embedding_model: Override embedding model (default: from config)
        api_key: API key for the embedding provider (if not provided, uses env vars)
        clear_existing: If True, delete existing collection before adding documents
        
    Returns:
        Chroma vector store with documents added
        
    Raises:
        FileNotFoundError: If config file for chatbot_type doesn't exist
        ValueError: If configuration is invalid
    """
    try:
        # Construct config filename from chatbot type
        config_filename = f"{chatbot_type}_chatbot_config.yaml"
        
        # Load configuration from YAML
        try:
            config_manager = ChatbotConfigManager(config_filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found: {config_filename}. "
                f"Please create {config_filename} in app/core/ directory."
            )
        
        # Get vector store config (with overrides from function args)
        vector_store_config = get_vector_store_config(chatbot_type, config_manager=config_manager)
        
        # Apply function argument overrides
        persist_dir = persist_directory or vector_store_config["persist_dir"]
        coll_name = collection_name or vector_store_config["collection_name"]
        emb_provider = embedding_provider or vector_store_config.get("embedding_provider", "auto")
        emb_model = embedding_model or vector_store_config.get("embedding_model") or None
        
        # Handle "auto" embedding provider
        if emb_provider == "auto":
            model_name = config_manager.get("model.name", "")
            if model_name and model_name.startswith("gemini"):
                emb_provider = "google"
            else:
                emb_provider = "openai"
        
        logger.info(f"Building vector store for chatbot type: {chatbot_type}")
        logger.info(f"Using vector store config:")
        logger.info(f"  persist_dir: {persist_dir}")
        logger.info(f"  collection_name: {coll_name}")
        logger.info(f"  embedding_provider: {emb_provider}")
        if emb_model:
            logger.info(f"  embedding_model: {emb_model}")
        
        # Create persist directory if it doesn't exist
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings using centralized function
        embeddings = create_embeddings(
            provider=emb_provider,
            embedding_model=emb_model,
            api_key=api_key
        )
        
        # Create or load ChromaDB vector store
        vector_store = Chroma(
            collection_name=coll_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        # Clear existing collection if requested
        if clear_existing:
            logger.info(f"Clearing existing collection: {coll_name}")
            vector_store.delete_collection()
            vector_store = Chroma(
                collection_name=coll_name,
                embedding_function=embeddings,
                persist_directory=persist_dir
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
            f"Successfully built {chatbot_type} vector store with {len(split_docs)} document chunks. "
            f"Persisted to: {persist_dir}"
        )
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building {chatbot_type} vector store from PDFs: {e}", exc_info=True)
        raise


def main():
    """Main entrypoint for chatbot vector store creation"""
    parser = argparse.ArgumentParser(
        description="Create chatbot vector store using ChromaDB (uses {chatbot_type}_chatbot_config.yaml)"
    )
    parser.add_argument(
        "--chatbot-type",
        type=str,
        required=True,
        help="Type of chatbot (e.g., 'hr', 'support', 'default')"
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
        default=None,
        help="Override persist directory (default: from {chatbot_type}_chatbot_config.yaml)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Override collection name (default: from {chatbot_type}_chatbot_config.yaml)"
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
        "--embedding-provider",
        type=str,
        choices=["openai", "google", "auto"],
        default=None,
        help="Override embedding provider (default: from {chatbot_type}_chatbot_config.yaml)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model (default: from {chatbot_type}_chatbot_config.yaml)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the embedding provider (if not provided, uses env vars)"
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
        config_filename = f"{args.chatbot_type}_chatbot_config.yaml"
        logger.info(f"Starting {args.chatbot_type} vector store creation from: {args.folder}")
        logger.info(f"Using config file: {config_filename}")
        if args.persist_dir:
            logger.info(f"Persist directory (override): {args.persist_dir}")
        if args.collection_name:
            logger.info(f"Collection name (override): {args.collection_name}")
        if args.embedding_provider:
            logger.info(f"Embedding provider (override): {args.embedding_provider}")
        if args.embedding_model:
            logger.info(f"Embedding model (override): {args.embedding_model}")
        logger.info(f"(Using defaults from {config_filename} for unspecified options)")
        
        vector_store = build_vectorstore_from_pdfs(
            chatbot_type=args.chatbot_type,
            folder_path=args.folder,
            persist_directory=args.persist_dir,
            collection_name=args.collection_name,
            recursive=not args.no_recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            api_key=args.api_key,
            clear_existing=args.clear_existing
        )
        
        # Verify the vector store
        logger.info("Verifying vector store...")
        collection_count = vector_store._collection.count()
        logger.info(f"Vector store contains {collection_count} document chunks")
        
        logger.info(f"{args.chatbot_type} vector store creation completed successfully")
        
    except Exception as e:
        logger.error(f"{args.chatbot_type} vector store creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

