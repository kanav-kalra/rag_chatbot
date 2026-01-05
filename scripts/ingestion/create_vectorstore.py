"""
CLI job for creating chatbot vector store using ChromaDB
This job can be run on-demand to build and persist vector stores for any chatbot type.
Uses the new architecture with ChatbotConfigManager and {chatbot_type}_chatbot.yaml

Supports multiple embedding providers/models by creating separate collections for each.
Collection names are automatically generated as: {base_name}_{provider}_{model_slug}
This allows storing multiple embeddings without overwriting existing ones.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
# scripts/ingestion/create_vectorstore.py -> scripts/ingestion -> scripts -> project_root
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from langchain_community.vectorstores import Chroma

from src.shared.config.logging import logger
from src.domain.chatbot.core.config import ChatbotConfigManager
from src.infrastructure.vectorstore.manager import (
    get_vector_store_config,
    create_embeddings,
    generate_collection_name,
    get_default_embedding_model
)
from src.application.ingestion.loader import load_pdf_documents
from src.application.ingestion.chunker import split_documents


def collection_exists(persist_dir: str, collection_name: str) -> bool:
    """
    Check if a ChromaDB collection exists.
    
    Args:
        persist_dir: ChromaDB persist directory
        collection_name: Collection name to check
        
    Returns:
        True if collection exists, False otherwise
    """
    if not CHROMADB_AVAILABLE:
        return False
    
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        # Try to get the collection - will raise if it doesn't exist
        client.get_collection(name=collection_name)
        return True
    except Exception:
        return False


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
    clear_existing: bool = False,
    use_embedding_suffix: bool = True,
    skip_if_exists: bool = False
) -> Chroma:
    """
    Load PDF documents, split them, and add to ChromaDB vector store.
    Uses configuration from {chatbot_type}_chatbot.yaml.
    
    Supports multiple embedding providers/models by creating separate collections.
    By default, collection names include provider and model info to avoid overwriting.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "support", "default")
        folder_path: Path to the folder containing PDF files
        persist_directory: Override persist directory (default: from config)
        collection_name: Override collection name (default: from config, or auto-generated)
        recursive: If True, search for PDFs recursively in subdirectories
        chunk_size: Maximum size of chunks to return (in characters)
        chunk_overlap: Overlap in characters between chunks
        embedding_provider: Override embedding provider (default: from config)
        embedding_model: Override embedding model (default: from config)
        api_key: API key for the embedding provider (if not provided, uses env vars)
        clear_existing: If True, delete existing collection before adding documents
        use_embedding_suffix: If True, automatically append provider/model to collection name
        skip_if_exists: If True, skip adding documents if collection already exists
        
    Returns:
        Chroma vector store with documents added
        
    Raises:
        FileNotFoundError: If config file for chatbot_type doesn't exist
        ValueError: If configuration is invalid
    """
    try:
        # Construct config filename from chatbot type
        # Note: Config files are named {chatbot_type}_chatbot.yaml (not {chatbot_type}_chatbot_config.yaml)
        config_filename = f"{chatbot_type}_chatbot.yaml"
        
        # Load configuration from YAML
        try:
            config_manager = ChatbotConfigManager(config_filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found: {config_filename}. "
                f"Please create {config_filename} in config/chatbot/ directory."
            )
        
        # Get vector store config (with overrides from function args)
        vector_store_config = get_vector_store_config(chatbot_type, config_manager=config_manager)
        
        # Apply function argument overrides
        persist_dir = persist_directory or vector_store_config["persist_dir"]
        base_coll_name = collection_name or vector_store_config["collection_name"]
        emb_provider = embedding_provider or vector_store_config.get("embedding_provider", "auto")
        emb_model = embedding_model or vector_store_config.get("embedding_model") or None
        
        # Handle "auto" embedding provider
        if emb_provider == "auto":
            model_name = config_manager.get("model.name", "")
            if model_name and model_name.startswith("gemini"):
                emb_provider = "google"
            else:
                emb_provider = "openai"
        
        # Get the actual model name that will be used (for collection naming)
        actual_model = get_default_embedding_model(emb_provider, emb_model)
        
        # Generate collection name with embedding info if requested
        if use_embedding_suffix and collection_name is None:
            # Only auto-generate if collection_name wasn't explicitly provided
            coll_name = generate_collection_name(base_coll_name, emb_provider, actual_model)
            logger.info(
                f"Auto-generated collection name with embedding suffix: {coll_name} "
                f"(base: {base_coll_name}, provider: {emb_provider}, model: {actual_model})"
            )
        else:
            coll_name = base_coll_name
            if use_embedding_suffix:
                logger.warning(
                    f"Using explicit collection name '{coll_name}' without embedding suffix. "
                    f"Consider using auto-generated names to support multiple embeddings."
                )
        
        logger.info(f"Building vector store for chatbot type: {chatbot_type}")
        logger.info(f"Using vector store config:")
        logger.info(f"  persist_dir: {persist_dir}")
        logger.info(f"  collection_name: {coll_name}")
        logger.info(f"  embedding_provider: {emb_provider}")
        logger.info(f"  embedding_model: {actual_model}")
        
        # Create persist directory if it doesn't exist
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Check if collection already exists
        collection_already_exists = collection_exists(persist_dir, coll_name)
        
        if collection_already_exists:
            if skip_if_exists:
                logger.info(
                    f"Collection '{coll_name}' already exists. Skipping document ingestion "
                    f"(use --clear-existing to overwrite or remove --skip-if-exists to add to existing)."
                )
                # Still return the vector store so caller can use it
                embeddings = create_embeddings(
                    provider=emb_provider,
                    embedding_model=emb_model,
                    api_key=api_key
                )
                vector_store = Chroma(
                    collection_name=coll_name,
                    embedding_function=embeddings,
                    persist_directory=persist_dir
                )
                count = vector_store._collection.count()
                logger.info(f"Existing collection contains {count} document chunks")
                return vector_store
            elif not clear_existing:
                # Get count from existing collection
                try:
                    temp_embeddings = create_embeddings(
                        provider=emb_provider,
                        embedding_model=emb_model,
                        api_key=api_key
                    )
                    temp_store = Chroma(
                        collection_name=coll_name,
                        embedding_function=temp_embeddings,
                        persist_directory=persist_dir
                    )
                    existing_count = temp_store._collection.count()
                    logger.warning(
                        f"Collection '{coll_name}' already exists with {existing_count} chunks. "
                        f"Documents will be added to existing collection. "
                        f"Use --clear-existing to replace or --skip-if-exists to skip."
                    )
                except Exception:
                    logger.warning(
                        f"Collection '{coll_name}' already exists. "
                        f"Documents will be added to existing collection. "
                        f"Use --clear-existing to replace or --skip-if-exists to skip."
                    )
        
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
        if clear_existing and collection_already_exists:
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
        description="Create chatbot vector store using ChromaDB (uses {chatbot_type}_chatbot.yaml)"
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
            help="Override persist directory (default: from {chatbot_type}_chatbot.yaml)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
            help="Override collection name (default: from {chatbot_type}_chatbot.yaml)"
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
            help="Override embedding provider (default: from {chatbot_type}_chatbot.yaml)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
            help="Override embedding model (default: from {chatbot_type}_chatbot.yaml)"
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
    parser.add_argument(
        "--no-embedding-suffix",
        action="store_true",
        help="Disable automatic collection name suffix with embedding provider/model. "
             "By default, collection names include provider and model to support multiple embeddings."
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip document ingestion if collection already exists. "
             "Useful when you want to avoid re-ingesting documents for existing embeddings."
    )
    
    args = parser.parse_args()
    
    try:
        config_filename = f"{args.chatbot_type}_chatbot.yaml"
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
            clear_existing=args.clear_existing,
            use_embedding_suffix=not args.no_embedding_suffix,
            skip_if_exists=args.skip_if_exists
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

