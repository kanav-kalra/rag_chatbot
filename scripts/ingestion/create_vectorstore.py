"""
CLI job for creating chatbot vector store using ChromaDB
This job can be run on-demand to build and persist vector stores for any chatbot type.
Uses the new architecture with ChatbotConfigManager and {chatbot_type}_chatbot.yaml

Supports multiple embedding providers/models by creating separate collections for each.
Collection names are automatically generated as: {base_name}_{provider}_{model_slug}
This allows storing multiple embeddings without overwriting existing ones.

Supports two indexing modes:
- incremental (default): Only indexes new or changed files (efficient, recommended)
- full: Always re-indexes everything (clears existing collection)
"""
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple
from datetime import datetime
import hashlib

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


def compute_file_hash(file_path: Path) -> str:
    """
    Compute a hash for a file based on its modification time and size.
    This is a lightweight way to detect file changes.
    
    Args:
        file_path: Path to the file (should be absolute/resolved)
        
    Returns:
        Hash string representing file state
    """
    try:
        # Resolve to absolute path to ensure consistency
        resolved_path = file_path.resolve()
        stat = resolved_path.stat()
        # Use modification time and size (don't include path to avoid path-based mismatches)
        # This is faster than computing full file hash but still effective
        content = f"{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute hash for {file_path}: {e}")
        return ""


def get_indexed_files(vector_store: Chroma) -> Dict[str, str]:
    """
    Get a mapping of file paths to their hashes from the vector store metadata.
    
    Args:
        vector_store: Chroma vector store instance
        
    Returns:
        Dictionary mapping file paths to their hashes
    """
    indexed_files = {}
    try:
        # Get all documents from the collection
        results = vector_store._collection.get(include=["metadatas"])
        if results and results.get("metadatas"):
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    # Store file hash if available, otherwise use empty string
                    file_hash = metadata.get("file_hash", "")
                    if source not in indexed_files:
                        indexed_files[source] = file_hash
    except Exception as e:
        logger.debug(f"Could not retrieve indexed files: {e}")
    return indexed_files


def find_changed_files(
    folder_path: str,
    recursive: bool,
    indexed_files: Dict[str, str]
) -> Tuple[List[Path], List[Path]]:
    """
    Find new and changed PDF files in the folder.
    
    Args:
        folder_path: Path to folder containing PDF files
        recursive: If True, search recursively
        indexed_files: Dictionary of already indexed files and their hashes
        
    Returns:
        Tuple of (new_files, changed_files) lists
    """
    folder = Path(folder_path).resolve()  # Normalize to absolute path
    
    # Find all PDF files
    if recursive:
        pdf_files = list(folder.rglob("*.pdf"))
    else:
        pdf_files = list(folder.glob("*.pdf"))
    
    new_files = []
    changed_files = []
    
    # Normalize indexed file paths to absolute paths for comparison
    normalized_indexed = {}
    for indexed_path, file_hash in indexed_files.items():
        try:
            # Try to resolve to absolute path for comparison
            normalized_path = str(Path(indexed_path).resolve())
            normalized_indexed[normalized_path] = file_hash
            # Also keep original path in case resolution fails
            if normalized_path != indexed_path:
                normalized_indexed[indexed_path] = file_hash
        except Exception:
            # If resolution fails, keep original
            normalized_indexed[indexed_path] = file_hash
    
    for pdf_file in pdf_files:
        # Use absolute path for comparison
        pdf_file_abs = pdf_file.resolve()
        file_path_str = str(pdf_file_abs)
        file_path_original = str(pdf_file)  # Also check original path
        
        current_hash = compute_file_hash(pdf_file_abs)
        
        # Check both absolute and original path
        indexed_hash = normalized_indexed.get(file_path_str) or normalized_indexed.get(file_path_original)
        
        if indexed_hash is None:
            # File not in index
            new_files.append(pdf_file)
        elif indexed_hash == "":
            # Old documents without file_hash - these were indexed before hash feature was added
            # We need to add the hash, but we can't tell if file changed without the hash
            # To be safe, we'll treat these as "needs hash update" but log it
            logger.debug(
                f"File {pdf_file.name} missing file_hash in metadata (old document). "
                f"Will update metadata with hash. File may be re-indexed if content changed."
            )
            # Add to changed_files to update metadata with hash
            # This is a one-time update for old documents
            changed_files.append(pdf_file)
        elif indexed_hash != current_hash:
            # Hash mismatch - file has actually changed
            logger.debug(f"File {pdf_file.name} hash changed: {indexed_hash[:8]}... -> {current_hash[:8]}...")
            changed_files.append(pdf_file)
    
    return new_files, changed_files


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
    folder_path: Optional[str] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    recursive: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None,
    clear_existing: bool = False,
    use_embedding_suffix: bool = True,
    skip_if_exists: bool = False,
    indexing_mode: Optional[str] = None
) -> Chroma:
    """
    Load PDF documents, split them, and add to ChromaDB vector store.
    Uses configuration from {chatbot_type}_chatbot.yaml.
    
    Supports multiple embedding providers/models by creating separate collections.
    By default, collection names include provider and model info to avoid overwriting.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr", "support", "default")
        folder_path: Path to the folder containing PDF files (default: from vector_store.ingestion.folder_path in config)
        persist_directory: Override persist directory (default: from vector_store.persist_dir in config)
        collection_name: Override collection name (default: from vector_store.collection_name in config)
        recursive: If True, search for PDFs recursively in subdirectories (default: from vector_store.ingestion.recursive in config)
        chunk_size: Maximum size of chunks to return (in characters) (default: from vector_store.ingestion.chunk_size in config)
        chunk_overlap: Overlap in characters between chunks (default: from vector_store.ingestion.chunk_overlap in config)
        embedding_provider: Override embedding provider (default: from vector_store.embedding_provider in config)
        embedding_model: Override embedding model (default: from vector_store.embedding_model in config)
        api_key: API key for the embedding provider (if not provided, uses env vars)
        clear_existing: If True, delete existing collection before adding documents
        use_embedding_suffix: If True, automatically append provider/model to collection name
        skip_if_exists: If True, skip adding documents if collection already exists
        indexing_mode: Indexing mode - "incremental" (default) or "full"
            - incremental: Only indexes new or changed files (efficient, skips if no changes)
            - full: Always re-indexes everything (equivalent to clear_existing=True)
        
    Returns:
        Chroma vector store with documents added
        
    Raises:
        FileNotFoundError: If config file for chatbot_type doesn't exist
        ValueError: If configuration is invalid
    """
    try:
        # Try to find config file - try both naming patterns for compatibility
        # Pattern 1: {chatbot_type}_chatbot_config.yaml (e.g., hr_chatbot_config.yaml)
        # Pattern 2: {chatbot_type}_chatbot.yaml (e.g., support_chatbot.yaml)
        config_filename = None
        config_manager = None
        
        for pattern in [f"{chatbot_type}_chatbot_config.yaml", f"{chatbot_type}_chatbot.yaml"]:
            try:
                config_manager = ChatbotConfigManager(pattern)
                config_filename = pattern
                break
            except FileNotFoundError:
                continue
        
        if config_manager is None or not config_manager.has_config():
            raise FileNotFoundError(
                f"Config file not found for chatbot type '{chatbot_type}'. "
                f"Tried: {chatbot_type}_chatbot_config.yaml and {chatbot_type}_chatbot.yaml. "
                f"Please create one of these files in config/chatbot/ directory."
            )
        
        # Get vector store config (with overrides from function args)
        vector_store_config = get_vector_store_config(chatbot_type, config_manager=config_manager)
        
        # Get ingestion defaults from config
        ingestion_config = config_manager.get_nested_dict("vector_store.ingestion", {})
        default_folder_path = ingestion_config.get("folder_path")
        default_chunk_size = ingestion_config.get("chunk_size", 1000)
        default_chunk_overlap = ingestion_config.get("chunk_overlap", 200)
        default_recursive = ingestion_config.get("recursive", True)
        default_indexing_mode = ingestion_config.get("indexing_mode", "incremental")
        
        # Apply function argument overrides (function args take precedence over config)
        persist_dir = persist_directory or vector_store_config["persist_dir"]
        # Get base collection name from config directly (not the generated one with suffix)
        # This prevents double-suffixing when generate_collection_name is called again
        base_coll_name = collection_name or config_manager.get("vector_store.collection_name")
        emb_provider = embedding_provider or vector_store_config.get("embedding_provider")
        emb_model = embedding_model or vector_store_config.get("embedding_model") or None
        
        # Apply ingestion defaults (use function args if provided, otherwise use config defaults)
        actual_folder_path = folder_path if folder_path is not None else default_folder_path
        actual_chunk_size = chunk_size if chunk_size is not None else default_chunk_size
        actual_chunk_overlap = chunk_overlap if chunk_overlap is not None else default_chunk_overlap
        actual_recursive = recursive if recursive is not None else default_recursive
        # Use provided indexing_mode, or config default, or fallback to "incremental"
        actual_indexing_mode = indexing_mode if indexing_mode is not None else (default_indexing_mode or "incremental")
        
        # Validate indexing mode
        if actual_indexing_mode not in ["incremental", "full"]:
            raise ValueError(
                f"Invalid indexing_mode: {actual_indexing_mode}. "
                f"Must be one of: 'incremental', 'full'"
            )
        
        # If indexing_mode is "full", it's equivalent to clear_existing
        if actual_indexing_mode == "full":
            clear_existing = True
            logger.info("Indexing mode is 'full' - will clear existing collection")
        
        # Validate folder path is set
        if not actual_folder_path:
            raise ValueError(
                f"folder_path is required. "
                f"Please set vector_store.ingestion.folder_path in {config_filename} or provide --folder argument."
            )
        
        # Validate embedding provider is set
        if not emb_provider:
            raise ValueError(
                f"embedding_provider is required. "
                f"Please set vector_store.embedding_provider in {config_filename} or use --embedding-provider argument."
            )
        
        # Get the actual model name that will be used (for collection naming)
        actual_model = get_default_embedding_model(emb_provider, emb_model)
        
        # Always generate collection name with embedding suffix to support multiple providers
        # The explicit collection_name (if provided) is treated as a base name
        if use_embedding_suffix:
            coll_name = generate_collection_name(base_coll_name, emb_provider, actual_model)
            logger.info(
                f"Auto-generated collection name with embedding suffix: {coll_name} "
                f"(base: {base_coll_name}, provider: {emb_provider}, model: {actual_model})"
            )
        else:
            # Only use base name if embedding suffix is explicitly disabled
            coll_name = base_coll_name
            logger.warning(
                f"Using collection name '{coll_name}' without embedding suffix. "
                f"This may cause conflicts when switching embedding providers. "
                f"Consider enabling embedding suffix (default) to support multiple embeddings."
            )
        
        logger.info(f"Building vector store for chatbot type: {chatbot_type}")
        logger.info(f"Using vector store config:")
        logger.info(f"  persist_dir: {persist_dir}")
        logger.info(f"  collection_name: {coll_name}")
        logger.info(f"  embedding_provider: {emb_provider}")
        logger.info(f"  embedding_model: {actual_model}")
        logger.info(f"Using ingestion config:")
        logger.info(f"  folder_path: {actual_folder_path}")
        logger.info(f"  chunk_size: {actual_chunk_size}")
        logger.info(f"  chunk_overlap: {actual_chunk_overlap}")
        logger.info(f"  recursive: {actual_recursive}")
        
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
        
        # Validate embedding function works
        try:
            test_embedding = embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding function returned empty embedding")
            logger.debug(f"Embedding function validated: dimension={len(test_embedding)}")
        except Exception as e:
            raise ValueError(
                f"Embedding function validation failed: {e}. "
                f"Please check your API key and embedding provider configuration."
            ) from e
        
        # Create or load ChromaDB vector store
        vector_store = Chroma(
            collection_name=coll_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        # Clear existing collection if requested (full mode or explicit clear_existing)
        if clear_existing and collection_already_exists:
            logger.info(f"Clearing existing collection: {coll_name}")
            vector_store.delete_collection()
            vector_store = Chroma(
                collection_name=coll_name,
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
        
        # Load documents from PDFs
        logger.info(f"Loading PDF documents from: {actual_folder_path}")
        
        # For incremental mode, filter to only new/changed files
        if actual_indexing_mode == "incremental" and collection_already_exists and not clear_existing:
            indexed_files = get_indexed_files(vector_store)
            logger.debug(f"Retrieved {len(indexed_files)} indexed file paths from vector store")
            
            new_files, changed_files = find_changed_files(actual_folder_path, actual_recursive, indexed_files)
            
            # Check how many changed files are due to missing hash vs actual changes
            files_missing_hash = 0
            files_actually_changed = 0
            for changed_file in changed_files:
                file_path_str = str(changed_file.resolve())
                file_path_original = str(changed_file)
                indexed_hash = indexed_files.get(file_path_str) or indexed_files.get(file_path_original)
                if indexed_hash == "":
                    files_missing_hash += 1
                else:
                    files_actually_changed += 1
            
            if files_missing_hash > 0:
                logger.info(
                    f"Found {files_missing_hash} files with missing file_hash (old documents from before hash feature). "
                    f"These will be updated to add hash metadata."
                )
            if files_actually_changed > 0:
                logger.info(f"Found {files_actually_changed} files with actual content changes.")
            
            if not new_files and not changed_files:
                logger.info("No new or changed files detected. Skipping indexing.")
                count = vector_store._collection.count()
                logger.info(f"Vector store contains {count} document chunks (unchanged)")
                return vector_store
            
            # Load only new and changed files
            files_to_index = new_files + changed_files
            logger.info(f"Incremental indexing: {len(new_files)} new files, {len(changed_files)} changed files")
            
            # Remove old chunks for changed files first
            if changed_files:
                logger.info(f"Removing old chunks for {len(changed_files)} changed files")
                changed_sources = {str(f) for f in changed_files}
                try:
                    # Get all documents with metadata (IDs are always returned)
                    results = vector_store._collection.get(include=["metadatas"])
                    if results and results.get("ids") and results.get("metadatas"):
                        ids_to_delete = []
                        for idx, metadata in enumerate(results["metadatas"]):
                            if metadata and metadata.get("source") in changed_sources:
                                ids_to_delete.append(results["ids"][idx])
                        
                        if ids_to_delete:
                            vector_store._collection.delete(ids=ids_to_delete)
                            logger.info(f"Removed {len(ids_to_delete)} old chunks for changed files")
                except Exception as e:
                    logger.warning(f"Could not remove old chunks for changed files: {e}")
            
            # Load only the files that need indexing
            documents = []
            for pdf_file in files_to_index:
                try:
                    from src.application.ingestion.loader import load_single_pdf
                    file_docs = load_single_pdf(str(pdf_file))
                    documents.extend(file_docs)
                except Exception as e:
                    logger.error(f"Error loading {pdf_file}: {e}", exc_info=True)
        else:
            # Full indexing mode - load all files
            documents = load_pdf_documents(folder_path=actual_folder_path, recursive=actual_recursive)
        
        if not documents:
            logger.warning(f"No PDF documents found or loaded from: {actual_folder_path}")
            return vector_store
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        split_docs = split_documents(
            documents,
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            add_start_index=True
        )
        
        # Filter out empty documents (they cause embedding errors)
        original_count = len(split_docs)
        split_docs = [doc for doc in split_docs if doc.page_content and doc.page_content.strip()]
        filtered_count = original_count - len(split_docs)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} empty document chunks")
        
        if not split_docs:
            logger.warning("No valid document chunks to add to vector store (all were empty)")
            return vector_store
        
        # Add file hash to metadata for change detection
        for doc in split_docs:
            if "source" in doc.metadata:
                source_path = Path(doc.metadata["source"])
                if source_path.exists():
                    doc.metadata["file_hash"] = compute_file_hash(source_path)
        
        # Add documents to vector store
        logger.info(f"Adding {len(split_docs)} document chunks to ChromaDB vector store")
        try:
            vector_store.add_documents(split_docs)
        except Exception as e:
            # Check if it's an embedding-related error
            error_msg = str(e).lower()
            if "embedding" in error_msg and "empty" in error_msg:
                logger.error(
                    f"Embedding error: Some documents may have empty content. "
                    f"Total chunks: {len(split_docs)}, "
                    f"First few chunk lengths: {[len(d.page_content) for d in split_docs[:5]]}"
                )
            raise
        
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
        description="Create chatbot vector store using ChromaDB (uses {chatbot_type}_chatbot_config.yaml or {chatbot_type}_chatbot.yaml)"
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
        default=None,
        help="Path to folder containing PDF files (default: from vector_store.ingestion.folder_path in config)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Override persist directory (default: from vector_store.persist_dir in config)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Override collection name (default: from vector_store.collection_name in config)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for text splitting (default: from vector_store.ingestion.chunk_size in config)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap for text splitting (default: from vector_store.ingestion.chunk_overlap in config)"
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["openai", "google"],
        default=None,
        help="Override embedding provider (default: from vector_store.embedding_provider in config)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model (default: from vector_store.embedding_model in config)"
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
        help="Do not search for PDFs recursively in subdirectories (default: from vector_store.ingestion.recursive in config)"
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
    parser.add_argument(
        "--indexing-mode",
        type=str,
        choices=["incremental", "full"],
        default=None,
        help="Indexing mode (default: from vector_store.ingestion.indexing_mode in config, or 'incremental'): "
             "'incremental' - only indexes new or changed files (efficient, skips if no changes), "
             "'full' - always re-indexes everything (clears existing)"
    )
    
    args = parser.parse_args()
    
    try:
        # Try to find config file to get defaults
        config_filename = None
        for pattern in [f"{args.chatbot_type}_chatbot_config.yaml", f"{args.chatbot_type}_chatbot.yaml"]:
            config_path = Path(__file__).parent.parent.parent / "config" / "chatbot" / pattern
            if config_path.exists():
                config_filename = pattern
                break
        
        logger.info(f"Starting {args.chatbot_type} vector store creation")
        if config_filename:
            logger.info(f"Using config file: {config_filename}")
        if args.folder:
            logger.info(f"Folder path (override): {args.folder}")
        if args.persist_dir:
            logger.info(f"Persist directory (override): {args.persist_dir}")
        if args.collection_name:
            logger.info(f"Collection name (override): {args.collection_name}")
        if args.embedding_provider:
            logger.info(f"Embedding provider (override): {args.embedding_provider}")
        if args.embedding_model:
            logger.info(f"Embedding model (override): {args.embedding_model}")
        if args.chunk_size:
            logger.info(f"Chunk size (override): {args.chunk_size}")
        if args.chunk_overlap:
            logger.info(f"Chunk overlap (override): {args.chunk_overlap}")
        if args.indexing_mode:
            logger.info(f"Indexing mode (override): {args.indexing_mode}")
        if config_filename:
            logger.info(f"(Using defaults from {config_filename} for unspecified options)")
        
        # Determine recursive value: explicit --no-recursive flag overrides config
        recursive_value = False if args.no_recursive else None
        
        vector_store = build_vectorstore_from_pdfs(
            chatbot_type=args.chatbot_type,
            folder_path=args.folder,
            persist_directory=args.persist_dir,
            collection_name=args.collection_name,
            recursive=recursive_value,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            api_key=args.api_key,
            clear_existing=args.clear_existing,
            use_embedding_suffix=not args.no_embedding_suffix,
            skip_if_exists=args.skip_if_exists,
            indexing_mode=args.indexing_mode
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

