"""
Reindex vector store if documents have changed
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logging import logger
from vectorstore.metadata import compute_document_hash, DocumentMetadata


def check_documents_changed(folder_path: str) -> bool:
    """
    Check if documents in folder have changed.
    
    Args:
        folder_path: Path to folder containing documents
        
    Returns:
        True if documents have changed, False otherwise
    """
    # TODO: Implement document change detection
    # This would compare current document hashes with stored hashes
    logger.info(f"Checking for document changes in: {folder_path}")
    return False


def reindex_if_changed(folder_path: str):
    """
    Reindex vector store if documents have changed.
    
    Args:
        folder_path: Path to folder containing documents
    """
    if check_documents_changed(folder_path):
        logger.info("Documents have changed, reindexing...")
        from jobs.ingest_vectors import main
        main()
    else:
        logger.info("No changes detected, skipping reindex")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reindex vector store if documents changed")
    parser.add_argument(
        "--folder",
        type=str,
        default="/Users/kanavkalra/Data/genAI/projects/policies",
        help="Path to folder containing PDF files"
    )
    
    args = parser.parse_args()
    reindex_if_changed(args.folder)

