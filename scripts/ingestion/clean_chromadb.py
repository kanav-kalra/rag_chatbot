"""
Script to clean all ChromaDB collections
This script will delete all collections from all ChromaDB persist directories.
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_all_collections(base_dir: str = "./data/vectorstores/chroma_db"):
    """
    Clean all collections from all ChromaDB persist directories.
    
    Args:
        base_dir: Base directory containing ChromaDB persist directories
    """
    if not CHROMADB_AVAILABLE:
        logger.error("chromadb package is not available. Please install it: pip install chromadb")
        return
    
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Base directory does not exist: {base_dir}")
        return
    
    logger.info(f"Scanning for ChromaDB persist directories in: {base_path.absolute()}")
    
    # Find all subdirectories that might contain ChromaDB instances
    persist_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this directory contains chroma.sqlite3 (indicates ChromaDB)
            chroma_db_file = item / "chroma.sqlite3"
            if chroma_db_file.exists():
                persist_dirs.append(item)
                logger.info(f"Found ChromaDB persist directory: {item}")
    
    if not persist_dirs:
        logger.info("No ChromaDB persist directories found.")
        return
    
    total_collections_deleted = 0
    
    for persist_dir in persist_dirs:
        try:
            logger.info(f"\nProcessing: {persist_dir}")
            
            # Create ChromaDB client
            client = chromadb.PersistentClient(path=str(persist_dir))
            
            # List all collections
            collections = client.list_collections()
            
            if not collections:
                logger.info(f"  No collections found in {persist_dir.name}")
                continue
            
            logger.info(f"  Found {len(collections)} collection(s):")
            for collection in collections:
                try:
                    count = collection.count()
                    logger.info(f"    - {collection.name} ({count} documents)")
                except Exception as e:
                    logger.warning(f"    - {collection.name} (could not get count: {e})")
            
            # Delete all collections
            for collection in collections:
                try:
                    logger.info(f"  Deleting collection: {collection.name}")
                    client.delete_collection(name=collection.name)
                    total_collections_deleted += 1
                    logger.info(f"  ✓ Successfully deleted: {collection.name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete {collection.name}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing {persist_dir}: {e}", exc_info=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary: Deleted {total_collections_deleted} collection(s) from {len(persist_dirs)} persist directory(ies)")
    logger.info(f"{'='*60}")


def main():
    """Main entrypoint"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean all ChromaDB collections")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./data/vectorstores/chroma_db",
        help="Base directory containing ChromaDB persist directories (default: ./data/vectorstores/chroma_db)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    if not args.confirm:
        response = input(
            f"\n⚠️  WARNING: This will delete ALL collections from ALL ChromaDB persist directories!\n"
            f"Base directory: {args.base_dir}\n\n"
            f"Are you sure you want to continue? (yes/no): "
        )
        if response.lower() not in ['yes', 'y']:
            logger.info("Operation cancelled.")
            return
    
    try:
        clean_all_collections(base_dir=args.base_dir)
        logger.info("\n✓ Cleanup completed successfully!")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
