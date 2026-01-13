"""
Script to check ChromaDB collections and diagnose collection name mismatches
"""
import sys
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

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_collections(chatbot_type: str = "hr"):
    """
    Check what collections exist and what the system is trying to load.
    
    Args:
        chatbot_type: Type of chatbot (e.g., "hr")
    """
    if not CHROMADB_AVAILABLE:
        logger.error("chromadb package is not available. Please install it: pip install chromadb")
        return
    
    persist_dir = f"./data/vectorstores/chroma_db/{chatbot_type}_chatbot"
    if chatbot_type == "hr":
        persist_dir = f"./data/vectorstores/chroma_db/hr_chatbot"
    
    persist_path = Path(persist_dir)
    
    if not persist_path.exists():
        logger.error(f"Persist directory does not exist: {persist_dir}")
        return
    
    logger.info(f"Checking collections in: {persist_dir}")
    
    # List all collections
    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        collections = client.list_collections()
        
        if not collections:
            logger.warning("No collections found in the persist directory")
            return
        
        logger.info(f"\nFound {len(collections)} collection(s):")
        for coll in collections:
            try:
                count = coll.count()
                logger.info(f"  ✓ {coll.name}: {count} documents")
            except Exception as e:
                logger.warning(f"  ✗ {coll.name}: Error getting count - {e}")
        
        # Try to determine what collection the system would load
        logger.info("\n" + "="*60)
        logger.info("Checking what collection the system would load...")
        
        try:
            from src.domain.chatbot.core.config import ChatbotConfigManager
            from src.infrastructure.vectorstore.manager import get_vector_store_config
            
            # Try to load config
            try:
                config_manager = ChatbotConfigManager('hr_chatbot')
            except:
                try:
                    config_manager = ChatbotConfigManager('hr')
                except:
                    config_manager = None
            
            config = get_vector_store_config(chatbot_type, config_manager=config_manager)
            expected_collection = config.get("collection_name", "unknown")
            embedding_provider = config.get("embedding_provider", "unknown")
            
            logger.info(f"Expected collection name: {expected_collection}")
            logger.info(f"Embedding provider: {embedding_provider}")
            
            # Check if expected collection exists
            collection_names = [c.name for c in collections]
            if expected_collection in collection_names:
                matching_coll = next(c for c in collections if c.name == expected_collection)
                count = matching_coll.count()
                logger.info(f"✓ Expected collection '{expected_collection}' exists with {count} documents")
            else:
                logger.warning(f"✗ Expected collection '{expected_collection}' does NOT exist!")
                logger.warning("Available collections:")
                for name in collection_names:
                    logger.warning(f"  - {name}")
                
                # Find collections with documents
                collections_with_data = [c for c in collections if c.count() > 0]
                if collections_with_data:
                    logger.info(f"\nCollections with data ({len(collections_with_data)}):")
                    for coll in collections_with_data:
                        logger.info(f"  - {coll.name}: {coll.count()} documents")
                    logger.info("\nPossible solutions:")
                    logger.info("  1. Re-index documents to create the expected collection")
                    logger.info("  2. Update the config to use an existing collection name")
                    logger.info("  3. Clear vector store cache and reload")
        
        except Exception as e:
            logger.error(f"Error checking expected collection: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error accessing ChromaDB: {e}", exc_info=True)


def main():
    """Main entrypoint"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check ChromaDB collections")
    parser.add_argument(
        "--chatbot-type",
        type=str,
        default="hr",
        help="Chatbot type to check (default: hr)"
    )
    
    args = parser.parse_args()
    
    try:
        check_collections(chatbot_type=args.chatbot_type)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
