"""
CLI entrypoint for ingesting vectors into the vector store
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ingestion.embedder import build_memory_from_pdfs
from vectorstore.client import set_vector_store
from app.core.logging import logger


def main():
    """Main entrypoint for vector ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--folder",
        type=str,
        default="/Users/kanavkalra/Data/genAI/projects/policies",
        help="Path to folder containing PDF files"
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
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting vector ingestion from: {args.folder}")
        vector_store = build_memory_from_pdfs(
            folder_path=args.folder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        set_vector_store(vector_store)
        logger.info("Vector ingestion completed successfully")
    except Exception as e:
        logger.error(f"Vector ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

