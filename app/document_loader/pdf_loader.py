"""
Document Loader for PDF files using PyPDFLoader
"""
import sys
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.core.logger import logger


def load_pdf_documents(
    folder_path: str = "/Users/kanavkalra/Data/genAI/projects/policies",
    recursive: bool = True
) -> List[Document]:
    """
    Load all PDF documents from a specified folder using PyPDFLoader.
    
    Args:
        folder_path: Path to the folder containing PDF files
        recursive: If True, search for PDFs recursively in subdirectories
        
    Returns:
        List of Document objects, one per page of each PDF
        
    Raises:
        ValueError: If the folder path doesn't exist
        FileNotFoundError: If no PDF files are found in the folder
    """
    folder = Path(folder_path)
    
    # Validate folder exists
    if not folder.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all PDF files
    if recursive:
        pdf_files = list(folder.rglob("*.pdf"))
    else:
        pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in folder: {folder_path}"
        )
    
    # Load all PDF documents
    all_documents = []
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file}")
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source file path to metadata if not already present
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = str(pdf_file)
                if "file_name" not in doc.metadata:
                    doc.metadata["file_name"] = pdf_file.name
            
            all_documents.extend(documents)
            logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {str(e)}", exc_info=True)
            continue
    
    logger.info(f"Total documents loaded: {len(all_documents)} pages from {len(pdf_files)} PDF files")
    
    return all_documents


def load_single_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file using PyPDFLoader.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    
    # Add source file path to metadata
    for doc in documents:
        if "source" not in doc.metadata:
            doc.metadata["source"] = str(pdf_file)
        if "file_name" not in doc.metadata:
            doc.metadata["file_name"] = pdf_file.name
    
    return documents


if __name__ == "__main__":
    # Example usage
    try:
        docs = load_pdf_documents()
        logger.info(f"Successfully loaded {len(docs)} document pages")
        
        # Log first document as example
        if docs:
            logger.debug("First document preview:")
            logger.debug(f"Content (first 200 chars): {docs[0].page_content[:200]}...")
            logger.debug(f"Metadata: {docs[0].metadata}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

