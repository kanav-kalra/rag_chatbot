"""
Document Retrieval Service - Generic document retrieval from vector stores
Uses dependency injection - vector store is provided at initialization.
No knowledge of chatbot types or domain-specific concepts.
"""
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from app.core.logging import logger


class RetrievalService:
    """
    Generic document retrieval service that works with any vector store.
    Uses dependency injection - vector store is provided at initialization.
    This service has no knowledge of chatbot types, HR, or any domain-specific concepts.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize retrieval service with a vector store.
        
        Args:
            vector_store: VectorStore instance to use for retrieval
            
        Raises:
            ValueError: If vector_store is None
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        self.vector_store = vector_store
        logger.debug("RetrievalService initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant documents from the vector store based on a query.
        
        Args:
            query: The search query string
            k: Number of documents to retrieve (default: 4)
            
        Returns:
            Tuple of (content, artifact):
                - content: Formatted string with retrieved documents
                - artifact: List of dictionaries with detailed document information (rank, content, metadata)
                
        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            logger.info(f"Retrieving documents for query: '{query}' (k={k})")
            documents = self.vector_store.similarity_search(query, k=k)
            
            artifact = []
            content_parts = []
            
            for i, doc in enumerate(documents):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                artifact.append(result)
                
                # Format content string
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                file_name = doc.metadata.get("file_name", "Unknown")
                
                content_parts.append(
                    f"Document {i + 1} (Source: {file_name}, Page: {page}):\n"
                    f"{doc.page_content}\n"
                )
            
            content = "\n---\n\n".join(content_parts)
            logger.info(f"Retrieved {len(artifact)} documents for query: '{query}'")
            
            return (content, artifact)
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with similarity scores from the vector store.
        
        Args:
            query: The search query string
            k: Number of documents to retrieve (default: 4)
            
        Returns:
            List of dictionaries containing:
                - rank: Document rank (1-indexed)
                - content: The document content
                - metadata: Document metadata (source, page, etc.)
                - score: Similarity score
                
        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            logger.info(f"Retrieving documents with scores for query: '{query}' (k={k})")
            documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for i, (doc, score) in enumerate(documents_with_scores):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} documents with scores for query: '{query}'")
            return results
            
        except Exception as e:
            error_msg = f"Error retrieving documents with scores: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def retrieve_as_string(
        self,
        query: str,
        k: int = 4,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Retrieve relevant documents and return them as a formatted string.
        
        Args:
            query: The search query string
            k: Number of documents to retrieve (default: 4)
            separator: String to separate documents (default: "\n\n---\n\n")
            
        Returns:
            Formatted string containing all retrieved documents
            
        Raises:
            RuntimeError: If retrieval fails
        """
        content, _ = self.retrieve(query=query, k=k)
        return content
    
    def create_tool(
        self,
        tool_name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Create a LangChain tool from this retrieval service instance.
        
        Args:
            tool_name: Optional custom name for the tool (default: "retrieve_documents")
            description: Optional custom description for the tool
            
        Returns:
            A LangChain tool that can be used by agents
        """
        @tool(response_format="content_and_artifact")
        def retrieve_documents_tool(
            query: str,
            k: int = 4,
        ) -> Tuple[str, List[Dict[str, Any]]]:
            """
            Retrieve relevant documents from the vector store based on a query.
            This tool searches through the document vector store to find the most 
            relevant documents matching the given query.
            
            Args:
                query: The search query string
                k: Number of documents to retrieve (default: 4)
                
            Returns:
                Tuple of (content, artifact):
                    - content: Formatted string with retrieved documents
                    - artifact: List of dictionaries with detailed document information
            """
            if description:
                retrieve_documents_tool.__doc__ = description
            return self.retrieve(query=query, k=k)
        
        # Set custom name if provided
        if tool_name:
            retrieve_documents_tool.name = tool_name
        
        return retrieve_documents_tool


if __name__ == "__main__":
    # Example usage
    try:
        from app.infra.vectorstore import get_vector_store
        
        query = "What is notice period of grade 8 employees?"
        logger.info(f"Testing retrieval with query: '{query}'")
        
        # Get vector store (example - in real usage, this would be done by the chatbot)
        vector_store = get_vector_store("hr")
        
        # Create retrieval service
        retrieval_service = RetrievalService(vector_store)
        
        # Use the service
        content, artifact = retrieval_service.retrieve(query=query, k=3)
        
        logger.info(f"\nRetrieved {len(artifact)} documents:")
        logger.info(f"Content:\n{content[:500] if content else 'No content'}...")
        
        for doc in artifact:
            logger.info(f"\nRank {doc['rank']}:")
            logger.info(f"Content preview: {doc['content'][:200]}...")
            logger.info(f"Metadata: {doc['metadata']}")
            
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
        logger.info("Make sure to run the vector store creation job first to create the ChromaDB vector store.")
