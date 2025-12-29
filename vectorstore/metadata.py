"""
Vector Store Metadata - Hashes, doc state, versions
"""
from typing import Dict, Any, Optional
from datetime import datetime


class DocumentMetadata:
    """Metadata for documents in the vector store"""
    
    def __init__(
        self,
        doc_id: str,
        source: str,
        hash: Optional[str] = None,
        version: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.doc_id = doc_id
        self.source = source
        self.hash = hash
        self.version = version
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "hash": self.hash,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create metadata from dictionary"""
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            hash=data.get("hash"),
            version=data.get("version"),
            created_at=created_at,
            updated_at=updated_at
        )


def compute_document_hash(content: str) -> str:
    """
    Compute hash for document content.
    
    Args:
        content: Document content
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()

