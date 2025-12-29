"""
Session Management for Chatbot - Handles multiple concurrent user sessions
Sessions are metadata containers only. Agents are fetched from agent pool at runtime.
"""
import sys
from pathlib import Path
from typing import Dict, Optional
from threading import Lock
from datetime import datetime, timedelta
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.logger import logger
from app.core.config import settings


class ChatbotSession:
    """
    Represents a single user's chat session.
    Manages session metadata only (chat history is managed by checkpointer).
    Agents are fetched from agent pool at runtime by API functions.
    """
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        """
        Initialize a chatbot session.
        
        Args:
            session_id: Unique session identifier (used as thread_id for checkpointer)
            user_id: Optional user identifier
        """
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout: timedelta) -> bool:
        """
        Check if session has expired.
        
        Args:
            timeout: Session timeout duration
        
        Returns:
            True if session is expired
        """
        return datetime.now() - self.last_activity > timeout
    
    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "note": "Chat history is managed by checkpointer (thread_id = session_id)"
        }


class ChatbotSessionManager:
    """
    Manages multiple user chat sessions.
    Sessions are metadata containers only. Agents are fetched from agent pool at runtime.
    Thread-safe implementation for production use.
    Designed for multi-server, multi-user deployments.
    """
    
    def __init__(
        self,
        session_timeout: Optional[timedelta] = None,
        max_sessions: Optional[int] = None
    ):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Time after which inactive sessions expire (default: 24 hours)
            max_sessions: Maximum number of concurrent sessions (None for unlimited)
        """
        self.sessions: Dict[str, ChatbotSession] = {}
        self.lock = Lock()
        self.session_timeout = session_timeout or timedelta(hours=24)
        self.max_sessions = max_sessions
        
        logger.info(
            f"Initialized ChatbotSessionManager with "
            f"timeout={self.session_timeout}, max_sessions={self.max_sessions}"
        )
    
    @property
    def max_sessions(self) -> Optional[int]:
        """Get max sessions limit."""
        return self._max_sessions
    
    @max_sessions.setter
    def max_sessions(self, value: Optional[int]):
        """Set max sessions limit."""
        self._max_sessions = value
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ChatbotSession:
        """
        Create a new chat session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            user_id: Optional user identifier
        
        Returns:
            New ChatbotSession instance
        
        Raises:
            RuntimeError: If max sessions limit is reached
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with self.lock:
            # Check max sessions limit
            if self.max_sessions and len(self.sessions) >= self.max_sessions:
                # Try to clean up expired sessions first
                self._cleanup_expired_sessions()
                
                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError(
                        f"Maximum number of sessions ({self.max_sessions}) reached"
                    )
            
            # Create new session (metadata only, no agent reference)
            try:
                session = ChatbotSession(
                    session_id=session_id,
                    user_id=user_id
                )
                self.sessions[session_id] = session
                logger.info(
                    f"Created new session {session_id} "
                    f"(user_id={user_id}, total_sessions={len(self.sessions)})"
                )
                return session
            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
                raise
    
    def get_session(self, session_id: str) -> Optional[ChatbotSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            ChatbotSession if found, None otherwise
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session and session.is_expired(self.session_timeout):
                # Session expired, remove it
                del self.sessions[session_id]
                logger.info(f"Removed expired session {session_id}")
                return None
            return session
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ChatbotSession:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session ID
            user_id: Optional user identifier
        
        Returns:
            ChatbotSession instance
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(session_id, user_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session was deleted, False if not found
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False
    
    def _cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        with self.lock:
            now = datetime.now()
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(self.session_timeout)
            ]
            
            for sid in expired:
                del self.sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
            
            return len(expired)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Public method to cleanup expired sessions.
        
        Returns:
            Number of sessions removed
        """
        return self._cleanup_expired_sessions()
    
    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        with self.lock:
            return len(self.sessions)
    
    def get_all_sessions(self) -> Dict[str, ChatbotSession]:
        """Get all active sessions (for admin/debugging)."""
        with self.lock:
            return self.sessions.copy()
    
    def get_session_stats(self) -> dict:
        """Get statistics about sessions."""
        with self.lock:
            return {
                "total_sessions": len(self.sessions),
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
                "max_sessions": self.max_sessions,
                "note": "Chat history is managed by checkpointer. Agents are fetched from agent pool at runtime."
            }

