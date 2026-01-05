"""
Session Management for Chatbot - Handles multiple concurrent user sessions
Sessions are metadata containers only. Agents are fetched from agent pool at runtime.
Sessions are stored in Redis for multi-server support.
"""
import sys
from pathlib import Path
from typing import Dict, Optional
from threading import Lock
from datetime import datetime, timedelta
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.config.logging import logger
from src.shared.config.settings import settings
from src.infrastructure.storage.redis.session_storage import RedisSessionStorage


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
    Sessions are stored in Redis for multi-server support.
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
        self.session_timeout = session_timeout or timedelta(hours=24)
        self.max_sessions = max_sessions
        self.lock = Lock()
        
        # Initialize Redis storage (required)
        self._storage = RedisSessionStorage()
        
        logger.info(
            f"Initialized ChatbotSessionManager with "
            f"timeout={self.session_timeout}, max_sessions={self.max_sessions}"
        )
    
    def _session_to_dict(self, session: ChatbotSession) -> dict:
        """Convert session to dictionary for storage."""
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    
    def _dict_to_session(self, data: dict) -> ChatbotSession:
        """Convert dictionary to session object."""
        session = ChatbotSession(
            session_id=data["session_id"],
            user_id=data.get("user_id")
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        return session
    
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
            if self.max_sessions:
                current_count = self._get_session_count()
                if current_count >= self.max_sessions:
                    # Try to clean up expired sessions first
                    self._cleanup_expired_sessions()
                    current_count = self._get_session_count()
                    
                    if current_count >= self.max_sessions:
                        raise RuntimeError(
                            f"Maximum number of sessions ({self.max_sessions}) reached"
                        )
            
            # Create new session (metadata only, no agent reference)
            try:
                session = ChatbotSession(
                    session_id=session_id,
                    user_id=user_id
                )
                
                # Store in Redis via storage abstraction
                session_data = self._session_to_dict(session)
                ttl_seconds = int(self.session_timeout.total_seconds())
                self._storage.save_session(session_id, session_data, ttl_seconds)
                
                # Increment session count
                self._storage.increment_session_count()
                
                logger.info(
                    f"Created new session {session_id} (user_id={user_id})"
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
        try:
            session_data = self._storage.get_session(session_id)
            
            if session_data:
                session = self._dict_to_session(session_data)
                # Check if expired
                if session.is_expired(self.session_timeout):
                    # Delete expired session
                    self._storage.delete_session(session_id)
                    # Decrement session count
                    self._storage.decrement_session_count()
                    logger.info(f"Removed expired session {session_id}")
                    return None
                return session
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve session: {str(e)}") from e
    
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
                # Update activity and persist to Redis
                session.update_activity()
                self._save_session(session)
                return session
        
        return self.create_session(session_id, user_id)
    
    def _save_session(self, session: ChatbotSession) -> None:
        """
        Save session to storage.
        
        Args:
            session: ChatbotSession instance to save
        """
        try:
            session_data = self._session_to_dict(session)
            # Update TTL when saving (refresh expiration)
            ttl_seconds = int(self.session_timeout.total_seconds())
            self._storage.save_session(session.session_id, session_data, ttl_seconds)
        except Exception as e:
            logger.error(f"Failed to save session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save session: {str(e)}") from e
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session was deleted, False if not found
        """
        try:
            deleted = self._storage.delete_session(session_id)
            if deleted:
                # Decrement session count
                self._storage.decrement_session_count()
                logger.info(f"Deleted session {session_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to delete session: {str(e)}") from e
    
    def _cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        Note: Redis TTL handles automatic expiration, but this method checks for
        sessions that expired based on last_activity timestamp.
        
        Returns:
            Number of sessions removed
        """
        try:
            expired_count = 0
            
            # Get all session keys via storage
            session_keys = self._storage.scan_sessions()
            
            for key in session_keys:
                try:
                    # Extract session_id from key
                    session_id = self._storage.extract_session_id_from_key(key)
                    session_data = self._storage.get_session(session_id)
                    
                    if session_data:
                        session = self._dict_to_session(session_data)
                        if session.is_expired(self.session_timeout):
                            self._storage.delete_session(session_id)
                            # Decrement session count
                            self._storage.decrement_session_count()
                            expired_count += 1
                except Exception as e:
                    logger.warning(f"Error checking session expiration for {key}: {e}")
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")
            
            return expired_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}", exc_info=True)
            raise RuntimeError(f"Failed to cleanup expired sessions: {str(e)}") from e
    
    def cleanup_expired_sessions(self) -> int:
        """
        Public method to cleanup expired sessions.
        
        Returns:
            Number of sessions removed
        """
        return self._cleanup_expired_sessions()
    
    def _get_session_count(self) -> int:
        """
        Get current number of active sessions using storage counter.
        Note: Counter may drift slightly if sessions expire via TTL without going through
        our delete methods, but this is acceptable for performance.
        """
        try:
            return self._storage.get_session_count()
        except Exception as e:
            logger.error(f"Failed to get session count: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get session count: {str(e)}") from e
    
    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        return self._get_session_count()
    
    def get_all_sessions(self) -> Dict[str, ChatbotSession]:
        """Get all active sessions (for admin/debugging)."""
        try:
            sessions = {}
            session_keys = self._storage.scan_sessions()
            
            for key in session_keys:
                try:
                    # Extract session_id from key
                    session_id = self._storage.extract_session_id_from_key(key)
                    session_data = self._storage.get_session(session_id)
                    if session_data:
                        session = self._dict_to_session(session_data)
                        sessions[session.session_id] = session
                except Exception as e:
                    logger.warning(f"Error loading session from {key}: {e}")
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get all sessions: {str(e)}") from e
    
    def get_session_stats(self) -> dict:
        """Get statistics about sessions."""
        return {
            "total_sessions": self._get_session_count(),
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            "max_sessions": self.max_sessions,
            "storage": "Redis",
            "note": "Chat history is managed by checkpointer. Agents are fetched from agent pool at runtime."
        }

