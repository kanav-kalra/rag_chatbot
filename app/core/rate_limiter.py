"""
Rate Limiting Middleware for API endpoints
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from threading import Lock
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.logger import logger


class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, consider using Redis-based rate limiting.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per IP
            requests_per_hour: Maximum requests per hour per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed.
        
        Args:
            identifier: Client identifier (IP address, user ID, etc.)
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = datetime.now()
        
        with self.lock:
            # Clean old entries
            self._cleanup_old_entries(identifier, now)
            
            # Check minute limit
            if len(self.minute_requests[identifier]) >= self.requests_per_minute:
                return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
            
            # Check hour limit
            if len(self.hour_requests[identifier]) >= self.requests_per_hour:
                return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
            
            # Record request
            self.minute_requests[identifier].append(now)
            self.hour_requests[identifier].append(now)
            
            return True, None
    
    def _cleanup_old_entries(self, identifier: str, now: datetime) -> None:
        """Remove old entries outside the time windows."""
        # Remove entries older than 1 minute
        cutoff_minute = now - timedelta(minutes=1)
        self.minute_requests[identifier] = [
            ts for ts in self.minute_requests[identifier]
            if ts > cutoff_minute
        ]
        
        # Remove entries older than 1 hour
        cutoff_hour = now - timedelta(hours=1)
        self.hour_requests[identifier] = [
            ts for ts in self.hour_requests[identifier]
            if ts > cutoff_hour
        ]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
            requests_per_hour: Maximum requests per hour per IP
            exclude_paths: List of paths to exclude from rate limiting
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute, requests_per_hour)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        is_allowed, error_msg = self.rate_limiter.is_allowed(client_ip)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {error_msg}")
            return Response(
                content=f'{{"detail": "{error_msg}"}}',
                status_code=429,
                media_type="application/json"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.rate_limiter.requests_per_hour)
        
        return response

